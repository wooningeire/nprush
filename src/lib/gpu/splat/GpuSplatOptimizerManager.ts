import backwardModuleSrc from "./splat_backward.wgsl?raw";
import stepModuleSrc from "./splat_step.wgsl?raw";
import renderModuleSrc from "./splat_render.wgsl?raw";
import adcModuleSrc from "./splat_adc.wgsl?raw";
import edgeModuleSrc from "./splat_edge.wgsl?raw";
import sortModuleSrc from "./splat_sort.wgsl?raw";
import initModuleSrc from "./splat_init.wgsl?raw";
import type { Mat4 } from "wgpu-matrix";
import { constants, injectWgslConstants } from "../constants";
import { nextPowerOfTwoAtLeast } from "../nextPowerOfTwoAtLeast";

export class GpuSplatOptimizerManager {
    private readonly device: GPUDevice;
    
    readonly numSplats: number;
    readonly numParams: number;

    readonly splatBuffer: GPUBuffer;
    readonly gradBuffer: GPUBuffer;
    readonly adamBuffer: GPUBuffer;
    readonly adcBuffer: GPUBuffer;
    readonly renderUniformsBuffer: GPUBuffer;
    readonly splatUniformsBuffer: GPUBuffer;

    // Depth sort buffers
    readonly sortKeysBufferA: GPUBuffer;
    readonly sortKeysBufferB: GPUBuffer;
    readonly sortIndicesBufferA: GPUBuffer;
    readonly sortIndicesBufferB: GPUBuffer;
    readonly sortKeysBuffer: GPUBuffer;
    readonly sortIndicesBuffer: GPUBuffer;
    readonly sortUniformsBuffer: GPUBuffer;

    private readonly backwardPipeline: GPUComputePipeline;
    private readonly stepPipeline: GPUComputePipeline;
    private readonly adcPipeline: GPUComputePipeline;
    private readonly edgePipeline: GPUComputePipeline;
    private readonly renderPipeline: GPURenderPipeline;
    private readonly sortInitPipeline: GPUComputePipeline;
    private readonly sortStepPipeline: GPUComputePipeline;
    private readonly initPipeline: GPUComputePipeline;
    private sortInitBindGroup: GPUBindGroup;
    private initBindGroup: GPUBindGroup;
    private sortStepBindGroupAtoB!: GPUBindGroup;
    private sortStepBindGroupBtoA!: GPUBindGroup;
    private numSortSteps: number = 0;
    private sortN: number = 0;

    private backwardBindGroupLayout: GPUBindGroupLayout;
    private stepBindGroupLayout: GPUBindGroupLayout;
    private edgeBindGroupLayout: GPUBindGroupLayout;
    private renderBindGroupLayout: GPUBindGroupLayout;

    private backwardBindGroup!: GPUBindGroup;
    private stepBindGroup: GPUBindGroup;
    private adcBindGroup: GPUBindGroup;
    private edgeBindGroup!: GPUBindGroup;
    private renderBindGroup!: GPUBindGroup;
    
    private stepCount: number = 0;

    private dims: { width: number, height: number } = { width: 0, height: 0 };
    /** Last optim pixel count written into AdamState; avoids redundant queue writes each dispatch. */
    private cachedAdamPixelCount: number | null = null;

    constructor({
        device,
        format,
        numSplats = 512,
        numBeziers,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        numSplats?: number,
        numBeziers?: number,
    }) {
        this.device = device;
        this.numSplats = numSplats;
        const floatsPer = constants.SPLAT_PARAMS_PER_SPLAT;
        this.numParams = numSplats * floatsPer;

        this.splatBuffer = device.createBuffer({
            label: "splat buffer",
            size: this.numSplats * floatsPer * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.gradBuffer = device.createBuffer({
            label: "splat grad buffer",
            size: this.numParams * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // m (numParams * 4) + v (numParams * 4) + t (4) + pixel_count (4) + pad (8) + extra padding (16)
        this.adamBuffer = device.createBuffer({
            label: "splat adam buffer",
            size: this.numParams * 8 + 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.adcBuffer = device.createBuffer({
            label: "splat adc buffer",
            size: this.numSplats * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.renderUniformsBuffer = device.createBuffer({
            label: "splat render uniforms buffer",
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // VP (64) + inv VP (64) + cam_world (16) + extras.x = blur flag (vec4 tail = 16) — 160 B
        this.splatUniformsBuffer = device.createBuffer({
            label: "splat VP uniforms buffer",
            size: 160,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Sort buffers
        this.sortKeysBufferA = device.createBuffer({
            label: "splat sort keys A",
            size: this.numSplats * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        this.sortKeysBufferB = device.createBuffer({
            label: "splat sort keys B",
            size: this.numSplats * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        this.sortIndicesBufferA = device.createBuffer({
            label: "splat sort indices A",
            size: this.numSplats * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        this.sortIndicesBufferB = device.createBuffer({
            label: "splat sort indices B",
            size: this.numSplats * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        const sortN = nextPowerOfTwoAtLeast(this.numSplats);
        this.sortN = sortN;
        const logN = Math.log2(this.sortN);
        const numSteps = (logN * (logN + 1)) / 2;
        const finalInA = numSteps % 2 === 0;

        this.sortKeysBuffer = finalInA ? this.sortKeysBufferA : this.sortKeysBufferB;
        this.sortIndicesBuffer = finalInA ? this.sortIndicesBufferA : this.sortIndicesBufferB;

        // Sort uniforms: VP (64) + block_k (4) + sub_k (4) + pad (8) = 80
        // Using dynamic offsets aligned to 256 bytes.
        // Chunk 0 is used for init_keys (VP matrix). Chunks 1..numSteps are for sort_step.
        this.sortUniformsBuffer = device.createBuffer({
            label: "splat sort uniforms",
            size: 256 * (numSteps + 1),
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const injectConstants = (src: string) => {
            return injectWgslConstants(src, {
                ...constants,
                NUM_SPLATS: this.numSplats,
                NUM_SPLATS_PLUS_ONE: this.numSplats + 1,
                NUM_SPLATS_MINUS_ONE: this.numSplats - 1,
                NUM_SPLATS_DIV_32: Math.ceil(this.numSplats / 32),
                NUM_PARAMS: this.numParams,
                SORT_N: sortN,
            });
        };

        // Backward Pipeline — now has 6 bindings (splats, grads, target, depth, VP uniform, sort order)
        this.backwardBindGroupLayout = device.createBindGroupLayout({
            label: "splat backward bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            ],
        });
        const backwardModule = device.createShaderModule({ label: "splat backward", code: injectConstants(backwardModuleSrc) });
        backwardModule.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[splat_backward] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
        });
        this.backwardPipeline = device.createComputePipeline({
            label: "splat backward pipeline",
            layout: device.createPipelineLayout({ 
                label: "splat backward pipeline layout",
                bindGroupLayouts: [this.backwardBindGroupLayout] 
            }),
            compute: { module: backwardModule, entryPoint: "main" },
        });

        // Step Pipeline
        this.stepBindGroupLayout = device.createBindGroupLayout({
            label: "splat step bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
        });
        const stepModule = device.createShaderModule({ label: "splat step", code: injectConstants(stepModuleSrc) });
        stepModule.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[splat_step] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
        });
        this.stepPipeline = device.createComputePipeline({
            label: "splat step pipeline",
            layout: device.createPipelineLayout({ 
                label: "splat step pipeline layout",
                bindGroupLayouts: [this.stepBindGroupLayout] 
            }),
            compute: { module: stepModule, entryPoint: "main" },
        });

        this.stepBindGroup = device.createBindGroup({
            label: "splat step bind group",
            layout: this.stepBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.splatBuffer } },
                { binding: 1, resource: { buffer: this.gradBuffer } },
                { binding: 2, resource: { buffer: this.adamBuffer } },
                { binding: 3, resource: { buffer: this.adcBuffer } },
                { binding: 4, resource: { buffer: this.splatUniformsBuffer } },
            ],
        });

        // ADC Pipeline
        const adcBindGroupLayout = device.createBindGroupLayout({
            label: "splat adc bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ],
        });
        const adcModule = device.createShaderModule({ label: "splat adc", code: injectConstants(adcModuleSrc) });
        adcModule.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[splat_adc] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
        });
        this.adcPipeline = device.createComputePipeline({
            label: "splat adc pipeline",
            layout: device.createPipelineLayout({ 
                label: "splat adc pipeline layout",
                bindGroupLayouts: [adcBindGroupLayout] 
            }),
            compute: { module: adcModule, entryPoint: "main" },
        });

        this.adcBindGroup = device.createBindGroup({
            label: "splat adc bind group",
            layout: adcBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.splatBuffer } },
                { binding: 1, resource: { buffer: this.adamBuffer } },
                { binding: 2, resource: { buffer: this.adcBuffer } },
            ],
        });

        // Edge Detection Pipeline
        this.edgeBindGroupLayout = device.createBindGroupLayout({
            label: "splat edge bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba8unorm" } },
            ],
        });
        const edgeModule = device.createShaderModule({ label: "splat edge", code: injectConstants(edgeModuleSrc) });
        edgeModule.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[splat_edge] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
        });
        this.edgePipeline = device.createComputePipeline({
            label: "splat edge pipeline",
            layout: device.createPipelineLayout({ 
                label: "splat edge pipeline layout",
                bindGroupLayouts: [this.edgeBindGroupLayout] 
            }),
            compute: { module: edgeModule, entryPoint: "main" },
        });

        const renderModule = device.createShaderModule({ label: "splat render", code: injectConstants(renderModuleSrc) });
        renderModule.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[splat_render] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
        });
        
        this.renderBindGroupLayout = device.createBindGroupLayout({
            label: "splat render bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
            ],
        });

        this.renderPipeline = device.createRenderPipeline({
            label: "splat render pipeline",
            layout: device.createPipelineLayout({ 
                label: "splat render pipeline layout",
                bindGroupLayouts: [this.renderBindGroupLayout] 
            }),
            vertex: { module: renderModule, entryPoint: "vert" },
            fragment: { module: renderModule, entryPoint: "frag", targets: [{ format }] },
            primitive: { topology: "triangle-list" },
        });
        // Sort pipelines
        const sortBindGroupLayout = device.createBindGroupLayout({
            label: "splat sort bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
            ],
        });
        const sortModule = device.createShaderModule({ label: "splat sort", code: injectConstants(sortModuleSrc) });
        sortModule.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[splat_sort] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
        });
        this.sortInitPipeline = device.createComputePipeline({
            label: "splat sort init pipeline",
            layout: device.createPipelineLayout({
                label: "splat sort init pipeline layout",
                bindGroupLayouts: [sortBindGroupLayout],
            }),
            compute: { module: sortModule, entryPoint: "init_keys" },
        });
        this.sortStepPipeline = device.createComputePipeline({
            label: "splat sort step pipeline",
            layout: device.createPipelineLayout({
                label: "splat sort step pipeline layout",
                bindGroupLayouts: [sortBindGroupLayout],
            }),
            compute: { module: sortModule, entryPoint: "sort_step" },
        });

        const initBindGroupLayout = device.createBindGroupLayout({
            label: "splat init bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ],
        });
        const initModule = device.createShaderModule({
            label: "splat init",
            code: injectConstants(initModuleSrc),
        });
        initModule.getCompilationInfo().then(info => {
            for (const m of info.messages) console.warn(`[splat_init] ${m.type}: ${m.message} (line ${m.lineNum})`);
        });
        this.initPipeline = device.createComputePipeline({
            label: "splat init pipeline",
            layout: device.createPipelineLayout({
                label: "splat init pipeline layout",
                bindGroupLayouts: [initBindGroupLayout],
            }),
            compute: { module: initModule, entryPoint: "main" },
        });
        this.initBindGroup = device.createBindGroup({
            label: "splat init bind group",
            layout: initBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.splatBuffer } },
            ],
        });

        // Run initialization pass immediately
        const initEncoder = device.createCommandEncoder({ label: "splat init encoder" });
        const initPass = initEncoder.beginComputePass({ label: "splat init pass" });
        initPass.setPipeline(this.initPipeline);
        initPass.setBindGroup(0, this.initBindGroup);
        initPass.dispatchWorkgroups(Math.ceil(this.numSplats / 64));
        initPass.end();
        device.queue.submit([initEncoder.finish()]);

        // Pre-create init bind group (uses sortUniformsBuffer for VP only at offset 0)
        // Note: init pass only writes to A. in_keys/in_indices are unused by init shader, but we bind them to A as dummies.
        this.sortInitBindGroup = device.createBindGroup({
            label: "splat sort init bind group",
            layout: sortBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.splatBuffer } },
                { binding: 1, resource: { buffer: this.sortKeysBufferB } },
                { binding: 2, resource: { buffer: this.sortIndicesBufferB } },
                { binding: 3, resource: { buffer: this.sortKeysBufferA } },
                { binding: 4, resource: { buffer: this.sortIndicesBufferA } },
                { binding: 5, resource: { buffer: this.sortUniformsBuffer, size: 80 } },
            ],
        });

        this.sortStepBindGroupAtoB = device.createBindGroup({
            label: "splat sort step bind group A to B",
            layout: sortBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.splatBuffer } },
                { binding: 1, resource: { buffer: this.sortKeysBufferA } },
                { binding: 2, resource: { buffer: this.sortIndicesBufferA } },
                { binding: 3, resource: { buffer: this.sortKeysBufferB } },
                { binding: 4, resource: { buffer: this.sortIndicesBufferB } },
                { binding: 5, resource: { buffer: this.sortUniformsBuffer, size: 80 } },
            ],
        });

        this.sortStepBindGroupBtoA = device.createBindGroup({
            label: "splat sort step bind group B to A",
            layout: sortBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.splatBuffer } },
                { binding: 1, resource: { buffer: this.sortKeysBufferB } },
                { binding: 2, resource: { buffer: this.sortIndicesBufferB } },
                { binding: 3, resource: { buffer: this.sortKeysBufferA } },
                { binding: 4, resource: { buffer: this.sortIndicesBufferA } },
                { binding: 5, resource: { buffer: this.sortUniformsBuffer, size: 80 } },
            ],
        });

        // Pre-write all block_k and sub_k step params at their dynamic offsets
        let stepIdx = 0;
        for (let block_k = 1; block_k <= logN; block_k++) {
            for (let sub_k = block_k - 1; sub_k >= 0; sub_k--) {
                const offset = 256 * (stepIdx + 1); // +1 because offset 0 is for init
                device.queue.writeBuffer(this.sortUniformsBuffer, offset + 64, new Uint32Array([block_k, sub_k, 0, 0]));
                stepIdx++;
            }
        }
        this.numSortSteps = stepIdx;
    }

    writeSplatVPMatrix(
        mat: Mat4,
        invMat: Mat4,
        blurEnabled: boolean = false,
        camWorldXYZ: readonly [number, number, number] | Float32Array,
    ) {
        this.device.queue.writeBuffer(
            this.splatUniformsBuffer,
            0,
            (mat as Float32Array).buffer,
            (mat as Float32Array).byteOffset,
            (mat as Float32Array).byteLength
        );
        this.device.queue.writeBuffer(
            this.splatUniformsBuffer,
            64,
            (invMat as Float32Array).buffer,
            (invMat as Float32Array).byteOffset,
            (invMat as Float32Array).byteLength
        );
        this.device.queue.writeBuffer(
            this.splatUniformsBuffer,
            128,
            new Float32Array([
                camWorldXYZ[0],
                camWorldXYZ[1],
                camWorldXYZ[2],
                1.0,
                blurEnabled ? 1 : 0,
                0,
                0,
                0,
            ])
        );
    }

    setBackwardTarget(targetTextureView: GPUTextureView, targetDepthTextureView: GPUTextureView, width: number, height: number) {
        this.dims = { width, height };
        
        this.backwardBindGroup = this.device.createBindGroup({
            label: "splat backward bind group",
            layout: this.backwardBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.splatBuffer } },
                { binding: 1, resource: { buffer: this.gradBuffer } },
                { binding: 2, resource: targetTextureView },
                { binding: 3, resource: targetDepthTextureView }, // actual depth, not edge
                { binding: 4, resource: { buffer: this.splatUniformsBuffer } },
                { binding: 5, resource: { buffer: this.sortIndicesBuffer } },
            ],
        });
    }

    setEdgeTarget(depthTextureView: GPUTextureView, edgeTextureView: GPUTextureView) {
        this.edgeBindGroup = this.device.createBindGroup({
            label: "splat edge bind group",
            layout: this.edgeBindGroupLayout,
            entries: [
                { binding: 0, resource: depthTextureView },
                { binding: 1, resource: edgeTextureView },
            ],
        });
    }

    setRenderTarget(
        targetTextureView: GPUTextureView,
        splatViewTextureView: GPUTextureView,
        depthTextureView: GPUTextureView,
        edgeTextureView: GPUTextureView,
        bezierViewTextureView: GPUTextureView,
        baseColorBezierViewTextureView: GPUTextureView,
        colorBezierViewTextureView: GPUTextureView,
        ptTextureView: GPUTextureView,
    ) {
        this.renderBindGroup = this.device.createBindGroup({
            label: "splat render bind group",
            layout: this.renderBindGroupLayout,
            entries: [
                { binding: 0, resource: targetTextureView },
                { binding: 1, resource: splatViewTextureView },
                { binding: 2, resource: depthTextureView },
                { binding: 3, resource: edgeTextureView },
                { binding: 4, resource: bezierViewTextureView },
                { binding: 5, resource: baseColorBezierViewTextureView },
                { binding: 6, resource: colorBezierViewTextureView },
                { binding: 7, resource: { buffer: this.renderUniformsBuffer } },
                { binding: 8, resource: ptTextureView },
            ],
        });
    }

    writeRenderUniforms(edgeEnabled: boolean, baseColorEnabled: boolean, colorEnabled: boolean, meshSplatsEnabled: boolean, splatsEnabled: boolean) {
        this.device.queue.writeBuffer(
            this.renderUniformsBuffer,
            0,
            new Float32Array([edgeEnabled ? 1 : 0, baseColorEnabled ? 1 : 0, colorEnabled ? 1 : 0, meshSplatsEnabled ? 1 : 0, splatsEnabled ? 1 : 0])
        );
    }

    /**
     * Reset Adam momentum (m, v) and step counter (t) without touching the
     * splat parameters themselves. Call this whenever the camera changes
     * during turntable training so stale cross-view momentum doesn't corrupt
     * the gradient step for the new viewpoint.
     */
    resetAdam() {
        // adamBuffer layout: m[numParams * f32] | v[numParams * f32] | t(f32) | pixel_count(f32) | pad(8)
        // Zero m and v, reset t to 0. pixel_count is written each dispatch so
        // we don't need to preserve it, but we invalidate the cache so it gets
        // re-written on the next dispatch.
        this.device.queue.writeBuffer(
            this.adamBuffer,
            0,
            new Float32Array(this.numParams * 2 + 1) // m + v + t, all zeros
        );
        this.cachedAdamPixelCount = null;
        // Also reset ADC grad_accum so stale positional gradient norms from the
        // previous view don't trigger spurious clone/kill decisions.
        this.device.queue.writeBuffer(
            this.adcBuffer,
            0,
            new Float32Array(this.numSplats)
        );
    }

    writeNoKill(noKill: boolean) {
        // adamBuffer layout: m[N*4] | v[N*4] | t(4) | pixel_count(4) | no_kill(4) | pad(4)
        this.device.queue.writeBuffer(
            this.adamBuffer,
            this.numParams * 8 + 8, // offset after t and pixel_count
            new Float32Array([noKill ? 1 : 0])
        );
    }

    dispatch(commandEncoder: GPUCommandEncoder, timestampWrites?: NonNullable<GPUComputePassDescriptor["timestampWrites"]>) {
        if (!this.backwardBindGroup) return;

        // Update pixel count for gradient normalization in the step shader.
        // AdamState layout: m [N], v [N], t [1], pixel_count [1], pad [2]
        const pixelCount = this.dims.width * this.dims.height;
        if (this.cachedAdamPixelCount !== pixelCount) {
            this.cachedAdamPixelCount = pixelCount;
            this.device.queue.writeBuffer(
                this.adamBuffer,
                this.numParams * 8 + 4,
                new Float32Array([pixelCount])
            );
        }

        const pass = commandEncoder.beginComputePass({
            label: "splat backward and step pass",
            ...(timestampWrites ? { timestampWrites } : {}),
        });
        
        pass.setPipeline(this.backwardPipeline);
        pass.setBindGroup(0, this.backwardBindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.dims.width / 16), Math.ceil(this.dims.height / 16));
        
        pass.setPipeline(this.stepPipeline);
        pass.setBindGroup(0, this.stepBindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.numSplats / 64));
        
        this.stepCount++;
        if (this.stepCount % constants.SPLAT_ADC_PERIOD === 0) {
            pass.setPipeline(this.adcPipeline);
            pass.setBindGroup(0, this.adcBindGroup);
            pass.dispatchWorkgroups(1);
        }
        
        pass.end();
    }

    dispatchEdge(commandEncoder: GPUCommandEncoder, width: number, height: number, timestampWrites?: NonNullable<GPUComputePassDescriptor["timestampWrites"]>) {
        if (!this.edgeBindGroup) return;
        
        const pass = commandEncoder.beginComputePass({
            label: "splat edge pass",
            ...(timestampWrites ? { timestampWrites } : {}),
        });
        pass.setPipeline(this.edgePipeline);
        pass.setBindGroup(0, this.edgeBindGroup);
        pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
        pass.end();
    }

    addDraw(renderPassEncoder: GPURenderPassEncoder) {
        if (!this.renderBindGroup) return;
        renderPassEncoder.setPipeline(this.renderPipeline);
        renderPassEncoder.setBindGroup(0, this.renderBindGroup);
        renderPassEncoder.draw(6);
    }

    /**
     * Run a full depth sort of all splats using the current VP matrix.
     * Writes the sort order into sortIndicesBuffer (back-to-front).
     */
    dispatchSort(commandEncoder: GPUCommandEncoder, vpMat: Mat4) {
        const vpData = vpMat as Float32Array;

        // Write VP matrix to the init uniform buffer
        this.device.queue.writeBuffer(
            this.sortUniformsBuffer, 0,
            vpData.buffer, vpData.byteOffset, vpData.byteLength,
        );

        const wg = Math.ceil(this.sortN / 256);

        // Pass 0: compute depth keys + init indices
        const initPass = commandEncoder.beginComputePass({ label: "splat sort init pass" });
        initPass.setPipeline(this.sortInitPipeline);
        initPass.setBindGroup(0, this.sortInitBindGroup, [0]);
        initPass.dispatchWorkgroups(wg);
        initPass.end();

        // Bitonic merge sort steps
        const stepWg = Math.ceil(this.sortN / 2 / 256);
        for (let i = 0; i < this.numSortSteps; i++) {
            const stepPass = commandEncoder.beginComputePass({ label: `splat sort step ${i}` });
            stepPass.setPipeline(this.sortStepPipeline);
            const bg = (i % 2 === 0) ? this.sortStepBindGroupAtoB : this.sortStepBindGroupBtoA;
            stepPass.setBindGroup(0, bg, [256 * (i + 1)]);
            stepPass.dispatchWorkgroups(stepWg);
            stepPass.end();
        }
    }

    destroy() {
        this.splatBuffer.destroy();
        this.gradBuffer.destroy();
        this.adamBuffer.destroy();
        this.adcBuffer.destroy();
        this.renderUniformsBuffer.destroy();
        this.splatUniformsBuffer.destroy();
        this.sortKeysBufferA.destroy();
        this.sortKeysBufferB.destroy();
        this.sortIndicesBufferA.destroy();
        this.sortIndicesBufferB.destroy();
        this.sortUniformsBuffer.destroy();
    }
}
