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
    private radixInitPipeline!: GPUComputePipeline;
    private radixCountPipeline!: GPUComputePipeline;
    private radixScanPipeline!: GPUComputePipeline;
    private radixScatterPipeline!: GPUComputePipeline;
    private readonly initPipeline: GPUComputePipeline;
    private initBindGroup: GPUBindGroup;
    private sortBindGroupAtoB!: GPUBindGroup;
    private sortBindGroupBtoA!: GPUBindGroup;
    
    private histBuffer!: GPUBuffer;

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

        // WG_SIZE = 256. Number of workgroups W = ceil(numSplats / 256).
        // histBuffer stores atomic<u32> for 256 buckets across W workgroups.
        const W = Math.ceil(this.numSplats / 256);
        this.histBuffer = device.createBuffer({
            label: "splat sort histogram",
            size: 256 * W * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        this.sortKeysBuffer = this.sortKeysBufferA;
        this.sortIndicesBuffer = this.sortIndicesBufferA;

        // Sort uniforms: VP (64) + shift (4) + pad (12) = 80 bytes.
        // We use 4 dynamic offsets (0, 256, 512, 768) for the 4 passes (shift = 0, 8, 16, 24).
        this.sortUniformsBuffer = device.createBuffer({
            label: "splat sort uniforms",
            size: 1024,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        for (let i = 0; i < 4; i++) {
            device.queue.writeBuffer(this.sortUniformsBuffer, i * 256 + 64, new Uint32Array([i * 8, 0, 0, 0]));
        }

        const injectConstants = (src: string) => {
            return injectWgslConstants(src, {
                ...constants,
                NUM_SPLATS: this.numSplats,
                NUM_SPLATS_PLUS_ONE: this.numSplats + 1,
                NUM_SPLATS_MINUS_ONE: this.numSplats - 1,
                NUM_SPLATS_DIV_32: Math.ceil(this.numSplats / 32),
                NUM_PARAMS: this.numParams,
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
        const sortBindGroupLayout = device.createBindGroupLayout({
            label: "splat sort bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // splats
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // in_keys
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // in_indices
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // out_keys
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // out_indices
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } }, // sort_uniforms
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // hist
            ],
        });

        const sortModule = device.createShaderModule({ label: "splat sort", code: injectConstants(sortModuleSrc) });
        sortModule.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[splat_sort] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
        });

        const sortLayout = device.createPipelineLayout({
            label: "splat sort pipeline layout",
            bindGroupLayouts: [sortBindGroupLayout]
        });

        this.radixInitPipeline = device.createComputePipeline({
            label: "splat sort init_keys pipeline",
            layout: sortLayout,
            compute: { module: sortModule, entryPoint: "init_keys" },
        });

        this.radixCountPipeline = device.createComputePipeline({
            label: "splat sort count pipeline",
            layout: sortLayout,
            compute: { module: sortModule, entryPoint: "count" },
        });

        this.radixScanPipeline = device.createComputePipeline({
            label: "splat sort scan pipeline",
            layout: sortLayout,
            compute: { module: sortModule, entryPoint: "scan" },
        });

        this.radixScatterPipeline = device.createComputePipeline({
            label: "splat sort scatter pipeline",
            layout: sortLayout,
            compute: { module: sortModule, entryPoint: "scatter" },
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

        this.sortBindGroupAtoB = device.createBindGroup({
            label: "splat sort bind group A to B",
            layout: sortBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.splatBuffer } },
                { binding: 1, resource: { buffer: this.sortKeysBufferA } },
                { binding: 2, resource: { buffer: this.sortIndicesBufferA } },
                { binding: 3, resource: { buffer: this.sortKeysBufferB } },
                { binding: 4, resource: { buffer: this.sortIndicesBufferB } },
                { binding: 5, resource: { buffer: this.sortUniformsBuffer, size: 80 } },
                { binding: 6, resource: { buffer: this.histBuffer } },
            ],
        });

        this.sortBindGroupBtoA = device.createBindGroup({
            label: "splat sort bind group B to A",
            layout: sortBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.splatBuffer } },
                { binding: 1, resource: { buffer: this.sortKeysBufferB } },
                { binding: 2, resource: { buffer: this.sortIndicesBufferB } },
                { binding: 3, resource: { buffer: this.sortKeysBufferA } },
                { binding: 4, resource: { buffer: this.sortIndicesBufferA } },
                { binding: 5, resource: { buffer: this.sortUniformsBuffer, size: 80 } },
                { binding: 6, resource: { buffer: this.histBuffer } },
            ],
        });
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

        for (let i = 0; i < 4; i++) {
            this.device.queue.writeBuffer(
                this.sortUniformsBuffer, i * 256,
                vpData.buffer, vpData.byteOffset, vpData.byteLength,
            );
        }

        const wg = Math.ceil(this.numSplats / 256);

        // Pass 0: compute depth keys + init indices
        const initPass = commandEncoder.beginComputePass({ label: "splat sort init pass" });
        initPass.setPipeline(this.radixInitPipeline);
        // Bind to AtoB to write to Buffer A. Uniform offset 0 (shift 0).
        initPass.setBindGroup(0, this.sortBindGroupAtoB, [0]);
        initPass.dispatchWorkgroups(wg);
        initPass.end();

        // 4 passes of 8-bit Radix Sort
        for (let i = 0; i < 4; i++) {
            const bg = (i % 2 === 0) ? this.sortBindGroupAtoB : this.sortBindGroupBtoA;
            const offset = i * 256;

            const countPass = commandEncoder.beginComputePass({ label: `splat sort count ${i}` });
            countPass.setPipeline(this.radixCountPipeline);
            countPass.setBindGroup(0, bg, [offset]);
            countPass.dispatchWorkgroups(wg);
            countPass.end();

            const scanPass = commandEncoder.beginComputePass({ label: `splat sort scan ${i}` });
            scanPass.setPipeline(this.radixScanPipeline);
            scanPass.setBindGroup(0, bg, [offset]);
            scanPass.dispatchWorkgroups(1);
            scanPass.end();

            const scatterPass = commandEncoder.beginComputePass({ label: `splat sort scatter ${i}` });
            scatterPass.setPipeline(this.radixScatterPipeline);
            scatterPass.setBindGroup(0, bg, [offset]);
            scatterPass.dispatchWorkgroups(wg);
            scatterPass.end();
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
        this.histBuffer.destroy();
    }
}
