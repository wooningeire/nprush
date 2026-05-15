import backwardModuleSrc from "./splat_backward.wgsl?raw";
import stepModuleSrc from "./splat_step.wgsl?raw";
import renderBlitModuleSrc from "./render_blit.wgsl?raw";
import renderCompositeModuleSrc from "./render_composite.wgsl?raw";
import adcModuleSrc from "./splat_adc.wgsl?raw";
import edgeModuleSrc from "./splat_edge.wgsl?raw";
import sortModuleSrc from "./splat_sort.wgsl?raw";
import initModuleSrc from "./splat_init.wgsl?raw";
import binningModuleSrc from "./splat_binning.wgsl?raw";
import type { Mat4 } from "wgpu-matrix";
import { constants, injectWgslConstants } from "../constants";

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
    private readonly targetPipeline: GPURenderPipeline;
    private readonly compositePipeline: GPURenderPipeline;
    private readonly blitPipeline: GPURenderPipeline;
    private readonly blitRPipeline: GPURenderPipeline;
    private readonly blitAPipeline: GPURenderPipeline;
    private readonly radixInitPipeline: GPUComputePipeline;
    private readonly radixCountPipeline: GPUComputePipeline;
    private readonly radixScanPipeline: GPUComputePipeline;
    private readonly radixScatterPipeline: GPUComputePipeline;
    private readonly initPipeline: GPUComputePipeline;
    private initBindGroup: GPUBindGroup;
    private readonly sortBindGroupAtoB: GPUBindGroup;
    private readonly sortBindGroupBtoA: GPUBindGroup;
    private readonly histBuffer: GPUBuffer;

    // Binning pre-pass buffers
    private readonly binningAtomicBuffer: GPUBuffer;
    private readonly instanceKeysBufferA: GPUBuffer;
    private readonly instanceKeysBufferB: GPUBuffer;
    private readonly instanceValsBufferA: GPUBuffer;
    private readonly instanceValsBufferB: GPUBuffer;
    private readonly binningUniformsBuffer: GPUBuffer;
    private readonly binningSortUniformsBuffer: GPUBuffer;
    private readonly binningHistBuffer: GPUBuffer;
    private readonly tileStartsBuffer: GPUBuffer;
    private readonly tileEndsBuffer: GPUBuffer;

    // Binning pipelines
    private readonly binInstantiatePipeline: GPUComputePipeline;
    private readonly binCountPipeline: GPUComputePipeline;
    private readonly binScanPipeline: GPUComputePipeline;
    private readonly binScatterPipeline: GPUComputePipeline;
    private readonly binCalcRangesPipeline: GPUComputePipeline;

    // Binning bind groups
    private readonly binInstantiateBindGroup: GPUBindGroup;
    private readonly binSortBindGroupAtoB: GPUBindGroup;
    private readonly binSortBindGroupBtoA: GPUBindGroup;
    private readonly binCalcRangesBindGroup: GPUBindGroup;

    private readonly maxInstances: number;

    private backwardBindGroupLayout: GPUBindGroupLayout;
    private stepBindGroupLayout: GPUBindGroupLayout;
    private edgeBindGroupLayout: GPUBindGroupLayout;
    private renderBindGroupLayout: GPUBindGroupLayout;
    private blitBindGroupLayout: GPUBindGroupLayout;

    private backwardBindGroup!: GPUBindGroup;
    private stepBindGroup: GPUBindGroup;
    private adcBindGroup: GPUBindGroup;
    private edgeBindGroup!: GPUBindGroup;
    private renderBindGroup!: GPUBindGroup;
    private blitBindGroups: Record<number, GPUBindGroup> = {};
    
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
            size: 256 * 10,
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

        // Radix sort: 4 passes (shift 0/8/16/24), result always in A after 4 passes.
        this.sortKeysBuffer = this.sortKeysBufferA;
        this.sortIndicesBuffer = this.sortIndicesBufferA;

        // hist[digit * W + wg_id]: 256 buckets × W workgroups, 4 bytes each.
        const sortWg = Math.ceil(this.numSplats / 256);
        this.histBuffer = device.createBuffer({
            label: "splat sort histogram",
            size: 256 * sortWg * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        // Sort uniforms: VP (64) + shift (4) + pad (12) = 80 bytes per 256-byte slot.
        // 4 slots, one per radix pass (shift = 0, 8, 16, 24).
        this.sortUniformsBuffer = device.createBuffer({
            label: "splat sort uniforms",
            size: 1024,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        for (let i = 0; i < 4; i++) {
            device.queue.writeBuffer(this.sortUniformsBuffer, i * 256 + 64, new Uint32Array([i * 8, 0, 0, 0]));
        }

        // Binning pre-pass: instantiate (splat→tiles) → 8-pass radix sort → calc_ranges
        // maxInstances caps total (splat, tile) pairs across all tiles.
        this.maxInstances = this.numSplats * 16;
        const maxInstances = this.maxInstances;
        const maxTiles = 4096; // covers up to 1024×1024 at 16px tiles (64×64 grid)

        this.binningAtomicBuffer = device.createBuffer({
            label: "splat binning atomic count",
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.instanceKeysBufferA = device.createBuffer({
            label: "splat instance keys A",
            size: maxInstances * 8,
            usage: GPUBufferUsage.STORAGE,
        });
        this.instanceKeysBufferB = device.createBuffer({
            label: "splat instance keys B",
            size: maxInstances * 8,
            usage: GPUBufferUsage.STORAGE,
        });
        this.instanceValsBufferA = device.createBuffer({
            label: "splat instance vals A",
            size: maxInstances * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        this.instanceValsBufferB = device.createBuffer({
            label: "splat instance vals B",
            size: maxInstances * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        // BinningUniforms: vp(64) + grid_width(4) + grid_height(4) + max_instances(4) + _pad(4) = 80
        this.binningUniformsBuffer = device.createBuffer({
            label: "splat binning uniforms",
            size: 80,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.binningUniformsBuffer, 72, new Uint32Array([maxInstances, 0]));

        // SortUniforms for binning: shift(4) + word_idx(4) + pad(4) + pad(4) = 16 bytes per 256-byte slot
        // 8 passes: passes 0-3 sort depth word, passes 4-7 sort tile_id word.
        this.binningSortUniformsBuffer = device.createBuffer({
            label: "splat binning sort uniforms",
            size: 2048,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        for (let i = 0; i < 8; i++) {
            device.queue.writeBuffer(
                this.binningSortUniformsBuffer, i * 256,
                new Uint32Array([(i % 4) * 8, i < 4 ? 0 : 1, 0, 0]),
            );
        }

        const binSortWg = Math.ceil(maxInstances / 256);
        this.binningHistBuffer = device.createBuffer({
            label: "splat binning histogram",
            size: 256 * binSortWg * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        this.tileStartsBuffer = device.createBuffer({
            label: "splat tile starts",
            size: maxTiles * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.tileEndsBuffer = device.createBuffer({
            label: "splat tile ends",
            size: maxTiles * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

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

        // Backward Pipeline — bindings: splats, grads, target, depth, VP uniform, instance_vals, tile_starts, tile_ends
        this.backwardBindGroupLayout = device.createBindGroupLayout({
            label: "splat backward bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
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
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
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

        this.renderBindGroupLayout = device.createBindGroupLayout({
            label: "splat composite render bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform", hasDynamicOffset: true } },
                { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
            ],
        });

        this.blitBindGroupLayout = device.createBindGroupLayout({
            label: "splat blit render bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform", hasDynamicOffset: true } },
            ],
        });

        const compositeModule = device.createShaderModule({ label: "splat composite render", code: injectConstants(renderCompositeModuleSrc) });
        const blitModule = device.createShaderModule({ label: "splat blit render", code: injectConstants(renderBlitModuleSrc) });

        this.targetPipeline = device.createRenderPipeline({
            label: "splat target render pipeline",
            layout: device.createPipelineLayout({ 
                label: "splat target render pipeline layout",
                bindGroupLayouts: [this.renderBindGroupLayout] 
            }),
            vertex: { module: compositeModule, entryPoint: "vert" },
            fragment: { module: compositeModule, entryPoint: "frag_target", targets: [{ format }] },
            primitive: { topology: "triangle-list" },
        });

        this.compositePipeline = device.createRenderPipeline({
            label: "splat composite render pipeline",
            layout: device.createPipelineLayout({ 
                label: "splat composite render pipeline layout",
                bindGroupLayouts: [this.renderBindGroupLayout] 
            }),
            vertex: { module: compositeModule, entryPoint: "vert" },
            fragment: { module: compositeModule, entryPoint: "frag_composite", targets: [{ format }] },
            primitive: { topology: "triangle-list" },
        });

        const blitLayout = device.createPipelineLayout({ 
            label: "splat blit render pipeline layout",
            bindGroupLayouts: [this.blitBindGroupLayout] 
        });

        this.blitPipeline = device.createRenderPipeline({
            label: "splat blit render pipeline",
            layout: blitLayout,
            vertex: { module: blitModule, entryPoint: "vert" },
            fragment: { module: blitModule, entryPoint: "frag_blit", targets: [{ format }] },
            primitive: { topology: "triangle-list" },
        });

        this.blitRPipeline = device.createRenderPipeline({
            label: "splat blit R render pipeline",
            layout: blitLayout,
            vertex: { module: blitModule, entryPoint: "vert" },
            fragment: { module: blitModule, entryPoint: "frag_blit_r", targets: [{ format }] },
            primitive: { topology: "triangle-list" },
        });

        this.blitAPipeline = device.createRenderPipeline({
            label: "splat blit A render pipeline",
            layout: blitLayout,
            vertex: { module: blitModule, entryPoint: "vert" },
            fragment: { module: blitModule, entryPoint: "frag_blit_a", targets: [{ format }] },
            primitive: { topology: "triangle-list" },
        });
        // Radix sort pipelines
        const sortBindGroupLayout = device.createBindGroupLayout({
            label: "splat sort bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // splats
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // in_keys
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // in_indices
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // out_keys
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // out_indices
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // hist
            ],
        });
        const sortModule = device.createShaderModule({ label: "splat sort", code: injectConstants(sortModuleSrc) });
        sortModule.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[splat_sort] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
        });
        const sortLayout = device.createPipelineLayout({
            label: "splat sort pipeline layout",
            bindGroupLayouts: [sortBindGroupLayout],
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

        // AtoB: reads from A, writes to B. BtoA: reads from B, writes to A.
        // init_keys uses BtoA so it writes to A; 4 radix passes alternate AtoB/BtoA.
        this.sortBindGroupBtoA = device.createBindGroup({
            label: "splat sort bind group B to A",
            layout: sortBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.splatBuffer } },
                { binding: 1, resource: { buffer: this.sortKeysBufferB } },
                { binding: 2, resource: { buffer: this.sortIndicesBufferB } },
                { binding: 3, resource: { buffer: this.sortKeysBufferA } },
                { binding: 4, resource: { buffer: this.sortIndicesBufferA } },
                { binding: 5, resource: { buffer: this.sortUniformsBuffer, size: 96 } },
                { binding: 6, resource: { buffer: this.histBuffer } },
            ],
        });

        this.sortBindGroupAtoB = device.createBindGroup({
            label: "splat sort bind group A to B",
            layout: sortBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.splatBuffer } },
                { binding: 1, resource: { buffer: this.sortKeysBufferA } },
                { binding: 2, resource: { buffer: this.sortIndicesBufferA } },
                { binding: 3, resource: { buffer: this.sortKeysBufferB } },
                { binding: 4, resource: { buffer: this.sortIndicesBufferB } },
                { binding: 5, resource: { buffer: this.sortUniformsBuffer, size: 96 } },
                { binding: 6, resource: { buffer: this.histBuffer } },
            ],
        });

        // Binning pipelines and bind groups
        const binningModule = device.createShaderModule({
            label: "splat binning",
            code: injectConstants(binningModuleSrc),
        });
        binningModule.getCompilationInfo().then(info => {
            for (const m of info.messages) console.warn(`[splat_binning] ${m.type}: ${m.message} (line ${m.lineNum})`);
        });

        const binInstantiateLayout = device.createBindGroupLayout({
            label: "splat binning instantiate layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
        });
        const binSortLayout = device.createBindGroupLayout({
            label: "splat binning sort layout",
            entries: [
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
                { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ],
        });
        const binCalcRangesLayout = device.createBindGroupLayout({
            label: "splat binning calc_ranges layout",
            entries: [
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ],
        });

        this.binInstantiatePipeline = device.createComputePipeline({
            label: "splat binning instantiate",
            layout: device.createPipelineLayout({ bindGroupLayouts: [binInstantiateLayout] }),
            compute: { module: binningModule, entryPoint: "instantiate" },
        });
        const binSortPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [binSortLayout] });
        this.binCountPipeline = device.createComputePipeline({
            label: "splat binning count",
            layout: binSortPipelineLayout,
            compute: { module: binningModule, entryPoint: "count" },
        });
        this.binScanPipeline = device.createComputePipeline({
            label: "splat binning scan",
            layout: binSortPipelineLayout,
            compute: { module: binningModule, entryPoint: "scan" },
        });
        this.binScatterPipeline = device.createComputePipeline({
            label: "splat binning scatter",
            layout: binSortPipelineLayout,
            compute: { module: binningModule, entryPoint: "scatter" },
        });
        this.binCalcRangesPipeline = device.createComputePipeline({
            label: "splat binning calc_ranges",
            layout: device.createPipelineLayout({ bindGroupLayouts: [binCalcRangesLayout] }),
            compute: { module: binningModule, entryPoint: "calc_ranges" },
        });

        this.binInstantiateBindGroup = device.createBindGroup({
            label: "splat binning instantiate bind group",
            layout: binInstantiateLayout,
            entries: [
                { binding: 0, resource: { buffer: this.splatBuffer } },
                { binding: 1, resource: { buffer: this.instanceKeysBufferA } },
                { binding: 2, resource: { buffer: this.instanceValsBufferA } },
                { binding: 3, resource: { buffer: this.binningAtomicBuffer } },
                { binding: 4, resource: { buffer: this.binningUniformsBuffer } },
            ],
        });
        this.binSortBindGroupAtoB = device.createBindGroup({
            label: "splat binning sort A to B",
            layout: binSortLayout,
            entries: [
                { binding: 3, resource: { buffer: this.binningAtomicBuffer } },
                { binding: 4, resource: { buffer: this.binningUniformsBuffer } },
                { binding: 5, resource: { buffer: this.instanceKeysBufferA } },
                { binding: 6, resource: { buffer: this.instanceValsBufferA } },
                { binding: 7, resource: { buffer: this.instanceKeysBufferB } },
                { binding: 8, resource: { buffer: this.instanceValsBufferB } },
                { binding: 9, resource: { buffer: this.binningSortUniformsBuffer, size: 16 } },
                { binding: 10, resource: { buffer: this.binningHistBuffer } },
            ],
        });
        this.binSortBindGroupBtoA = device.createBindGroup({
            label: "splat binning sort B to A",
            layout: binSortLayout,
            entries: [
                { binding: 3, resource: { buffer: this.binningAtomicBuffer } },
                { binding: 4, resource: { buffer: this.binningUniformsBuffer } },
                { binding: 5, resource: { buffer: this.instanceKeysBufferB } },
                { binding: 6, resource: { buffer: this.instanceValsBufferB } },
                { binding: 7, resource: { buffer: this.instanceKeysBufferA } },
                { binding: 8, resource: { buffer: this.instanceValsBufferA } },
                { binding: 9, resource: { buffer: this.binningSortUniformsBuffer, size: 16 } },
                { binding: 10, resource: { buffer: this.binningHistBuffer } },
            ],
        });
        this.binCalcRangesBindGroup = device.createBindGroup({
            label: "splat binning calc_ranges bind group",
            layout: binCalcRangesLayout,
            entries: [
                { binding: 3, resource: { buffer: this.binningAtomicBuffer } },
                { binding: 4, resource: { buffer: this.binningUniformsBuffer } },
                { binding: 5, resource: { buffer: this.instanceKeysBufferA } },
                { binding: 11, resource: { buffer: this.tileStartsBuffer } },
                { binding: 12, resource: { buffer: this.tileEndsBuffer } },
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
                { binding: 3, resource: targetDepthTextureView },
                { binding: 4, resource: { buffer: this.splatUniformsBuffer } },
                { binding: 5, resource: { buffer: this.instanceValsBufferA } },
                { binding: 6, resource: { buffer: this.tileStartsBuffer } },
                { binding: 7, resource: { buffer: this.tileEndsBuffer } },
            ],
        });
    }

    setEdgeTarget(depthTextureView: GPUTextureView, edgeTextureView: GPUTextureView, normalTextureView?: GPUTextureView) {
        this.edgeBindGroup = this.device.createBindGroup({
            label: "splat edge bind group",
            layout: this.edgeBindGroupLayout,
            entries: [
                { binding: 0, resource: depthTextureView },
                { binding: 1, resource: edgeTextureView },
                { binding: 2, resource: normalTextureView ?? depthTextureView },
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
                { binding: 7, resource: { buffer: this.renderUniformsBuffer, size: 256 } },
                { binding: 8, resource: ptTextureView },
            ],
        });

        const blitTextures: Record<number, GPUTextureView> = {
            2: splatViewTextureView,
            3: depthTextureView,
            4: edgeTextureView,
            5: bezierViewTextureView,
            6: baseColorBezierViewTextureView,
            7: colorBezierViewTextureView,
        };

        this.blitBindGroups = {};
        for (const [mode, tex] of Object.entries(blitTextures)) {
            this.blitBindGroups[Number(mode)] = this.device.createBindGroup({
                label: `splat blit bind group mode ${mode}`,
                layout: this.blitBindGroupLayout,
                entries: [
                    { binding: 0, resource: tex },
                    { binding: 1, resource: { buffer: this.renderUniformsBuffer, size: 256 } },
                ],
            });
        }
    }

    writeRenderUniforms(edgeEnabled: boolean, baseColorEnabled: boolean, colorEnabled: boolean, meshSplatsEnabled: boolean, splatsEnabled: boolean, aspects: Record<number, number>) {
        for (let mode = 0; mode < 10; mode++) {
            const aspect = aspects[mode] ?? 1.0;
            this.device.queue.writeBuffer(
                this.renderUniformsBuffer,
                mode * 256,
                new Float32Array([
                    edgeEnabled ? 1 : 0, 
                    baseColorEnabled ? 1 : 0, 
                    colorEnabled ? 1 : 0, 
                    meshSplatsEnabled ? 1 : 0, 
                    splatsEnabled ? 1 : 0, 
                    aspect,
                    0, 0, // padding
                ])
            );
        }
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

    dispatchBinning(commandEncoder: GPUCommandEncoder, vpMat: Mat4) {
        if (!this.backwardBindGroup) return;
        const { width, height } = this.dims;
        if (width === 0 || height === 0) return;

        const gridWidth = Math.ceil(width / 16);
        const gridHeight = Math.ceil(height / 16);
        const numTiles = gridWidth * gridHeight;

        // Write VP and grid dimensions into binning uniforms
        const vpData = vpMat as Float32Array;
        this.device.queue.writeBuffer(
            this.binningUniformsBuffer, 0,
            vpData.buffer, vpData.byteOffset, vpData.byteLength,
        );
        this.device.queue.writeBuffer(
            this.binningUniformsBuffer, 64,
            new Uint32Array([gridWidth, gridHeight, this.maxInstances, 0]),
        );

        // Clear atomic count and tile range arrays before instantiate
        commandEncoder.clearBuffer(this.binningAtomicBuffer, 0, 4);
        commandEncoder.clearBuffer(this.tileStartsBuffer, 0, numTiles * 4);
        commandEncoder.clearBuffer(this.tileEndsBuffer, 0, numTiles * 4);

        // Instantiate: map each splat to all overlapping tiles
        const instantiatePass = commandEncoder.beginComputePass({ label: "splat bin instantiate" });
        instantiatePass.setPipeline(this.binInstantiatePipeline);
        instantiatePass.setBindGroup(0, this.binInstantiateBindGroup);
        instantiatePass.dispatchWorkgroups(Math.ceil(this.numSplats / 256));
        instantiatePass.end();

        // 8-pass radix sort: primary key = tile_id, secondary key = depth (farthest first)
        const sortWg = Math.ceil(this.maxInstances / 256);
        for (let i = 0; i < 8; i++) {
            const bg = (i % 2 === 0) ? this.binSortBindGroupAtoB : this.binSortBindGroupBtoA;
            const offset = i * 256;

            const countPass = commandEncoder.beginComputePass({ label: `splat bin count ${i}` });
            countPass.setPipeline(this.binCountPipeline);
            countPass.setBindGroup(0, bg, [offset]);
            countPass.dispatchWorkgroups(sortWg);
            countPass.end();

            const scanPass = commandEncoder.beginComputePass({ label: `splat bin scan ${i}` });
            scanPass.setPipeline(this.binScanPipeline);
            scanPass.setBindGroup(0, bg, [offset]);
            scanPass.dispatchWorkgroups(1);
            scanPass.end();

            const scatterPass = commandEncoder.beginComputePass({ label: `splat bin scatter ${i}` });
            scatterPass.setPipeline(this.binScatterPipeline);
            scatterPass.setBindGroup(0, bg, [offset]);
            scatterPass.dispatchWorkgroups(sortWg);
            scatterPass.end();
        }

        // Compute tile_starts and tile_ends from the sorted instance array
        const rangesPass = commandEncoder.beginComputePass({ label: "splat bin calc_ranges" });
        rangesPass.setPipeline(this.binCalcRangesPipeline);
        rangesPass.setBindGroup(0, this.binCalcRangesBindGroup);
        rangesPass.dispatchWorkgroups(Math.ceil(this.maxInstances / 256));
        rangesPass.end();
    }

    dispatch(commandEncoder: GPUCommandEncoder, vpMat: Mat4, timestampWrites?: NonNullable<GPUComputePassDescriptor["timestampWrites"]>) {
        if (!this.backwardBindGroup) return;

        this.dispatchBinning(commandEncoder, vpMat);

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

        this.device.queue.writeBuffer(
            this.splatUniformsBuffer,
            148, // offset of extras.y
            new Float32Array([this.stepCount])
        );

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

    addDraw(renderPassEncoder: GPURenderPassEncoder, mode: number) {
        if (mode === 0) {
            if (!this.renderBindGroup) return;
            renderPassEncoder.setPipeline(this.targetPipeline);
            renderPassEncoder.setBindGroup(0, this.renderBindGroup, [mode * 256]);
        } else if (mode === 1) {
            if (!this.renderBindGroup) return;
            renderPassEncoder.setPipeline(this.compositePipeline);
            renderPassEncoder.setBindGroup(0, this.renderBindGroup, [mode * 256]);
        } else {
            const bg = this.blitBindGroups[mode];
            if (!bg) return;
            
            if (mode === 3 || mode === 4) {
                renderPassEncoder.setPipeline(this.blitRPipeline);
            } else if (mode === 5) {
                renderPassEncoder.setPipeline(this.blitAPipeline);
            } else {
                renderPassEncoder.setPipeline(this.blitPipeline);
            }
            
            renderPassEncoder.setBindGroup(0, bg, [mode * 256]);
        }
        renderPassEncoder.draw(6);
    }

    /**
     * Run a full depth sort of all splats using the current VP matrix.
     * Writes the sort order into sortIndicesBuffer (back-to-front).
     */
    dispatchSort(commandEncoder: GPUCommandEncoder, vpMat: Mat4) {
        const vpData = vpMat as Float32Array;

        // Write VP to all 4 uniform slots (init_keys reads it from slot 0).
        for (let i = 0; i < 4; i++) {
            this.device.queue.writeBuffer(
                this.sortUniformsBuffer, i * 256,
                vpData.buffer, vpData.byteOffset, vpData.byteLength,
            );
        }

        const wg = Math.ceil(this.numSplats / 256);

        // init_keys uses BtoA so it writes depth keys into buffer A.
        const initPass = commandEncoder.beginComputePass({ label: "splat sort init pass" });
        initPass.setPipeline(this.radixInitPipeline);
        initPass.setBindGroup(0, this.sortBindGroupBtoA, [0]);
        initPass.dispatchWorkgroups(wg);
        initPass.end();

        // 4 radix passes: even → AtoB, odd → BtoA. After 4 passes result is in A.
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
        this.binningAtomicBuffer.destroy();
        this.instanceKeysBufferA.destroy();
        this.instanceKeysBufferB.destroy();
        this.instanceValsBufferA.destroy();
        this.instanceValsBufferB.destroy();
        this.binningUniformsBuffer.destroy();
        this.binningSortUniformsBuffer.destroy();
        this.binningHistBuffer.destroy();
        this.tileStartsBuffer.destroy();
        this.tileEndsBuffer.destroy();
    }
}
