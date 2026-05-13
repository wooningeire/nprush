import backwardModuleSrc from "./bezier_backward.wgsl?raw";
import stepModuleSrc from "./bezier_step.wgsl?raw";
import adcModuleSrc from "./bezier_adc.wgsl?raw";
import sortModuleSrc from "./bezier_sort.wgsl?raw";
import initModuleSrc from "./bezier_init.wgsl?raw";
import binningModuleSrc from "./bezier_binning.wgsl?raw";
import type { Mat4 } from "wgpu-matrix";
import { constants, injectWgslConstants } from "../constants";

// Optim resolution — must match GpuRunner's OPTIM_SHORT logic.
// We use the square short-side; the actual pixel count is written at runtime.
// The pixel_loss buffer is sized to the worst-case square (OPTIM_SHORT²).
const PIXEL_LOSS_MAX = constants.PIXEL_LOSS_MAX;

export class GpuBezierOptimizerManager {
    private readonly device: GPUDevice;

    readonly numBeziers: number;
    readonly numParams: number;

    readonly bezierBuffer: GPUBuffer;
    readonly gradBuffer: GPUBuffer;
    readonly adamBuffer: GPUBuffer;
    readonly adcBuffer: GPUBuffer;
    readonly bezierUniformsBuffer: GPUBuffer;
    readonly sortIndicesBuffer: GPUBuffer;
    private readonly sortKeysBufferA: GPUBuffer;
    private readonly sortKeysBufferB: GPUBuffer;
    private readonly sortIndicesBufferA: GPUBuffer;
    private readonly sortIndicesBufferB: GPUBuffer;
    private readonly sortKeysBuffer: GPUBuffer;
    private readonly sortUniformsBuffer: GPUBuffer;
    private readonly pixelLossBuffer: GPUBuffer;

    private readonly backwardPipeline: GPUComputePipeline;
    private readonly stepPipeline: GPUComputePipeline;
    private readonly adcPipeline: GPUComputePipeline;
    private readonly initPipeline: GPUComputePipeline;

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

    private readonly backwardBindGroupLayout: GPUBindGroupLayout;
    private readonly stepBindGroup: GPUBindGroup;
    private readonly initBindGroup: GPUBindGroup;
    private readonly adcBindGroupLayout: GPUBindGroupLayout;

    private readonly adcScratchBuffer: GPUBuffer;
    private readonly radixInitPipeline: GPUComputePipeline;
    private readonly radixCountPipeline: GPUComputePipeline;
    private readonly radixScanPipeline: GPUComputePipeline;
    private readonly radixScatterPipeline: GPUComputePipeline;
    private readonly sortBindGroupAtoB: GPUBindGroup;
    private readonly sortBindGroupBtoA: GPUBindGroup;
    private readonly histBuffer: GPUBuffer;

    private backwardBindGroup: GPUBindGroup | null = null;
    private adcBindGroup: GPUBindGroup | null = null;
    private stepCount: number = 0;
    private adcPeriod: number = constants.BEZIER_ADC_PERIOD;

    private dims: { width: number, height: number } = { width: 0, height: 0 };
    /** Last optim pixel count written into AdamState; avoids redundant queue writes each dispatch. */
    private cachedAdamPixelCount: number | null = null;

    constructor({
        device,
        numBeziers = 16,
    }: {
        device: GPUDevice,
        numBeziers?: number,
    }) {
        this.device = device;
        this.numBeziers = numBeziers;
        this.numParams = numBeziers * constants.BEZIER_PARAMS_PER;

        this.bezierBuffer = device.createBuffer({
            label: "bezier buffer",
            size: this.numBeziers * constants.BEZIER_FLOATS_PER * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.gradBuffer = device.createBuffer({
            label: "bezier grad buffer",
            size: this.numParams * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // m + v + t + pad (matches splat adam layout for shader struct compatibility).
        this.adamBuffer = device.createBuffer({
            label: "bezier adam buffer",
            size: this.numParams * 8 + 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Two f32 per curve: positional gradient norm (grad_accum) and an ADC
        // pruning signal (loss_accum). Color layers store color loss there;
        // edge mode stores target-edge support so unsupported orphan strokes die.
        this.adcBuffer = device.createBuffer({
            label: "bezier adc buffer",
            size: this.numBeziers * 8,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.bezierUniformsBuffer = device.createBuffer({
            label: "bezier VP uniforms buffer",
            size: 208, // BezierUniforms through cam_world (+ optional tail pad)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(
            this.bezierUniformsBuffer,
            92,
            new Float32Array([this.adcPeriod]),
        );

        this.pixelLossBuffer = device.createBuffer({
            label: "bezier pixel loss buffer",
            size: PIXEL_LOSS_MAX * 4, // one i32 per pixel
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.adcScratchBuffer = device.createBuffer({
            label: "bezier adc scratch buffer",
            size: this.numBeziers * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        // Sort Buffers — radix sort: result always ends in A after 4 passes.
        this.sortKeysBufferA = device.createBuffer({
            label: "bezier sort keys A",
            size: this.numBeziers * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        this.sortKeysBufferB = device.createBuffer({
            label: "bezier sort keys B",
            size: this.numBeziers * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        this.sortIndicesBufferA = device.createBuffer({
            label: "bezier sort indices A",
            size: this.numBeziers * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.sortIndicesBufferB = device.createBuffer({
            label: "bezier sort indices B",
            size: this.numBeziers * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.sortKeysBuffer = this.sortKeysBufferA;
        this.sortIndicesBuffer = this.sortIndicesBufferA;

        const sortWg = Math.ceil(this.numBeziers / 256);
        this.histBuffer = device.createBuffer({
            label: "bezier sort histogram",
            size: 256 * sortWg * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        this.sortUniformsBuffer = device.createBuffer({
            label: "bezier sort uniforms",
            size: 1024,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        for (let i = 0; i < 4; i++) {
            device.queue.writeBuffer(this.sortUniformsBuffer, i * 256 + 64, new Uint32Array([i * 8, 0, 0, 0]));
        }

        // Binning pre-pass
        this.maxInstances = this.numBeziers * 8;
        const maxInstances = this.maxInstances;
        const maxTiles = 4096;

        this.binningAtomicBuffer = device.createBuffer({
            label: "bezier binning atomic count",
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.instanceKeysBufferA = device.createBuffer({
            label: "bezier instance keys A",
            size: maxInstances * 8,
            usage: GPUBufferUsage.STORAGE,
        });
        this.instanceKeysBufferB = device.createBuffer({
            label: "bezier instance keys B",
            size: maxInstances * 8,
            usage: GPUBufferUsage.STORAGE,
        });
        this.instanceValsBufferA = device.createBuffer({
            label: "bezier instance vals A",
            size: maxInstances * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        this.instanceValsBufferB = device.createBuffer({
            label: "bezier instance vals B",
            size: maxInstances * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        this.binningUniformsBuffer = device.createBuffer({
            label: "bezier binning uniforms",
            size: 80,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.binningUniformsBuffer, 72, new Uint32Array([maxInstances, 0]));

        this.binningSortUniformsBuffer = device.createBuffer({
            label: "bezier binning sort uniforms",
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
            label: "bezier binning histogram",
            size: 256 * binSortWg * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        this.tileStartsBuffer = device.createBuffer({
            label: "bezier tile starts",
            size: maxTiles * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.tileEndsBuffer = device.createBuffer({
            label: "bezier tile ends",
            size: maxTiles * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // NUM_BEZIERS_PLUS_ONE / NUM_BEZIERS_MINUS_ONE must come before
        // NUM_BEZIERS for the same substring reason as the splat shaders.
        // PIXEL_LOSS_SIZE = OPTIM_WIDTH * OPTIM_HEIGHT is injected then.
        // For the shader module we use placeholder values that get replaced
        // via a separate per-dispatch inject — here we bake in the max size
        // so the buffer declaration compiles.
        const inject = (src: string) => {
            return injectWgslConstants(src, {
                ...constants,
                NUM_BEZIERS: this.numBeziers,
                NUM_BEZIERS_PLUS_ONE: this.numBeziers + 1,
                NUM_BEZIERS_MINUS_ONE: this.numBeziers - 1,
                NUM_BEZIERS_DIV_32: Math.ceil(this.numBeziers / 32),
                /** Contiguous sort-index chunk per thread in backward tile compact (ceil(N/256)). */
                BEZIER_SORT_CHUNK: Math.ceil(this.numBeziers / 256),
                NUM_BEZIER_PARAMS: this.numParams,
                PIXEL_LOSS_SIZE: PIXEL_LOSS_MAX,
            });
        };

        this.backwardBindGroupLayout = device.createBindGroupLayout({
            label: "bezier backward bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 8, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            ],
        });
        const backwardModule = device.createShaderModule({
            label: "bezier backward",
            code: inject(backwardModuleSrc),
        });
        backwardModule.getCompilationInfo().then(info => {
            for (const m of info.messages) console.warn(`[bezier_backward] ${m.type}: ${m.message} (line ${m.lineNum})`);
        });
        this.backwardPipeline = device.createComputePipeline({
            label: "bezier backward pipeline",
            layout: device.createPipelineLayout({ 
                label: "bezier backward pipeline layout",
                bindGroupLayouts: [this.backwardBindGroupLayout] 
            }),
            compute: { module: backwardModule, entryPoint: "main" },
        });

        const stepBindGroupLayout = device.createBindGroupLayout({
            label: "bezier step bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
        });
        const stepModule = device.createShaderModule({
            label: "bezier step",
            code: inject(stepModuleSrc),
        });
        stepModule.getCompilationInfo().then(info => {
            for (const m of info.messages) console.warn(`[bezier_step] ${m.type}: ${m.message} (line ${m.lineNum})`);
        });
        this.stepPipeline = device.createComputePipeline({
            label: "bezier step pipeline",
            layout: device.createPipelineLayout({ 
                label: "bezier step pipeline layout",
                bindGroupLayouts: [stepBindGroupLayout] 
            }),
            compute: { module: stepModule, entryPoint: "main" },
        });

        this.stepBindGroup = device.createBindGroup({
            label: "bezier step bind group",
            layout: stepBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bezierBuffer } },
                { binding: 1, resource: { buffer: this.gradBuffer } },
                { binding: 2, resource: { buffer: this.adamBuffer } },
                { binding: 3, resource: { buffer: this.adcBuffer } },
                { binding: 4, resource: { buffer: this.bezierUniformsBuffer } },
            ],
        });

        // ADC pipeline: clones/splits high-gradient curves into dead slots.
        this.adcBindGroupLayout = device.createBindGroupLayout({
            label: "bezier adc bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
            ],
        });
        const adcModule = device.createShaderModule({
            label: "bezier adc",
            code: inject(adcModuleSrc),
        });
        adcModule.getCompilationInfo().then(info => {
            for (const m of info.messages) console.warn(`[bezier_adc] ${m.type}: ${m.message} (line ${m.lineNum})`);
        });
        this.adcPipeline = device.createComputePipeline({
            label: "bezier adc pipeline",
            layout: device.createPipelineLayout({ 
                label: "bezier adc pipeline layout",
                bindGroupLayouts: [this.adcBindGroupLayout] 
            }),
            compute: { module: adcModule, entryPoint: "main" },
        });

        // Radix sort pipelines
        const sortBindGroupLayout = device.createBindGroupLayout({
            label: "bezier sort bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // beziers
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // in_keys
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // in_indices
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // out_keys
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // out_indices
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // hist
            ],
        });
        const sortModule = device.createShaderModule({ label: "bezier sort", code: inject(sortModuleSrc) });
        sortModule.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[bezier_sort] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
        });

        const initBindGroupLayout = device.createBindGroupLayout({
            label: "bezier init bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ],
        });
        const initModule = device.createShaderModule({
            label: "bezier init",
            code: inject(initModuleSrc),
        });
        initModule.getCompilationInfo().then(info => {
            for (const m of info.messages) console.warn(`[bezier_init] ${m.type}: ${m.message} (line ${m.lineNum})`);
        });
        this.initPipeline = device.createComputePipeline({
            label: "bezier init pipeline",
            layout: device.createPipelineLayout({
                label: "bezier init pipeline layout",
                bindGroupLayouts: [initBindGroupLayout],
            }),
            compute: { module: initModule, entryPoint: "main" },
        });
        this.initBindGroup = device.createBindGroup({
            label: "bezier init bind group",
            layout: initBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bezierBuffer } },
            ],
        });

        // Run initialization pass immediately
        const initEncoder = device.createCommandEncoder({ label: "bezier init encoder" });
        const initPass = initEncoder.beginComputePass({ label: "bezier init pass" });
        initPass.setPipeline(this.initPipeline);
        initPass.setBindGroup(0, this.initBindGroup);
        initPass.dispatchWorkgroups(Math.ceil(this.numBeziers / 64));
        initPass.end();
        device.queue.submit([initEncoder.finish()]);

        const sortLayout = device.createPipelineLayout({
            label: "bezier sort pipeline layout",
            bindGroupLayouts: [sortBindGroupLayout],
        });
        this.radixInitPipeline = device.createComputePipeline({
            label: "bezier sort init_keys pipeline",
            layout: sortLayout,
            compute: { module: sortModule, entryPoint: "init_keys" },
        });
        this.radixCountPipeline = device.createComputePipeline({
            label: "bezier sort count pipeline",
            layout: sortLayout,
            compute: { module: sortModule, entryPoint: "count" },
        });
        this.radixScanPipeline = device.createComputePipeline({
            label: "bezier sort scan pipeline",
            layout: sortLayout,
            compute: { module: sortModule, entryPoint: "scan" },
        });
        this.radixScatterPipeline = device.createComputePipeline({
            label: "bezier sort scatter pipeline",
            layout: sortLayout,
            compute: { module: sortModule, entryPoint: "scatter" },
        });

        // init_keys uses BtoA so it writes to A; 4 radix passes alternate AtoB/BtoA.
        this.sortBindGroupBtoA = device.createBindGroup({
            label: "bezier sort bind group B to A",
            layout: sortBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bezierBuffer } },
                { binding: 1, resource: { buffer: this.sortKeysBufferB } },
                { binding: 2, resource: { buffer: this.sortIndicesBufferB } },
                { binding: 3, resource: { buffer: this.sortKeysBufferA } },
                { binding: 4, resource: { buffer: this.sortIndicesBufferA } },
                { binding: 5, resource: { buffer: this.sortUniformsBuffer, size: 96 } },
                { binding: 6, resource: { buffer: this.histBuffer } },
            ],
        });

        this.sortBindGroupAtoB = device.createBindGroup({
            label: "bezier sort bind group A to B",
            layout: sortBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bezierBuffer } },
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
            label: "bezier binning",
            code: inject(binningModuleSrc),
        });
        binningModule.getCompilationInfo().then(info => {
            for (const m of info.messages) console.warn(`[bezier_binning] ${m.type}: ${m.message} (line ${m.lineNum})`);
        });

        const binInstantiateLayout = device.createBindGroupLayout({
            label: "bezier binning instantiate layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
        });
        const binSortLayout = device.createBindGroupLayout({
            label: "bezier binning sort layout",
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
            label: "bezier binning calc_ranges layout",
            entries: [
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ],
        });

        this.binInstantiatePipeline = device.createComputePipeline({
            label: "bezier binning instantiate",
            layout: device.createPipelineLayout({ bindGroupLayouts: [binInstantiateLayout] }),
            compute: { module: binningModule, entryPoint: "instantiate" },
        });
        const binSortPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [binSortLayout] });
        this.binCountPipeline = device.createComputePipeline({
            label: "bezier binning count",
            layout: binSortPipelineLayout,
            compute: { module: binningModule, entryPoint: "count" },
        });
        this.binScanPipeline = device.createComputePipeline({
            label: "bezier binning scan",
            layout: binSortPipelineLayout,
            compute: { module: binningModule, entryPoint: "scan" },
        });
        this.binScatterPipeline = device.createComputePipeline({
            label: "bezier binning scatter",
            layout: binSortPipelineLayout,
            compute: { module: binningModule, entryPoint: "scatter" },
        });
        this.binCalcRangesPipeline = device.createComputePipeline({
            label: "bezier binning calc_ranges",
            layout: device.createPipelineLayout({ bindGroupLayouts: [binCalcRangesLayout] }),
            compute: { module: binningModule, entryPoint: "calc_ranges" },
        });

        this.binInstantiateBindGroup = device.createBindGroup({
            label: "bezier binning instantiate bind group",
            layout: binInstantiateLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bezierBuffer } },
                { binding: 1, resource: { buffer: this.instanceKeysBufferA } },
                { binding: 2, resource: { buffer: this.instanceValsBufferA } },
                { binding: 3, resource: { buffer: this.binningAtomicBuffer } },
                { binding: 4, resource: { buffer: this.binningUniformsBuffer } },
            ],
        });
        this.binSortBindGroupAtoB = device.createBindGroup({
            label: "bezier binning sort A to B",
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
            label: "bezier binning sort B to A",
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
            label: "bezier binning calc_ranges bind group",
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

    writeVPMatrix(mat: Float32Array | number[]) {
        this.device.queue.writeBuffer(
            this.bezierUniformsBuffer,
            0,
            (mat as Float32Array).buffer,
            (mat as Float32Array).byteOffset,
            (mat as Float32Array).byteLength
        );
    }

    writeVPInvMatrix(mat: Mat4) {
        // vp_inv is at offset 112 in BezierUniforms (after optim_width/height + padding)
        this.device.queue.writeBuffer(
            this.bezierUniformsBuffer,
            112,
            (mat as Float32Array).buffer,
            (mat as Float32Array).byteOffset,
            (mat as Float32Array).byteLength
        );
    }

    writeOptimDims(width: number, height: number) {
        // Writes optim_width (96) and optim_height (100)
        this.device.queue.writeBuffer(
            this.bezierUniformsBuffer,
            96,
            new Float32Array([width, height]),
        );
    }

    /** Camera world position (`invView * (0,0,0,1)`), for degree-1 SH view dependence. */
    writeCamWorld(x: number, y: number, z: number, w: number = 1) {
        // cam_world is at offset 176
        this.device.queue.writeBuffer(
            this.bezierUniformsBuffer,
            176,
            new Float32Array([x, y, z, w]),
        );
    }

    writeMode(mode: number = 0) {
        this.device.queue.writeBuffer(
            this.bezierUniformsBuffer,
            64,
            new Float32Array([mode])
        );
    }

    writeBgPenalty(weight: number = 0) {
        // Writes BezierUniforms.bg_penalty at offset 80.
        // Layout: vp(64) + mode(4) + max_width(4) + prune_alpha(4) + prune_width(4) = 80
        // 0 = disabled (coarse bezier layer), >0 = enabled (fine bezier layer).
        this.device.queue.writeBuffer(
            this.bezierUniformsBuffer,
            80,
            new Float32Array([weight])
        );
    }

    writeMaxWidth(maxWidth: number = 0) {
        // Writes into StepUniforms.max_width (offset 68). 0 = use default cap.
        this.device.queue.writeBuffer(
            this.bezierUniformsBuffer,
            68,
            new Float32Array([maxWidth])
        );
    }

    writeKillThresholds(alphaThresh: number = 0, widthThresh: number = 0) {
        // Writes prune_alpha_thresh (offset 72) and prune_width_thresh (offset 76).
        // 0 = use default (0.001). Set higher to kill more aggressively, lower for less.
        this.device.queue.writeBuffer(
            this.bezierUniformsBuffer,
            72,
            new Float32Array([alphaThresh, widthThresh])
        );
    }

    setAdcPeriod(period: number) {
        this.adcPeriod = period;
        this.device.queue.writeBuffer(
            this.bezierUniformsBuffer,
            92,
            new Float32Array([period]),
        );
    }

    writeNoKill(noKill: boolean) {
        // Writes adam.no_kill flag at offset numParams*8 + 8 (after t and pixel_count).
        this.device.queue.writeBuffer(
            this.adamBuffer,
            this.numParams * 8 + 8,
            new Float32Array([noKill ? 1.0 : 0.0])
        );
    }

    /**
     * Reset Adam momentum (m, v) and step counter (t) without touching the
     * bezier parameters themselves. Call this whenever the camera changes
     * during turntable training so stale cross-view momentum doesn't corrupt
     * the gradient step for the new viewpoint.
     *
     * AdamState layout: m[numParams * f32] | v[numParams * f32] | t(f32) | pixel_count(f32) | no_kill(f32) | pad(f32)
     *
     * We intentionally do NOT clear the ADC buffers (grad/loss accumulators or
     * pixel_loss) here: the camera effect runs whenever the view-projection
     * matrix changes, including every frame during orbit. Clearing them would
     * leave at most one step of signal while bezier_adc.wgsl still divides by
     * adc_period, so clone/split and loss-based killing would effectively
     * never fire. Those buffers are reset inside the ADC shader after each
     * ADC pass instead.
     */
    resetAdam() {
        // Zero m and v, reset t to 0. pixel_count and no_kill are written
        // separately each frame so we don't need to preserve them here.
        this.device.queue.writeBuffer(
            this.adamBuffer,
            0,
            new Float32Array(this.numParams * 2 + 1) // m + v + t, all zeros
        );
    }

    /** Clear ADC statistics and pixel-loss map; reset the step counter used for ADC cadence. */
    resetAdcState() {
        this.device.queue.writeBuffer(
            this.adcBuffer,
            0,
            new Float32Array(this.numBeziers * 2), // grad_accum + loss_accum
        );
        this.device.queue.writeBuffer(
            this.pixelLossBuffer,
            0,
            new Int32Array(this.pixelLossBuffer.size / 4),
        );
        this.stepCount = 0;
    }

    setBackwardTarget(
        targetTextureView: GPUTextureView,
        targetDepthTextureView: GPUTextureView,
        bgColorTextureView: GPUTextureView,
        normalTextureView: GPUTextureView,
        width: number,
        height: number,
    ) {
        this.dims = { width, height };
        this.writeOptimDims(width, height);

        this.backwardBindGroup = this.device.createBindGroup({
            label: "bezier backward bind group",
            layout: this.backwardBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bezierBuffer } },
                { binding: 1, resource: { buffer: this.gradBuffer } },
                { binding: 2, resource: targetTextureView },
                { binding: 3, resource: targetDepthTextureView },
                { binding: 4, resource: { buffer: this.bezierUniformsBuffer } },
                { binding: 5, resource: bgColorTextureView },
                { binding: 7, resource: { buffer: this.adcBuffer } },
                { binding: 8, resource: normalTextureView },
                { binding: 9, resource: { buffer: this.pixelLossBuffer } },
                { binding: 10, resource: { buffer: this.instanceValsBufferA } },
                { binding: 11, resource: { buffer: this.tileStartsBuffer } },
                { binding: 12, resource: { buffer: this.tileEndsBuffer } },
            ],
        });

        this.adcBindGroup = this.device.createBindGroup({
            label: "bezier adc bind group",
            layout: this.adcBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bezierBuffer } },
                { binding: 1, resource: { buffer: this.adamBuffer } },
                { binding: 2, resource: { buffer: this.adcBuffer } },
                { binding: 3, resource: { buffer: this.pixelLossBuffer } },
                { binding: 4, resource: { buffer: this.bezierUniformsBuffer } },
                { binding: 5, resource: { buffer: this.adcScratchBuffer } },
                { binding: 6, resource: targetDepthTextureView },
            ],
        });
    }

    dispatchBinning(commandEncoder: GPUCommandEncoder, vpMat: Mat4) {
        if (!this.backwardBindGroup) return;
        const { width, height } = this.dims;
        if (width === 0 || height === 0) return;

        const gridWidth = Math.ceil(width / 16);
        const gridHeight = Math.ceil(height / 16);
        const numTiles = gridWidth * gridHeight;

        const vpData = vpMat as Float32Array;
        this.device.queue.writeBuffer(
            this.binningUniformsBuffer, 0,
            vpData.buffer, vpData.byteOffset, vpData.byteLength,
        );
        this.device.queue.writeBuffer(
            this.binningUniformsBuffer, 64,
            new Uint32Array([gridWidth, gridHeight, this.maxInstances, 0]),
        );

        commandEncoder.clearBuffer(this.binningAtomicBuffer, 0, 4);
        commandEncoder.clearBuffer(this.tileStartsBuffer, 0, numTiles * 4);
        commandEncoder.clearBuffer(this.tileEndsBuffer, 0, numTiles * 4);

        const instantiatePass = commandEncoder.beginComputePass({ label: "bezier bin instantiate" });
        instantiatePass.setPipeline(this.binInstantiatePipeline);
        instantiatePass.setBindGroup(0, this.binInstantiateBindGroup);
        instantiatePass.dispatchWorkgroups(Math.ceil(this.numBeziers / 256));
        instantiatePass.end();

        const sortWg = Math.ceil(this.maxInstances / 256);
        for (let i = 0; i < 8; i++) {
            const bg = (i % 2 === 0) ? this.binSortBindGroupAtoB : this.binSortBindGroupBtoA;
            const offset = i * 256;

            const countPass = commandEncoder.beginComputePass({ label: `bezier bin count ${i}` });
            countPass.setPipeline(this.binCountPipeline);
            countPass.setBindGroup(0, bg, [offset]);
            countPass.dispatchWorkgroups(sortWg);
            countPass.end();

            const scanPass = commandEncoder.beginComputePass({ label: `bezier bin scan ${i}` });
            scanPass.setPipeline(this.binScanPipeline);
            scanPass.setBindGroup(0, bg, [offset]);
            scanPass.dispatchWorkgroups(1);
            scanPass.end();

            const scatterPass = commandEncoder.beginComputePass({ label: `bezier bin scatter ${i}` });
            scatterPass.setPipeline(this.binScatterPipeline);
            scatterPass.setBindGroup(0, bg, [offset]);
            scatterPass.dispatchWorkgroups(sortWg);
            scatterPass.end();
        }

        const rangesPass = commandEncoder.beginComputePass({ label: "bezier bin calc_ranges" });
        rangesPass.setPipeline(this.binCalcRangesPipeline);
        rangesPass.setBindGroup(0, this.binCalcRangesBindGroup);
        rangesPass.dispatchWorkgroups(Math.ceil(this.maxInstances / 256));
        rangesPass.end();
    }

    dispatch(commandEncoder: GPUCommandEncoder, vpMat: Mat4, timestampWrites?: NonNullable<GPUComputePassDescriptor["timestampWrites"]>) {
        if (!this.backwardBindGroup) return;

        this.dispatchBinning(commandEncoder, vpMat);

        // Update pixel count for normalization in the step shader.
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
            label: "bezier backward and step pass",
            ...(timestampWrites ? { timestampWrites } : {}),
        });

        pass.setPipeline(this.backwardPipeline);
        pass.setBindGroup(0, this.backwardBindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.dims.width / 16), Math.ceil(this.dims.height / 16));

        pass.setPipeline(this.stepPipeline);
        pass.setBindGroup(0, this.stepBindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.numBeziers / 64));

        // ADC fires every adcPeriod steps; bezier_adc.wgsl reads the same value
        // from uniforms.adc_period_steps for grad/loss averaging.
        this.stepCount++;
        if (this.stepCount % this.adcPeriod === 0 && this.adcBindGroup) {
            pass.setPipeline(this.adcPipeline);
            pass.setBindGroup(0, this.adcBindGroup);
            pass.dispatchWorkgroups(1);
        }

        pass.end();
    }

    dispatchSort(commandEncoder: GPUCommandEncoder, vpMat: Mat4) {
        const vpData = vpMat as Float32Array;

        for (let i = 0; i < 4; i++) {
            this.device.queue.writeBuffer(
                this.sortUniformsBuffer, i * 256,
                vpData.buffer, vpData.byteOffset, vpData.byteLength,
            );
        }

        const wg = Math.ceil(this.numBeziers / 256);

        const initPass = commandEncoder.beginComputePass({ label: "bezier sort init pass" });
        initPass.setPipeline(this.radixInitPipeline);
        initPass.setBindGroup(0, this.sortBindGroupBtoA, [0]);
        initPass.dispatchWorkgroups(wg);
        initPass.end();

        for (let i = 0; i < 4; i++) {
            const bg = (i % 2 === 0) ? this.sortBindGroupAtoB : this.sortBindGroupBtoA;
            const offset = i * 256;

            const countPass = commandEncoder.beginComputePass({ label: `bezier sort count ${i}` });
            countPass.setPipeline(this.radixCountPipeline);
            countPass.setBindGroup(0, bg, [offset]);
            countPass.dispatchWorkgroups(wg);
            countPass.end();

            const scanPass = commandEncoder.beginComputePass({ label: `bezier sort scan ${i}` });
            scanPass.setPipeline(this.radixScanPipeline);
            scanPass.setBindGroup(0, bg, [offset]);
            scanPass.dispatchWorkgroups(1);
            scanPass.end();

            const scatterPass = commandEncoder.beginComputePass({ label: `bezier sort scatter ${i}` });
            scatterPass.setPipeline(this.radixScatterPipeline);
            scatterPass.setBindGroup(0, bg, [offset]);
            scatterPass.dispatchWorkgroups(wg);
            scatterPass.end();
        }
    }

    destroy() {
        this.bezierBuffer.destroy();
        this.gradBuffer.destroy();
        this.adamBuffer.destroy();
        this.adcBuffer.destroy();
        this.adcScratchBuffer.destroy();
        this.bezierUniformsBuffer.destroy();
        this.pixelLossBuffer.destroy();
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
