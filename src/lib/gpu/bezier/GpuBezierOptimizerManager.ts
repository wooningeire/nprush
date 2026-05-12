import backwardModuleSrc from "./bezier_backward.wgsl?raw";
import stepModuleSrc from "./bezier_step.wgsl?raw";
import adcModuleSrc from "./bezier_adc.wgsl?raw";
import sortModuleSrc from "./bezier_sort.wgsl?raw";
import initModuleSrc from "./bezier_init.wgsl?raw";
import binningModuleSrc from "./bezier_binning.wgsl?raw";
import type { Mat4 } from "wgpu-matrix";
import { constants, injectWgslConstants } from "../constants";
import { nextPowerOfTwoAtLeast } from "../nextPowerOfTwoAtLeast";

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

    private readonly backwardBindGroupLayout: GPUBindGroupLayout;
    private readonly stepBindGroup: GPUBindGroup;
    private readonly adcBindGroup: GPUBindGroup;
    private readonly initBindGroup: GPUBindGroup;
    private readonly adcBindGroupLayout: GPUBindGroupLayout;

    private readonly adcScratchBuffer: GPUBuffer;
    private readonly histBuffer: GPUBuffer;
    private readonly radixInitPipeline: GPUComputePipeline;
    private readonly radixCountPipeline: GPUComputePipeline;
    private readonly radixScanPipeline: GPUComputePipeline;
    private readonly radixScatterPipeline: GPUComputePipeline;
    private readonly sortBindGroupAtoB: GPUBindGroup;
    private readonly sortBindGroupBtoA: GPUBindGroup;

    // Binning buffers and pipelines
    private readonly binningAtomicBuffer: GPUBuffer;
    private readonly instanceKeysBufferA: GPUBuffer;
    private readonly instanceKeysBufferB: GPUBuffer;
    private readonly instanceValsBufferA: GPUBuffer;
    private readonly instanceValsBufferB: GPUBuffer;
    private readonly binningUniformsBuffer: GPUBuffer;
    private readonly binningHistBuffer: GPUBuffer;
    private readonly tileStartsBuffer: GPUBuffer;
    private readonly tileEndsBuffer: GPUBuffer;

    private readonly binInstantiatePipeline: GPUComputePipeline;
    private readonly binCountPipeline: GPUComputePipeline;
    private readonly binScanPipeline: GPUComputePipeline;
    private readonly binScatterPipeline: GPUComputePipeline;
    private readonly binRangesPipeline: GPUComputePipeline;

    private readonly binBindGroupAtoB: GPUBindGroup;
    private readonly binBindGroupBtoA: GPUBindGroup;

    private backwardBindGroup: GPUBindGroup | null = null;
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

        // Two f32 per curve: positional gradient norm (grad_accum) and color
        // loss contribution (loss_accum), both accumulated across each ADC
        // period and reset to 0 inside the ADC shader.
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

        // Sort Buffers
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

        const W = Math.ceil(this.numBeziers / 256);
        this.histBuffer = device.createBuffer({
            label: "bezier sort histogram",
            size: 256 * W * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        this.sortKeysBuffer = this.sortKeysBufferA;
        this.sortIndicesBuffer = this.sortIndicesBufferA;

        this.sortUniformsBuffer = device.createBuffer({
            label: "bezier sort uniforms",
            size: 1024,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        for (let i = 0; i < 4; i++) {
            device.queue.writeBuffer(this.sortUniformsBuffer, i * 256 + 64, new Uint32Array([i * 8, 0, 0, 0]));
        }

        // Binning buffers
        this.binningAtomicBuffer = device.createBuffer({
            label: "bezier binning atomic",
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const maxInstances = this.numBeziers * 8; // More instances per bezier than splats? 8 seems safe.
        this.instanceKeysBufferA = device.createBuffer({
            label: "bezier binning keys A",
            size: maxInstances * 8, // vec2u
            usage: GPUBufferUsage.STORAGE,
        });
        this.instanceKeysBufferB = device.createBuffer({
            label: "bezier binning keys B",
            size: maxInstances * 8, // vec2u
            usage: GPUBufferUsage.STORAGE,
        });
        this.instanceValsBufferA = device.createBuffer({
            label: "bezier binning vals A",
            size: maxInstances * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        this.instanceValsBufferB = device.createBuffer({
            label: "bezier binning vals B",
            size: maxInstances * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        this.binningUniformsBuffer = device.createBuffer({
            label: "bezier binning uniforms",
            size: 1024,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        // We write 8 shift passes: depth (word 0, shift 0,8,16,24), tile (word 1, shift 0,8,16,24)
        for (let i = 0; i < 8; i++) {
            const word_idx = i < 4 ? 0 : 1;
            const shift = (i % 4) * 8;
            device.queue.writeBuffer(this.binningUniformsBuffer, i * 256 + 64, new Uint32Array([shift, word_idx, 0, 0]));
        }
        
        const W_bin = Math.ceil(maxInstances / 256);
        this.binningHistBuffer = device.createBuffer({
            label: "bezier binning histogram",
            size: 256 * W_bin * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        this.tileStartsBuffer = device.createBuffer({
            label: "bezier tile starts",
            size: 40000 * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        this.tileEndsBuffer = device.createBuffer({
            label: "bezier tile ends",
            size: 40000 * 4,
            usage: GPUBufferUsage.STORAGE,
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
                { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // instance_vals
                { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // tile_starts
                { binding: 13, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // tile_ends
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

        this.adcBindGroup = device.createBindGroup({
            label: "bezier adc bind group",
            layout: this.adcBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bezierBuffer } },
                { binding: 1, resource: { buffer: this.adamBuffer } },
                { binding: 2, resource: { buffer: this.adcBuffer } },
                { binding: 3, resource: { buffer: this.pixelLossBuffer } },
                { binding: 4, resource: { buffer: this.bezierUniformsBuffer } },
                { binding: 5, resource: { buffer: this.adcScratchBuffer } },
            ],
        });

        // Sort pipelines
        const sortBindGroupLayout = device.createBindGroupLayout({
            label: "bezier sort bind group layout",
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
            label: "bezier sort init pipeline",
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

        this.sortBindGroupAtoB = device.createBindGroup({
            label: "bezier sort bind group A to B",
            layout: sortBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bezierBuffer } },
                { binding: 1, resource: { buffer: this.sortKeysBufferA } },
                { binding: 2, resource: { buffer: this.sortIndicesBufferA } },
                { binding: 3, resource: { buffer: this.sortKeysBufferB } },
                { binding: 4, resource: { buffer: this.sortIndicesBufferB } },
                { binding: 5, resource: { buffer: this.sortUniformsBuffer, size: 80 } },
                { binding: 6, resource: { buffer: this.histBuffer } },
            ],
        });

        this.sortBindGroupBtoA = device.createBindGroup({
            label: "bezier sort bind group B to A",
            layout: sortBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bezierBuffer } },
                { binding: 1, resource: { buffer: this.sortKeysBufferB } },
                { binding: 2, resource: { buffer: this.sortIndicesBufferB } },
                { binding: 3, resource: { buffer: this.sortKeysBufferA } },
                { binding: 4, resource: { buffer: this.sortIndicesBufferA } },
                { binding: 5, resource: { buffer: this.sortUniformsBuffer, size: 80 } },
                { binding: 6, resource: { buffer: this.histBuffer } },
            ],
        });

        // Binning Pipelines
        const binningBindGroupLayout = device.createBindGroupLayout({
            label: "bezier binning bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // beziers
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // instance_keys
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // instance_vals
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // atomic_count
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }, // binning_uniforms
                
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // in_keys
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // in_vals
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // out_keys
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // out_vals
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } }, // sort_uniforms
                { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // hist
                { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // tile_starts
                { binding: 12, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // tile_ends
            ],
        });

        const binningModule = device.createShaderModule({ 
            label: "bezier binning", 
            code: inject(binningModuleSrc) 
        });
        binningModule.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[bezier_binning] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
        });

        const binningLayout = device.createPipelineLayout({
            label: "bezier binning pipeline layout",
            bindGroupLayouts: [binningBindGroupLayout]
        });

        this.binInstantiatePipeline = device.createComputePipeline({
            label: "bezier binning instantiate", layout: binningLayout,
            compute: { module: binningModule, entryPoint: "instantiate" },
        });
        this.binCountPipeline = device.createComputePipeline({
            label: "bezier binning count", layout: binningLayout,
            compute: { module: binningModule, entryPoint: "count" },
        });
        this.binScanPipeline = device.createComputePipeline({
            label: "bezier binning scan", layout: binningLayout,
            compute: { module: binningModule, entryPoint: "scan" },
        });
        this.binScatterPipeline = device.createComputePipeline({
            label: "bezier binning scatter", layout: binningLayout,
            compute: { module: binningModule, entryPoint: "scatter" },
        });
        this.binRangesPipeline = device.createComputePipeline({
            label: "bezier binning ranges", layout: binningLayout,
            compute: { module: binningModule, entryPoint: "calc_ranges" },
        });

        this.binBindGroupAtoB = device.createBindGroup({
            label: "bezier binning bind group A to B",
            layout: binningBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bezierBuffer } },
                { binding: 1, resource: { buffer: this.instanceKeysBufferA } },
                { binding: 2, resource: { buffer: this.instanceValsBufferA } },
                { binding: 3, resource: { buffer: this.binningAtomicBuffer } },
                { binding: 4, resource: { buffer: this.binningUniformsBuffer, size: 80 } },
                { binding: 5, resource: { buffer: this.instanceKeysBufferA } },
                { binding: 6, resource: { buffer: this.instanceValsBufferA } },
                { binding: 7, resource: { buffer: this.instanceKeysBufferB } },
                { binding: 8, resource: { buffer: this.instanceValsBufferB } },
                { binding: 9, resource: { buffer: this.binningUniformsBuffer, offset: 256, size: 80 } },
                { binding: 10, resource: { buffer: this.binningHistBuffer } },
                { binding: 11, resource: { buffer: this.tileStartsBuffer } },
                { binding: 12, resource: { buffer: this.tileEndsBuffer } },
            ],
        });

        this.binBindGroupBtoA = device.createBindGroup({
            label: "bezier binning bind group B to A",
            layout: binningBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bezierBuffer } },
                { binding: 1, resource: { buffer: this.instanceKeysBufferB } },
                { binding: 2, resource: { buffer: this.instanceValsBufferB } },
                { binding: 3, resource: { buffer: this.binningAtomicBuffer } },
                { binding: 4, resource: { buffer: this.binningUniformsBuffer, size: 80 } },
                { binding: 5, resource: { buffer: this.instanceKeysBufferB } },
                { binding: 6, resource: { buffer: this.instanceValsBufferB } },
                { binding: 7, resource: { buffer: this.instanceKeysBufferA } },
                { binding: 8, resource: { buffer: this.instanceValsBufferA } },
                { binding: 9, resource: { buffer: this.binningUniformsBuffer, offset: 256, size: 80 } },
                { binding: 10, resource: { buffer: this.binningHistBuffer } },
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
                { binding: 11, resource: { buffer: this.instanceValsBufferA } },
                { binding: 12, resource: { buffer: this.tileStartsBuffer } },
                { binding: 13, resource: { buffer: this.tileEndsBuffer } },
            ],
        });
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
        if (this.stepCount % this.adcPeriod === 0) {
            pass.setPipeline(this.adcPipeline);
            pass.setBindGroup(0, this.adcBindGroup);
            pass.dispatchWorkgroups(1);
        }

        pass.end();
    }
    dispatchBinning(commandEncoder: GPUCommandEncoder, vpMat: Mat4) {
        const vpData = vpMat as Float32Array;
        const gridW = Math.ceil(this.dims.width / 16);
        const gridH = Math.ceil(this.dims.height / 16);
        const maxInstances = this.numBeziers * 8;

        this.device.queue.writeBuffer(
            this.binningUniformsBuffer, 0,
            vpData.buffer, vpData.byteOffset, vpData.byteLength,
        );
        this.device.queue.writeBuffer(
            this.binningUniformsBuffer, 64,
            new Uint32Array([gridW, gridH, maxInstances, 0])
        );

        for (let i = 1; i < 8; i++) {
            commandEncoder.copyBufferToBuffer(this.binningUniformsBuffer, 0, this.binningUniformsBuffer, i * 256, 80);
        }

        this.device.queue.writeBuffer(this.binningAtomicBuffer, 0, new Uint32Array([0]));

        const wg = Math.ceil(this.numBeziers / 256);
        const initPass = commandEncoder.beginComputePass({ label: "bezier binning instantiate" });
        initPass.setPipeline(this.binInstantiatePipeline);
        initPass.setBindGroup(0, this.binBindGroupAtoB, [0]);
        initPass.dispatchWorkgroups(wg);
        initPass.end();

        const W_bin = Math.ceil(maxInstances / 256);
        for (let i = 0; i < 8; i++) {
            const bg = (i % 2 === 0) ? this.binBindGroupAtoB : this.binBindGroupBtoA;
            const offset = i * 256;

            const countPass = commandEncoder.beginComputePass({ label: `bezier binning count ${i}` });
            countPass.setPipeline(this.binCountPipeline);
            countPass.setBindGroup(0, bg, [offset]);
            countPass.dispatchWorkgroups(W_bin);
            countPass.end();

            const scanPass = commandEncoder.beginComputePass({ label: `bezier binning scan ${i}` });
            scanPass.setPipeline(this.binScanPipeline);
            scanPass.setBindGroup(0, bg, [offset]);
            scanPass.dispatchWorkgroups(1);
            scanPass.end();

            const scatterPass = commandEncoder.beginComputePass({ label: `bezier binning scatter ${i}` });
            scatterPass.setPipeline(this.binScatterPipeline);
            scatterPass.setBindGroup(0, bg, [offset]);
            scatterPass.dispatchWorkgroups(W_bin);
            scatterPass.end();
        }

        const rangesPass = commandEncoder.beginComputePass({ label: "bezier binning ranges" });
        rangesPass.setPipeline(this.binRangesPipeline);
        rangesPass.setBindGroup(0, this.binBindGroupAtoB, [0]);
        rangesPass.dispatchWorkgroups(W_bin);
        rangesPass.end();
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
        // Bind to AtoB to write to Buffer A. Uniform offset 0 (shift 0).
        initPass.setBindGroup(0, this.sortBindGroupAtoB, [0]);
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
    }
}
