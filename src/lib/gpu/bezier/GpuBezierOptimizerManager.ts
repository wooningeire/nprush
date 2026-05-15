import backwardModuleSrc from "./bezier_backward.wgsl?raw";
import stepModuleSrc from "./bezier_step.wgsl?raw";
import adcModuleSrc from "./bezier_adc.wgsl?raw";
import initModuleSrc from "./bezier_init.wgsl?raw";
import type { Mat4 } from "wgpu-matrix";
import { constants, injectWgslConstants } from "../constants.ts";
import { GpuBezierBinningManager } from "./GpuBezierBinningManager.ts";
import { GpuBezierDepthSortManager } from "./GpuBezierDepthSortManager.ts";

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
    private readonly pixelLossBuffer: GPUBuffer;
    private readonly adcScratchBuffer: GPUBuffer;

    private readonly binningManager: GpuBezierBinningManager;
    private readonly sortManager: GpuBezierDepthSortManager;

    get sortIndicesBuffer() { return this.sortManager.sortIndicesBufferA; }
    get tileStartsBuffer() { return this.binningManager.tileStartsBuffer; }
    get tileEndsBuffer() { return this.binningManager.tileEndsBuffer; }

    private readonly backwardPipeline: GPUComputePipeline;
    private readonly stepPipeline: GPUComputePipeline;
    private readonly adcPipeline: GPUComputePipeline;
    private readonly initPipeline: GPUComputePipeline;

    private readonly backwardBindGroupLayout: GPUBindGroupLayout;
    private readonly stepBindGroup: GPUBindGroup;
    private readonly initBindGroup: GPUBindGroup;
    private readonly adcBindGroupLayout: GPUBindGroupLayout;

    private backwardBindGroup: GPUBindGroup | null = null;
    private adcBindGroup: GPUBindGroup | null = null;
    private stepCount: number = 0;
    private adcPeriod: number = constants.BEZIER_ADC_PERIOD;

    private dims: { width: number, height: number } = { width: 0, height: 0 };
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

        this.bezierBuffer = device.createBuffer({ label: "bezier buffer", size: this.numBeziers * constants.BEZIER_FLOATS_PER * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.gradBuffer = device.createBuffer({ label: "bezier grad buffer", size: this.numParams * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.adamBuffer = device.createBuffer({ label: "bezier adam buffer", size: this.numParams * 8 + 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.adcBuffer = device.createBuffer({ label: "bezier adc buffer", size: this.numBeziers * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.bezierUniformsBuffer = device.createBuffer({ label: "bezier VP uniforms buffer", size: 208, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(this.bezierUniformsBuffer, 92, new Float32Array([this.adcPeriod]));
        this.pixelLossBuffer = device.createBuffer({ label: "bezier pixel loss buffer", size: PIXEL_LOSS_MAX * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.adcScratchBuffer = device.createBuffer({ label: "bezier adc scratch buffer", size: this.numBeziers * 4, usage: GPUBufferUsage.STORAGE });

        this.binningManager = new GpuBezierBinningManager({ device, numBeziers, numParams: this.numParams, bezierBuffer: this.bezierBuffer });
        this.sortManager = new GpuBezierDepthSortManager({ device, numBeziers, bezierBuffer: this.bezierBuffer });

        const inject = (src: string) => injectWgslConstants(src, {
            ...constants,
            NUM_BEZIERS: this.numBeziers,
            NUM_BEZIERS_PLUS_ONE: this.numBeziers + 1,
            NUM_BEZIERS_MINUS_ONE: this.numBeziers - 1,
            NUM_BEZIERS_DIV_32: Math.ceil(this.numBeziers / 32),
            BEZIER_SORT_CHUNK: Math.ceil(this.numBeziers / 256),
            NUM_BEZIER_PARAMS: this.numParams,
            PIXEL_LOSS_SIZE: PIXEL_LOSS_MAX,
        });

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
        this.backwardPipeline = device.createComputePipeline({
            label: "bezier backward pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.backwardBindGroupLayout] }),
            compute: { module: device.createShaderModule({ label: "bezier backward", code: inject(backwardModuleSrc) }), entryPoint: "main" },
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
        this.stepPipeline = device.createComputePipeline({
            label: "bezier step pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [stepBindGroupLayout] }),
            compute: { module: device.createShaderModule({ label: "bezier step", code: inject(stepModuleSrc) }), entryPoint: "main" },
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
        this.adcPipeline = device.createComputePipeline({
            label: "bezier adc pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.adcBindGroupLayout] }),
            compute: { module: device.createShaderModule({ label: "bezier adc", code: inject(adcModuleSrc) }), entryPoint: "main" },
        });

        const initBindGroupLayout = device.createBindGroupLayout({
            label: "bezier init bind group layout",
            entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }],
        });
        this.initPipeline = device.createComputePipeline({
            label: "bezier init pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [initBindGroupLayout] }),
            compute: { module: device.createShaderModule({ label: "bezier init", code: inject(initModuleSrc) }), entryPoint: "main" },
        });
        this.initBindGroup = device.createBindGroup({
            label: "bezier init bind group",
            layout: initBindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: this.bezierBuffer } }],
        });

        const initEncoder = device.createCommandEncoder({ label: "bezier init encoder" });
        const initPass = initEncoder.beginComputePass({ label: "bezier init pass" });
        initPass.setPipeline(this.initPipeline);
        initPass.setBindGroup(0, this.initBindGroup);
        initPass.dispatchWorkgroups(Math.ceil(this.numBeziers / 64));
        initPass.end();
        device.queue.submit([initEncoder.finish()]);
    }

    writeVPMatrix(mat: Float32Array | number[]) { this.device.queue.writeBuffer(this.bezierUniformsBuffer, 0, (mat as Float32Array).buffer, (mat as Float32Array).byteOffset, (mat as Float32Array).byteLength); }
    writeVPInvMatrix(mat: Mat4) { this.device.queue.writeBuffer(this.bezierUniformsBuffer, 112, (mat as Float32Array).buffer, (mat as Float32Array).byteOffset, (mat as Float32Array).byteLength); }
    writeOptimDims(width: number, height: number) { this.device.queue.writeBuffer(this.bezierUniformsBuffer, 96, new Float32Array([width, height])); }
    writeCamWorld(x: number, y: number, z: number, w: number = 1) { this.device.queue.writeBuffer(this.bezierUniformsBuffer, 176, new Float32Array([x, y, z, w])); }
    writeMode(mode: number = 0) { this.device.queue.writeBuffer(this.bezierUniformsBuffer, 64, new Float32Array([mode])); }
    writeBgPenalty(weight: number = 0) { this.device.queue.writeBuffer(this.bezierUniformsBuffer, 80, new Float32Array([weight])); }
    writeMaxWidth(maxWidth: number = 0) { this.device.queue.writeBuffer(this.bezierUniformsBuffer, 68, new Float32Array([maxWidth])); }
    writeKillThresholds(alphaThresh: number = 0, widthThresh: number = 0) { this.device.queue.writeBuffer(this.bezierUniformsBuffer, 72, new Float32Array([alphaThresh, widthThresh])); }
    setAdcPeriod(period: number) { this.adcPeriod = period; this.device.queue.writeBuffer(this.bezierUniformsBuffer, 92, new Float32Array([period])); }
    writeNoKill(noKill: boolean) { this.device.queue.writeBuffer(this.adamBuffer, this.numParams * 8 + 8, new Float32Array([noKill ? 1.0 : 0.0])); }

    resetAdam() { this.device.queue.writeBuffer(this.adamBuffer, 0, new Float32Array(this.numParams * 2 + 1)); }
    resetAdcState() {
        this.device.queue.writeBuffer(this.adcBuffer, 0, new Float32Array(this.numBeziers * 2));
        this.device.queue.writeBuffer(this.pixelLossBuffer, 0, new Int32Array(this.pixelLossBuffer.size / 4));
        this.stepCount = 0;
    }

    setBackwardTarget(targetTextureView: GPUTextureView, targetDepthTextureView: GPUTextureView, bgColorTextureView: GPUTextureView, normalTextureView: GPUTextureView, width: number, height: number) {
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
                { binding: 10, resource: { buffer: this.binningManager.instanceValsBufferA } },
                { binding: 11, resource: { buffer: this.binningManager.tileStartsBuffer } },
                { binding: 12, resource: { buffer: this.binningManager.tileEndsBuffer } },
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

    addBinningDispatches(pass: GPUComputePassEncoder, vpMat: Mat4, commandEncoder: GPUCommandEncoder) {
        const { width, height } = this.dims;
        if (width === 0 || height === 0) return;
        const gridWidth = Math.ceil(width / 16);
        const gridHeight = Math.ceil(height / 16);
        const numTiles = gridWidth * gridHeight;

        commandEncoder.clearBuffer(this.binningManager.binningAtomicBuffer, 0, 4);
        commandEncoder.clearBuffer(this.binningManager.tileStartsBuffer, 0, numTiles * 4);
        commandEncoder.clearBuffer(this.binningManager.tileEndsBuffer, 0, numTiles * 4);

        if (pass) {
            this.binningManager.addDispatches(pass, vpMat, width, height);
        }
    }

    addOptimizationDispatches(pass: GPUComputePassEncoder) {
        if (!this.backwardBindGroup || !this.adcBindGroup) return;
        const { width, height } = this.dims;
        if (width === 0 || height === 0) return;

        const pixelCount = width * height;
        if (this.cachedAdamPixelCount !== pixelCount) {
            this.cachedAdamPixelCount = pixelCount;
            this.device.queue.writeBuffer(this.adamBuffer, this.numParams * 8 + 4, new Float32Array([pixelCount]));
        }

        this.device.queue.writeBuffer(this.bezierUniformsBuffer, 84, new Float32Array([this.stepCount]));

        pass.setPipeline(this.backwardPipeline);
        pass.setBindGroup(0, this.backwardBindGroup);
        pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));

        pass.setPipeline(this.stepPipeline);
        pass.setBindGroup(0, this.stepBindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.numBeziers / 64));

        this.stepCount++;
        if (this.stepCount % this.adcPeriod === 0) {
            pass.setPipeline(this.adcPipeline);
            pass.setBindGroup(0, this.adcBindGroup);
            pass.dispatchWorkgroups(1);
        }
    }

    addSortDispatches(pass: GPUComputePassEncoder, vpMat: Mat4) {
        this.sortManager.addDispatches(pass, vpMat);
    }

    destroy() {
        this.bezierBuffer.destroy();
        this.gradBuffer.destroy();
        this.adamBuffer.destroy();
        this.adcBuffer.destroy();
        this.adcScratchBuffer.destroy();
        this.bezierUniformsBuffer.destroy();
        this.pixelLossBuffer.destroy();
        this.binningManager.destroy();
        this.sortManager.destroy();
    }
}
