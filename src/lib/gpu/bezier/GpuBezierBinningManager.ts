import binningModuleSrc from "./bezier_binning.wgsl?raw";
import { injectWgslConstants, constants } from "../constants.ts";
import type { Mat4 } from "wgpu-matrix";

export class GpuBezierBinningManager {
    private readonly device: GPUDevice;
    private readonly numBeziers: number;
    private readonly maxInstances: number;

    readonly binningAtomicBuffer: GPUBuffer;
    readonly instanceKeysBufferA: GPUBuffer;
    readonly instanceKeysBufferB: GPUBuffer;
    readonly instanceValsBufferA: GPUBuffer;
    readonly instanceValsBufferB: GPUBuffer;
    readonly binningUniformsBuffer: GPUBuffer;
    readonly binningSortUniformsBuffer: GPUBuffer;
    readonly binningHistBuffer: GPUBuffer;
    readonly tileStartsBuffer: GPUBuffer;
    readonly tileEndsBuffer: GPUBuffer;

    private readonly binInstantiatePipeline: GPUComputePipeline;
    private readonly binCountPipeline: GPUComputePipeline;
    private readonly binScanPipeline: GPUComputePipeline;
    private readonly binScatterPipeline: GPUComputePipeline;
    private readonly binCalcRangesPipeline: GPUComputePipeline;

    private readonly binInstantiateBindGroup: GPUBindGroup;
    private readonly binSortBindGroupAtoB: GPUBindGroup;
    private readonly binSortBindGroupBtoA: GPUBindGroup;
    private readonly binCalcRangesBindGroup: GPUBindGroup;

    constructor({
        device,
        numBeziers,
        numParams,
        bezierBuffer,
    }: {
        device: GPUDevice,
        numBeziers: number,
        numParams: number,
        bezierBuffer: GPUBuffer,
    }) {
        this.device = device;
        this.numBeziers = numBeziers;
        this.maxInstances = numBeziers * 16;
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

        const binningModule = device.createShaderModule({
            label: "bezier binning",
            code: injectWgslConstants(binningModuleSrc, {
                ...constants,
                NUM_BEZIERS: numBeziers,
                NUM_PARAMS: numParams,
            }),
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
                { binding: 0, resource: { buffer: bezierBuffer } },
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

    addDispatches(pass: GPUComputePassEncoder, vpMat: Mat4, width: number, height: number) {
        const gridWidth = Math.ceil(width / 16);
        const gridHeight = Math.ceil(height / 16);

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

        // Instantiate: map each bezier to all overlapping tiles
        pass.setPipeline(this.binInstantiatePipeline);
        pass.setBindGroup(0, this.binInstantiateBindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.numBeziers / 256));

        // 8-pass radix sort
        const sortWg = Math.ceil(this.maxInstances / 256);
        for (let i = 0; i < 8; i++) {
            const bg = (i % 2 === 0) ? this.binSortBindGroupAtoB : this.binSortBindGroupBtoA;
            const offset = i * 256;

            pass.setPipeline(this.binCountPipeline);
            pass.setBindGroup(0, bg, [offset]);
            pass.dispatchWorkgroups(sortWg);

            pass.setPipeline(this.binScanPipeline);
            pass.setBindGroup(0, bg, [offset]);
            pass.dispatchWorkgroups(1);

            pass.setPipeline(this.binScatterPipeline);
            pass.setBindGroup(0, bg, [offset]);
            pass.dispatchWorkgroups(sortWg);
        }

        // Compute tile_starts and tile_ends
        pass.setPipeline(this.binCalcRangesPipeline);
        pass.setBindGroup(0, this.binCalcRangesBindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.maxInstances / 256));
    }

    destroy() {
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
