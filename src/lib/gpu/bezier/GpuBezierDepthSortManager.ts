import sortModuleSrc from "./bezier_sort.wgsl?raw";
import { injectWgslConstants, constants } from "../constants.ts";
import type { Mat4 } from "wgpu-matrix";

export class GpuBezierDepthSortManager {
    private readonly device: GPUDevice;
    private readonly numBeziers: number;

    readonly sortKeysBufferA: GPUBuffer;
    readonly sortKeysBufferB: GPUBuffer;
    readonly sortIndicesBufferA: GPUBuffer;
    readonly sortIndicesBufferB: GPUBuffer;
    readonly sortUniformsBuffer: GPUBuffer;
    readonly histBuffer: GPUBuffer;

    private readonly radixInitPipeline: GPUComputePipeline;
    private readonly radixCountPipeline: GPUComputePipeline;
    private readonly radixScanPipeline: GPUComputePipeline;
    private readonly radixScatterPipeline: GPUComputePipeline;

    private readonly sortBindGroupAtoB: GPUBindGroup;
    private readonly sortBindGroupBtoA: GPUBindGroup;

    constructor({
        device,
        numBeziers,
        bezierBuffer,
    }: {
        device: GPUDevice,
        numBeziers: number,
        bezierBuffer: GPUBuffer,
    }) {
        this.device = device;
        this.numBeziers = numBeziers;

        this.sortKeysBufferA = device.createBuffer({
            label: "bezier sort keys A",
            size: numBeziers * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        this.sortKeysBufferB = device.createBuffer({
            label: "bezier sort keys B",
            size: numBeziers * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        this.sortIndicesBufferA = device.createBuffer({
            label: "bezier sort indices A",
            size: numBeziers * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        this.sortIndicesBufferB = device.createBuffer({
            label: "bezier sort indices B",
            size: numBeziers * 4,
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

        const sortWg = Math.ceil(numBeziers / 256);
        this.histBuffer = device.createBuffer({
            label: "bezier sort histogram",
            size: 256 * sortWg * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        const sortModule = device.createShaderModule({
            label: "bezier sort",
            code: injectWgslConstants(sortModuleSrc, {
                ...constants,
                NUM_BEZIERS: numBeziers,
            }),
        });

        const sortBindGroupLayout = device.createBindGroupLayout({
            label: "bezier sort layout",
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

        const sortPipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [sortBindGroupLayout] });

        this.radixInitPipeline = device.createComputePipeline({
            label: "bezier radix init pipeline",
            layout: sortPipelineLayout,
            compute: { module: sortModule, entryPoint: "init_keys" },
        });
        this.radixCountPipeline = device.createComputePipeline({
            label: "bezier radix count pipeline",
            layout: sortPipelineLayout,
            compute: { module: sortModule, entryPoint: "count" },
        });
        this.radixScanPipeline = device.createComputePipeline({
            label: "bezier radix scan pipeline",
            layout: sortPipelineLayout,
            compute: { module: sortModule, entryPoint: "scan" },
        });
        this.radixScatterPipeline = device.createComputePipeline({
            label: "bezier radix scatter pipeline",
            layout: sortPipelineLayout,
            compute: { module: sortModule, entryPoint: "scatter" },
        });

        this.sortBindGroupAtoB = device.createBindGroup({
            label: "bezier sort A to B",
            layout: sortBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bezierBuffer } },
                { binding: 1, resource: { buffer: this.sortKeysBufferA } },
                { binding: 2, resource: { buffer: this.sortIndicesBufferA } },
                { binding: 3, resource: { buffer: this.sortKeysBufferB } },
                { binding: 4, resource: { buffer: this.sortIndicesBufferB } },
                { binding: 5, resource: { buffer: this.sortUniformsBuffer, size: 256 } },
                { binding: 6, resource: { buffer: this.histBuffer } },
            ],
        });
        this.sortBindGroupBtoA = device.createBindGroup({
            label: "bezier sort B to A",
            layout: sortBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bezierBuffer } },
                { binding: 1, resource: { buffer: this.sortKeysBufferB } },
                { binding: 2, resource: { buffer: this.sortIndicesBufferB } },
                { binding: 3, resource: { buffer: this.sortKeysBufferA } },
                { binding: 4, resource: { buffer: this.sortIndicesBufferA } },
                { binding: 5, resource: { buffer: this.sortUniformsBuffer, size: 256 } },
                { binding: 6, resource: { buffer: this.histBuffer } },
            ],
        });
    }

    addDispatches(pass: GPUComputePassEncoder, vpMat: Mat4) {
        const vpData = vpMat as Float32Array;
        for (let i = 0; i < 4; i++) {
            this.device.queue.writeBuffer(
                this.sortUniformsBuffer, i * 256,
                vpData.buffer, vpData.byteOffset, vpData.byteLength,
            );
        }

        const wg = Math.ceil(this.numBeziers / 256);

        pass.setPipeline(this.radixInitPipeline);
        pass.setBindGroup(0, this.sortBindGroupBtoA, [0]);
        pass.dispatchWorkgroups(wg);

        for (let i = 0; i < 4; i++) {
            const bg = (i % 2 === 0) ? this.sortBindGroupAtoB : this.sortBindGroupBtoA;
            const offset = i * 256;

            pass.setPipeline(this.radixCountPipeline);
            pass.setBindGroup(0, bg, [offset]);
            pass.dispatchWorkgroups(wg);

            pass.setPipeline(this.radixScanPipeline);
            pass.setBindGroup(0, bg, [offset]);
            pass.dispatchWorkgroups(1);

            pass.setPipeline(this.radixScatterPipeline);
            pass.setBindGroup(0, bg, [offset]);
            pass.dispatchWorkgroups(wg);
        }
    }

    destroy() {
        this.sortKeysBufferA.destroy();
        this.sortKeysBufferB.destroy();
        this.sortIndicesBufferA.destroy();
        this.sortIndicesBufferB.destroy();
        this.sortUniformsBuffer.destroy();
        this.histBuffer.destroy();
    }
}
