import sortModuleSrc from "./splat_sort.wgsl?raw";
import { injectWgslConstants, constants } from "../constants.ts";
import type { Mat4 } from "wgpu-matrix";

export class GpuSplatDepthSortManager {
    private readonly device: GPUDevice;

    private readonly nSplats: number;


    readonly sortKeysPingPongBufferA: GPUBuffer;
    readonly sortKeysPingPongBufferB: GPUBuffer;
    readonly sortIndicesPingPongBufferA: GPUBuffer;
    readonly sortIndicesPingPongBufferB: GPUBuffer;
    readonly sortKeysBuffer: GPUBuffer;
    readonly sortIndicesBuffer: GPUBuffer;
    readonly sortUniformsBuffer: GPUBuffer;

    private readonly sortBindGroupAtoB: GPUBindGroup;
    private readonly sortBindGroupBtoA: GPUBindGroup;
    private readonly histBuffer: GPUBuffer;

    private readonly radixInitPipeline: GPUComputePipeline;
    private readonly radixCountPipeline: GPUComputePipeline;
    private readonly radixScanPipeline: GPUComputePipeline;
    private readonly radixScatterPipeline: GPUComputePipeline;


    constructor({
        device,
        nSplats,
        nParams,
        splatBuffer,
    }: {
        device: GPUDevice,
        nSplats: number,
        nParams: number,
        splatBuffer: GPUBuffer,
    }) {
        this.device = device;

        this.nSplats = nSplats;


        this.sortKeysPingPongBufferA = device.createBuffer({
            label: "splat sort keys ping pong A",
            size: nSplats * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        this.sortKeysPingPongBufferB = device.createBuffer({
            label: "splat sort keys ping pong B",
            size: nSplats * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        this.sortIndicesPingPongBufferA = device.createBuffer({
            label: "splat sort indices ping pong A",
            size: nSplats * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        this.sortIndicesPingPongBufferB = device.createBuffer({
            label: "splat sort indices ping pong B",
            size: nSplats * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        // Radix sort: 4 passes (shift 0/8/16/24), result always in A after 4 passes.
        this.sortKeysBuffer = this.sortKeysPingPongBufferA;
        this.sortIndicesBuffer = this.sortIndicesPingPongBufferA;

        // hist[digit * W + wg_id]: 256 buckets × W workgroups, 4 bytes each.
        const sortWg = Math.ceil(nSplats / 256);
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
        const sortModule = device.createShaderModule({
            label: "splat sort",
            code: injectWgslConstants(sortModuleSrc, {
                ...constants,
                NUM_SPLATS: nSplats,
                NUM_SPLATS_PLUS_ONE: nSplats + 1,
                NUM_SPLATS_MINUS_ONE: nSplats - 1,
                NUM_SPLATS_DIV_32: Math.ceil(nSplats / 32),
                NUM_PARAMS: nParams,
            }),
        });
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


        this.sortBindGroupBtoA = device.createBindGroup({
            label: "splat sort bind group B to A",
            layout: sortBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: splatBuffer } },
                { binding: 1, resource: { buffer: this.sortKeysPingPongBufferB } },
                { binding: 2, resource: { buffer: this.sortIndicesPingPongBufferB } },
                { binding: 3, resource: { buffer: this.sortKeysPingPongBufferA } },
                { binding: 4, resource: { buffer: this.sortIndicesPingPongBufferA } },
                { binding: 5, resource: { buffer: this.sortUniformsBuffer, size: 96 } },
                { binding: 6, resource: { buffer: this.histBuffer } },
            ],
        });

        this.sortBindGroupAtoB = device.createBindGroup({
            label: "splat sort bind group A to B",
            layout: sortBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: splatBuffer } },
                { binding: 1, resource: { buffer: this.sortKeysPingPongBufferA } },
                { binding: 2, resource: { buffer: this.sortIndicesPingPongBufferA } },
                { binding: 3, resource: { buffer: this.sortKeysPingPongBufferB } },
                { binding: 4, resource: { buffer: this.sortIndicesPingPongBufferB } },
                { binding: 5, resource: { buffer: this.sortUniformsBuffer, size: 96 } },
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

        const nWorkgroups = Math.ceil(this.nSplats / 256);

        pass.setPipeline(this.radixInitPipeline);
        pass.setBindGroup(0, this.sortBindGroupBtoA, [0]);
        pass.dispatchWorkgroups(nWorkgroups);

        for (let i = 0; i < 4; i++) {
            const sourceBuffer = i % 2 === 0 ? this.sortBindGroupAtoB : this.sortBindGroupBtoA;
            const offset = i * 256;

            pass.setPipeline(this.radixCountPipeline);
            pass.setBindGroup(0, sourceBuffer, [offset]);
            pass.dispatchWorkgroups(nWorkgroups);

            pass.setPipeline(this.radixScanPipeline);
            pass.setBindGroup(0, sourceBuffer, [offset]);
            pass.dispatchWorkgroups(1);

            pass.setPipeline(this.radixScatterPipeline);
            pass.setBindGroup(0, sourceBuffer, [offset]);
            pass.dispatchWorkgroups(nWorkgroups);
        }
    }


    destroy() {
        this.sortKeysPingPongBufferA.destroy();
        this.sortKeysPingPongBufferB.destroy();
        this.sortIndicesPingPongBufferA.destroy();
        this.sortIndicesPingPongBufferB.destroy();

        this.sortUniformsBuffer.destroy();
        this.histBuffer.destroy();
    }
}