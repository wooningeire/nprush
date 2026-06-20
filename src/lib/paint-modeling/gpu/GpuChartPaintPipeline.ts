import shaderSource from "./paint_chart_write.wgsl?raw";
import type { Vec2 } from "../types.ts";

export type GpuChartPaintSample = {
    point: Vec2,
    depth: number,
};

export type GpuChartPaintOptions = {
    radius: number,
    requireCoverage: boolean,
    depthWriteMode: "blend" | "replace",
    coverageEpsilon: number,
    minDepth: number,
};

const WORKGROUP_SIZE = 64;
const FLOATS_PER_SAMPLE = 4;
const PARAM_FLOATS = 8;

export class GpuChartPaintPipeline {
    private readonly device: GPUDevice;
    private readonly pipeline: GPUComputePipeline;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private sampleBuffer: GPUBuffer | null = null;
    private sampleCapacity = 0;
    private paramsBuffer: GPUBuffer;

    constructor(device: GPUDevice) {
        this.device = device;
        this.bindGroupLayout = device.createBindGroupLayout({
            label: "paint chart write bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
            ],
        });
        const module = device.createShaderModule({
            label: "paint chart write shader",
            code: shaderSource,
        });
        this.pipeline = device.createComputePipeline({
            label: "paint chart write pipeline",
            layout: device.createPipelineLayout({
                label: "paint chart write pipeline layout",
                bindGroupLayouts: [this.bindGroupLayout],
            }),
            compute: {
                module,
                entryPoint: "main",
            },
        });
        this.paramsBuffer = device.createBuffer({
            label: "paint chart write params",
            size: PARAM_FLOATS * Float32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
    }

    dispatch(
        encoder: GPUCommandEncoder,
        fieldBuffer: GPUBuffer,
        width: number,
        height: number,
        samples: GpuChartPaintSample[],
        options: GpuChartPaintOptions,
    ) {
        if (samples.length === 0) return;
        const sampleBuffer = this.writeSamples(samples);
        this.writeParams(width, height, samples.length, options);
        const bindGroup = this.device.createBindGroup({
            label: "paint chart write bind group",
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: fieldBuffer } },
                { binding: 1, resource: { buffer: sampleBuffer } },
                { binding: 2, resource: { buffer: this.paramsBuffer } },
            ],
        });
        const pass = encoder.beginComputePass({ label: "paint chart write pass" });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(width * height / WORKGROUP_SIZE));
        pass.end();
    }

    destroy() {
        this.sampleBuffer?.destroy();
        this.sampleBuffer = null;
        this.paramsBuffer.destroy();
    }

    private writeSamples(samples: GpuChartPaintSample[]): GPUBuffer {
        if (!this.sampleBuffer || samples.length > this.sampleCapacity) {
            this.sampleBuffer?.destroy();
            this.sampleCapacity = Math.max(samples.length, this.sampleCapacity * 2, 16);
            this.sampleBuffer = this.device.createBuffer({
                label: "paint chart write samples",
                size: this.sampleCapacity * FLOATS_PER_SAMPLE * Float32Array.BYTES_PER_ELEMENT,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
        }

        const data = new Float32Array(samples.length * FLOATS_PER_SAMPLE);
        for (let index = 0; index < samples.length; index++) {
            const sample = samples[index];
            const offset = index * FLOATS_PER_SAMPLE;
            data[offset] = sample.point.x;
            data[offset + 1] = sample.point.y;
            data[offset + 2] = sample.depth;
        }
        this.device.queue.writeBuffer(this.sampleBuffer, 0, data);
        return this.sampleBuffer;
    }

    private writeParams(
        width: number,
        height: number,
        sampleCount: number,
        options: GpuChartPaintOptions,
    ) {
        this.device.queue.writeBuffer(this.paramsBuffer, 0, new Float32Array([
            width,
            height,
            sampleCount,
            options.requireCoverage ? 1 : 0,
            options.depthWriteMode === "replace" ? 1 : 0,
            options.radius,
            options.coverageEpsilon,
            options.minDepth,
        ]));
    }
}