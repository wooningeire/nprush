import forwardModuleSrc from "./splat_forward.wgsl?raw";
import type { Mat4 } from "wgpu-matrix";
import { constants, injectWgslConstants } from "../constants";

export class GpuSplatForwardPipelineManager {
    private readonly device: GPUDevice;
    private readonly pipeline: GPUComputePipeline;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private bindGroup: GPUBindGroup | null = null;
    private dims: { width: number, height: number } = { width: 0, height: 0 };
    private readonly splatBuffer: GPUBuffer;
    private readonly instanceValsBuffer: GPUBuffer;
    private readonly tileStartsBuffer: GPUBuffer;
    private readonly tileEndsBuffer: GPUBuffer;
    private readonly uniformsBuffer: GPUBuffer;
    private readonly numSplats: number;
    private targetColorView: GPUTextureView | null = null;
    private targetDepthView: GPUTextureView | null = null;

    constructor({
        device,
        numSplats,
        splatBuffer,
        instanceValsBuffer,
        tileStartsBuffer,
        tileEndsBuffer,
    }: {
        device: GPUDevice,
        numSplats: number,
        splatBuffer: GPUBuffer,
        instanceValsBuffer: GPUBuffer,
        tileStartsBuffer: GPUBuffer,
        tileEndsBuffer: GPUBuffer,
    }) {
        this.device = device;
        this.splatBuffer = splatBuffer;
        this.instanceValsBuffer = instanceValsBuffer;
        this.tileStartsBuffer = tileStartsBuffer;
        this.tileEndsBuffer = tileEndsBuffer;
        this.numSplats = numSplats;

        // mat4x4 (64) + dims vec2 (8) + pad (8) + cam_world vec4 (16) = 96
        this.uniformsBuffer = device.createBuffer({
            label: "splat forward uniforms buffer",
            size: 96,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bindGroupLayout = device.createBindGroupLayout({
            label: "splat forward bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // splats
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }, // uniforms
                { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: "rgba8unorm", access: "write-only", viewDimension: "2d" } }, // out color
                { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: "rgba8unorm", access: "write-only", viewDimension: "2d" } }, // out depth
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // instance_vals
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // tile_starts
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // tile_ends
            ],
        });

        const code = injectWgslConstants(forwardModuleSrc, {
            ...constants,
            NUM_SPLATS: numSplats,
        });
        const module = device.createShaderModule({ label: "splat forward compute", code });

        this.pipeline = device.createComputePipeline({
            label: "splat forward compute pipeline",
            layout: device.createPipelineLayout({ 
                label: "splat forward pipeline layout",
                bindGroupLayouts: [this.bindGroupLayout] 
            }),
            compute: { module, entryPoint: "main" },
        });
    }

    writeVPMatrix(mat: Mat4) {
        this.device.queue.writeBuffer(
            this.uniformsBuffer,
            0,
            (mat as Float32Array).buffer,
            (mat as Float32Array).byteOffset,
            (mat as Float32Array).byteLength
        );
    }

    writeCameraWorld(x: number, y: number, z: number) {
        this.device.queue.writeBuffer(this.uniformsBuffer, 80, new Float32Array([x, y, z, 1.0]));
    }

    setTarget(targetColorView: GPUTextureView, targetDepthView: GPUTextureView, width: number, height: number) {
        this.targetColorView = targetColorView;
        this.targetDepthView = targetDepthView;
        
        if (this.dims.width !== width || this.dims.height !== height) {
            this.dims = { width, height };
            this.device.queue.writeBuffer(this.uniformsBuffer, 64, new Float32Array([width, height, 0, 0]));

            this.bindGroup = this.device.createBindGroup({
                label: "splat forward bind group",
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.splatBuffer } },
                    { binding: 1, resource: { buffer: this.uniformsBuffer } },
                    { binding: 2, resource: targetColorView },
                    { binding: 3, resource: targetDepthView },
                    { binding: 4, resource: { buffer: this.instanceValsBuffer } },
                    { binding: 5, resource: { buffer: this.tileStartsBuffer } },
                    { binding: 6, resource: { buffer: this.tileEndsBuffer } },
                ],
            });
        }
    }

    dispatch(commandEncoder: GPUCommandEncoder, clear: boolean = false, draw: boolean = true, timestampWrites?: NonNullable<GPUComputePassDescriptor["timestampWrites"]>) {
        if (!this.bindGroup || !draw) return;

        const pass = commandEncoder.beginComputePass({
            label: "splat forward compute pass",
            ...(timestampWrites ? { timestampWrites } : {}),
        });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.dims.width / 16), Math.ceil(this.dims.height / 16));
        pass.end();
    }

    destroy() {
        this.uniformsBuffer.destroy();
    }
}
