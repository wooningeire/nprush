import forwardModuleSrc from "./bezier_forward.wgsl?raw";
import { constants, injectWgslConstants } from "../constants";

export class GpuBezierForwardPipelineManager {
    private readonly device: GPUDevice;
    private readonly pipeline: GPUComputePipeline;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private bindGroup: GPUBindGroup | null = null;
    private targetView: GPUTextureView | null = null;
    private dims: { width: number, height: number } = { width: 0, height: 0 };
    private readonly bezierBuffer: GPUBuffer;
    private readonly instanceValsBuffer: GPUBuffer;
    private readonly tileStartsBuffer: GPUBuffer;
    private readonly tileEndsBuffer: GPUBuffer;
    private readonly bezierUniformsBuffer: GPUBuffer;
    private readonly numBeziers: number;
    private readonly brushSampler: GPUSampler;
    private readonly brushTextureView: GPUTextureView;

    constructor({
        device,
        numBeziers,
        bezierBuffer,
        instanceValsBuffer,
        tileStartsBuffer,
        tileEndsBuffer,
        brushTexture,
    }: {
        device: GPUDevice,
        numBeziers: number,
        bezierBuffer: GPUBuffer,
        instanceValsBuffer: GPUBuffer,
        tileStartsBuffer: GPUBuffer,
        tileEndsBuffer: GPUBuffer,
        brushTexture: GPUTexture,
    }) {
        this.device = device;
        this.bezierBuffer = bezierBuffer;
        this.instanceValsBuffer = instanceValsBuffer;
        this.tileStartsBuffer = tileStartsBuffer;
        this.tileEndsBuffer = tileEndsBuffer;
        this.numBeziers = numBeziers;
        this.brushTextureView = brushTexture.createView();

        this.brushSampler = device.createSampler({
            label: "brush sampler",
            addressModeU: "repeat",
            addressModeV: "clamp-to-edge",
            magFilter: "linear",
            minFilter: "linear",
            mipmapFilter: "linear",
        });

        // mat4x4 (64) + dims vec2 + pad + cam_world vec4
        this.bezierUniformsBuffer = device.createBuffer({
            label: "bezier forward uniforms buffer",
            size: 96,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bindGroupLayout = device.createBindGroupLayout({
            label: "bezier forward bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // beziers
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }, // uniforms
                { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } }, // brush_sampler
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } }, // brush_texture
                { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: "rgba8unorm", access: "write-only", viewDimension: "2d" } }, // out color
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // instance_vals
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // tile_starts
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // tile_ends
            ],
        });

        const code = injectWgslConstants(forwardModuleSrc, {
            ...constants,
            NUM_BEZIERS: numBeziers,
        });
        const module = device.createShaderModule({ label: "bezier forward", code });

        this.pipeline = device.createComputePipeline({
            label: "bezier forward compute pipeline",
            layout: device.createPipelineLayout({ 
                label: "bezier forward pipeline layout",
                bindGroupLayouts: [this.bindGroupLayout] 
            }),
            compute: { module, entryPoint: "main" },
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

    writeCameraWorld(x: number, y: number, z: number) {
        this.device.queue.writeBuffer(this.bezierUniformsBuffer, 80, new Float32Array([x, y, z, 1.0]));
    }

    setTarget(targetView: GPUTextureView, width: number, height: number) {
        this.targetView = targetView;
        if (this.dims.width !== width || this.dims.height !== height) {
            this.dims = { width, height };
            this.device.queue.writeBuffer(this.bezierUniformsBuffer, 64, new Float32Array([width, height, 0, 0]));

            this.bindGroup = this.device.createBindGroup({
                label: "bezier forward bind group",
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: this.bezierBuffer } },
                    { binding: 1, resource: { buffer: this.bezierUniformsBuffer } },
                    { binding: 2, resource: this.brushSampler },
                    { binding: 3, resource: this.brushTextureView },
                    { binding: 4, resource: targetView },
                    { binding: 5, resource: { buffer: this.instanceValsBuffer } },
                    { binding: 6, resource: { buffer: this.tileStartsBuffer } },
                    { binding: 7, resource: { buffer: this.tileEndsBuffer } },
                ],
            });
        }
    }

    dispatch(commandEncoder: GPUCommandEncoder, draw: boolean = true, timestampWrites?: NonNullable<GPUComputePassDescriptor["timestampWrites"]>) {
        if (!this.bindGroup || !this.targetView || !draw) return;
        
        const pass = commandEncoder.beginComputePass({
            label: "bezier forward compute pass",
            ...(timestampWrites ? { timestampWrites } : {}),
        });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.dims.width / 16), Math.ceil(this.dims.height / 16));
        pass.end();
    }

    destroy() {
        this.bezierUniformsBuffer.destroy();
    }
}
