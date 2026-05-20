import shaderSrc from "./luminance_extract.wgsl?raw";

/**
 * Dispatches a compute pass that converts an RGBA color texture into a
 * grayscale luminance texture (BT.709 weights).
 */
export class GpuLuminanceExtractManager {
    private readonly device: GPUDevice;
    private readonly pipeline: GPUComputePipeline;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private bindGroup: GPUBindGroup | null = null;

    constructor(device: GPUDevice) {
        this.device = device;

        this.bindGroupLayout = device.createBindGroupLayout({
            label: "luminance extract bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba8unorm" } },
            ],
        });

        this.pipeline = device.createComputePipeline({
            label: "luminance extract pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ label: "luminance extract", code: shaderSrc }),
                entryPoint: "main",
            },
        });
    }

    setTextures(src: GPUTextureView, dst: GPUTextureView) {
        this.bindGroup = this.device.createBindGroup({
            label: "luminance extract bind group",
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: src },
                { binding: 1, resource: dst },
            ],
        });
    }

    addDispatches(commandEncoder: GPUCommandEncoder, width: number, height: number, timestampWrites?: GPUComputePassTimestampWrites) {
        if (!this.bindGroup) return;
        const pass = commandEncoder.beginComputePass({
            label: "luminance extract pass",
            ...(timestampWrites ? { timestampWrites } : {}),
        });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
        pass.end();
    }

    destroy() {
        // Pipeline and bind group layout are GC'd; nothing to manually destroy.
    }
}
