import depthAwareBlurModuleSrc from "./depth_aware_blur.wgsl?raw";

export class GpuDepthAwareBlurPipelineManager {
    private readonly device: GPUDevice;
    private readonly pipeline: GPUComputePipeline;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private readonly paramsBuffer: GPUBuffer;

    constructor(device: GPUDevice) {
        this.device = device;

        this.paramsBuffer = device.createBuffer({
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba8unorm" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
        });

        this.pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: depthAwareBlurModuleSrc }),
                entryPoint: "main",
            },
        });
    }

    blur(
        commandEncoder: GPUCommandEncoder,
        colorView: GPUTextureView,
        depthView: GPUTextureView,
        dstView: GPUTextureView,
        width: number,
        height: number,
        radius: number,
    ) {
        this.device.queue.writeBuffer(this.paramsBuffer, 0, new Int32Array([radius]));

        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: colorView },
                { binding: 1, resource: depthView },
                { binding: 2, resource: dstView },
                { binding: 3, resource: { buffer: this.paramsBuffer } },
            ],
        });

        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
        pass.end();
    }
}
