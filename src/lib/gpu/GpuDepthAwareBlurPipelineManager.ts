import depthAwareBlurModuleSrc from "./depth_aware_blur.wgsl?raw";

export class GpuDepthAwareBlurPipelineManager {
    private readonly device: GPUDevice;
    private readonly pipeline: GPUComputePipeline;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private readonly paramsBuffer: GPUBuffer;

    constructor(device: GPUDevice) {
        this.device = device;

        this.paramsBuffer = device.createBuffer({
            label: "depth aware blur params buffer",
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bindGroupLayout = device.createBindGroupLayout({
            label: "depth aware blur bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba8unorm" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
        });

        this.pipeline = device.createComputePipeline({
            label: "depth aware blur compute pipeline",
            layout: device.createPipelineLayout({ 
                label: "depth aware blur pipeline layout",
                bindGroupLayouts: [this.bindGroupLayout] 
            }),
            compute: {
                module: device.createShaderModule({ label: "depth aware blur shader", code: depthAwareBlurModuleSrc }),
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
            label: "depth aware blur bind group",
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: colorView },
                { binding: 1, resource: depthView },
                { binding: 2, resource: dstView },
                { binding: 3, resource: { buffer: this.paramsBuffer } },
            ],
        });

        const pass = commandEncoder.beginComputePass({
            label: "depth aware blur pass",
        });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
        pass.end();
    }
}
