import blurModuleSrc from "./blur.wgsl?raw";

export class GpuBlurPipelineManager {
    private readonly device: GPUDevice;
    private readonly pipeline: GPUComputePipeline;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private readonly hParamsBuffer: GPUBuffer;
    private readonly vParamsBuffer: GPUBuffer;

    constructor(device: GPUDevice) {
        this.device = device;

        this.hParamsBuffer = device.createBuffer({
            size: 48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.vParamsBuffer = device.createBuffer({
            size: 48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba8unorm" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
        });

        this.pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: blurModuleSrc }),
                entryPoint: "main",
            },
        });
    }

    blur(
        commandEncoder: GPUCommandEncoder,
        srcView: GPUTextureView,
        dstView: GPUTextureView,
        tempView: GPUTextureView,
        width: number,
        height: number,
        radius: number,
        sigma: number,
        isSrgb: boolean = false
    ) {
        // Horizontal pass
        const hFlags = isSrgb ? 1 : 0; // In: sRGB, Out: Linear
        this.device.queue.writeBuffer(this.hParamsBuffer, 0, new Int32Array([1, 0, radius, hFlags]));
        this.device.queue.writeBuffer(this.hParamsBuffer, 16, new Float32Array([sigma]));
        
        const hBindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: srcView },
                { binding: 1, resource: tempView },
                { binding: 2, resource: { buffer: this.hParamsBuffer } },
            ],
        });

        // Vertical pass
        const vFlags = isSrgb ? 2 : 0; // In: Linear, Out: sRGB
        this.device.queue.writeBuffer(this.vParamsBuffer, 0, new Int32Array([0, 1, radius, vFlags]));
        this.device.queue.writeBuffer(this.vParamsBuffer, 16, new Float32Array([sigma]));
        
        const vBindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: tempView },
                { binding: 1, resource: dstView },
                { binding: 2, resource: { buffer: this.vParamsBuffer } },
            ],
        });

        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        
        pass.setBindGroup(0, hBindGroup);
        pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));

        pass.setBindGroup(0, vBindGroup);
        pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));

        pass.end();
    }
}
