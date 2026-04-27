import forwardModuleSrc from "./splat_forward.wgsl?raw";

export class GpuSplatForwardPipelineManager {
    private readonly device: GPUDevice;
    private readonly pipeline: GPUComputePipeline;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private bindGroup: GPUBindGroup | null = null;
    private dims: { width: number, height: number } = { width: 0, height: 0 };
    private readonly splatBuffer: GPUBuffer;

    constructor({
        device,
        numSplats,
        splatBuffer,
    }: {
        device: GPUDevice,
        numSplats: number,
        splatBuffer: GPUBuffer,
    }) {
        this.device = device;
        this.splatBuffer = splatBuffer;

        this.bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba8unorm" } },
            ],
        });

        const code = forwardModuleSrc.replace(/NUM_SPLATS/g, `${numSplats}u`);
        const module = device.createShaderModule({ label: "splat forward", code });
        module.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[splat_forward] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
        });

        this.pipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
            compute: { module, entryPoint: "main" },
        });
    }

    setTarget(targetView: GPUTextureView, width: number, height: number) {
        this.dims = { width, height };
        this.bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.splatBuffer } },
                { binding: 1, resource: targetView },
            ],
        });
    }

    dispatch(commandEncoder: GPUCommandEncoder) {
        if (!this.bindGroup) return;
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.dims.width / 16), Math.ceil(this.dims.height / 16));
        pass.end();
    }
}
