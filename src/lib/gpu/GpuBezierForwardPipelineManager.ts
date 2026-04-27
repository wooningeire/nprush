import forwardModuleSrc from "./bezier_forward.wgsl?raw";

export class GpuBezierForwardPipelineManager {
    private readonly device: GPUDevice;
    private readonly pipeline: GPUComputePipeline;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private bindGroup: GPUBindGroup | null = null;
    private dims: { width: number, height: number } = { width: 0, height: 0 };
    private readonly bezierBuffer: GPUBuffer;

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
        this.bezierBuffer = bezierBuffer;

        this.bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba8unorm" } },
            ],
        });

        const code = forwardModuleSrc.replace(/NUM_BEZIERS/g, `${numBeziers}u`);
        const module = device.createShaderModule({ label: "bezier forward", code });
        module.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[bezier_forward] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
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
                { binding: 0, resource: { buffer: this.bezierBuffer } },
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
