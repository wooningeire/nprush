import forwardModuleSrc from "./splat_forward.wgsl?raw";

export class GpuSplatForwardPipelineManager {
    private readonly device: GPUDevice;
    private readonly pipeline: GPURenderPipeline;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private bindGroup: GPUBindGroup | null = null;
    private dims: { width: number, height: number } = { width: 0, height: 0 };
    private readonly splatBuffer: GPUBuffer;
    private readonly uniformsBuffer: GPUBuffer;
    private readonly numSplats: number;
    private targetView: GPUTextureView | null = null;

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
        this.numSplats = numSplats;

        this.uniformsBuffer = device.createBuffer({
            size: 8, // vec2f
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
            ],
        });

        const code = forwardModuleSrc.replace(/NUM_SPLATS/g, `${numSplats}u`);
        const module = device.createShaderModule({ label: "splat forward render", code });
        module.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[splat_forward] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
        });

        this.pipeline = device.createRenderPipeline({
            label: "splat forward render pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
            vertex: { module, entryPoint: "vert" },
            fragment: {
                module,
                entryPoint: "frag",
                targets: [
                    {
                        format: "rgba8unorm",
                        blend: {
                            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
                            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
                        },
                    },
                ],
            },
            primitive: { topology: "triangle-list" },
        });

        this.bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.splatBuffer } },
                { binding: 1, resource: { buffer: this.uniformsBuffer } },
            ],
        });
    }

    setTarget(targetView: GPUTextureView, width: number, height: number) {
        this.targetView = targetView;
        if (this.dims.width !== width || this.dims.height !== height) {
            this.dims = { width, height };
            this.device.queue.writeBuffer(this.uniformsBuffer, 0, new Float32Array([width, height]));
        }
    }

    dispatch(commandEncoder: GPUCommandEncoder) {
        if (!this.targetView || !this.bindGroup) return;
        const pass = commandEncoder.beginRenderPass({
            label: "splat forward render pass",
            colorAttachments: [
                {
                    view: this.targetView,
                    clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
                    loadOp: "clear",
                    storeOp: "store",
                },
            ],
        });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.draw(6, this.numSplats);
        pass.end();
    }
}
