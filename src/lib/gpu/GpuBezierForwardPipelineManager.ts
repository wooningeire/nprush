import forwardModuleSrc from "./bezier_forward.wgsl?raw";

export class GpuBezierForwardPipelineManager {
    private readonly device: GPUDevice;
    private readonly pipeline: GPURenderPipeline;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private bindGroup: GPUBindGroup | null = null;
    private targetView: GPUTextureView | null = null;
    private dims: { width: number, height: number } = { width: 0, height: 0 };
    private readonly bezierBuffer: GPUBuffer;
    private readonly bezierUniformsBuffer: GPUBuffer;
    private readonly numBeziers: number;
    private readonly brushSampler: GPUSampler;
    private readonly brushTextureView: GPUTextureView;

    constructor({
        device,
        numBeziers,
        bezierBuffer,
        brushTexture,
    }: {
        device: GPUDevice,
        numBeziers: number,
        bezierBuffer: GPUBuffer,
        brushTexture: GPUTexture,
    }) {
        this.device = device;
        this.bezierBuffer = bezierBuffer;
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

        this.bezierUniformsBuffer = device.createBuffer({
            label: "bezier forward uniforms buffer",
            size: 64 + 16, // mat4x4f + dims (vec2f) + pad (vec2f)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
            ],
        });

        const code = forwardModuleSrc.replace(/NUM_BEZIERS/g, `${numBeziers}u`);
        const module = device.createShaderModule({ label: "bezier forward", code });
        module.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[bezier_forward] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
        });

        this.pipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
            vertex: {
                module,
                entryPoint: "vs_main",
            },
            fragment: {
                module,
                entryPoint: "fs_main",
                targets: [{
                    format: "rgba8unorm",
                    blend: {
                        color: {
                            operation: "add",
                            srcFactor: "one",
                            dstFactor: "one-minus-src-alpha",
                        },
                        alpha: {
                            operation: "add",
                            srcFactor: "one",
                            dstFactor: "one-minus-src-alpha",
                        },
                    },
                }],
            },
            primitive: {
                topology: "triangle-strip",
            },
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

    setTarget(targetView: GPUTextureView, width: number, height: number) {
        this.targetView = targetView;
        if (this.dims.width !== width || this.dims.height !== height) {
            this.dims = { width, height };
            this.device.queue.writeBuffer(this.bezierUniformsBuffer, 64, new Float32Array([width, height]));
        }

        this.bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bezierBuffer } },
                { binding: 1, resource: { buffer: this.bezierUniformsBuffer } },
                { binding: 2, resource: this.brushSampler },
                { binding: 3, resource: this.brushTextureView },
            ],
        });
    }

    dispatch(commandEncoder: GPUCommandEncoder, clear: boolean = true) {
        if (!this.bindGroup || !this.targetView) return;
        const pass = commandEncoder.beginRenderPass({
            label: "bezier forward pass",
            colorAttachments: [
                {
                    view: this.targetView,
                    clearValue: clear ? { r: 0.0, g: 0.0, b: 0.0, a: 0.0 } : undefined,
                    loadOp: clear ? "clear" : "load",
                    storeOp: "store",
                },
            ],
        });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.draw(4, this.numBeziers);
        pass.end();
    }
}
