import forwardModuleSrc from "./bezier_forward.wgsl?raw";

export class GpuBezierForwardPipelineManager {
    private readonly device: GPUDevice;
    private readonly pipeline: GPUComputePipeline;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private bindGroup: GPUBindGroup | null = null;
    private dims: { width: number, height: number } = { width: 0, height: 0 };
    private readonly bezierBuffer: GPUBuffer;
    private readonly bezierUniformsBuffer: GPUBuffer;

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

        this.bezierUniformsBuffer = device.createBuffer({
            label: "bezier forward uniforms buffer",
            size: 64 + 16, // mat4x4f + dims (vec2f) + pad (vec2f)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba8unorm" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
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
        if (this.dims.width !== width || this.dims.height !== height) {
            this.dims = { width, height };
            this.device.queue.writeBuffer(this.bezierUniformsBuffer, 64, new Float32Array([width, height]));
        }

        this.bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bezierBuffer } },
                { binding: 1, resource: targetView },
                { binding: 2, resource: { buffer: this.bezierUniformsBuffer } },
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
