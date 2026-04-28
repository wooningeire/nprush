import forwardModuleSrc from "./splat_forward.wgsl?raw";
import type { Mat4 } from "wgpu-matrix";

export class GpuSplatForwardPipelineManager {
    private readonly device: GPUDevice;
    private readonly pipeline: GPURenderPipeline;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private bindGroup: GPUBindGroup | null = null;
    private dims: { width: number, height: number } = { width: 0, height: 0 };
    private readonly splatBuffer: GPUBuffer;
    private readonly uniformsBuffer: GPUBuffer;
    private readonly numSplats: number;
    private targetColorView: GPUTextureView | null = null;
    private targetDepthView: GPUTextureView | null = null;

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

        // VP matrix (64 bytes) + dims (8 bytes) + pad (8 bytes) = 80 bytes
        this.uniformsBuffer = device.createBuffer({
            size: 80,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
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

    writeVPMatrix(mat: Mat4) {
        this.device.queue.writeBuffer(
            this.uniformsBuffer,
            0,
            (mat as Float32Array).buffer,
            (mat as Float32Array).byteOffset,
            (mat as Float32Array).byteLength
        );
    }

    setTarget(targetColorView: GPUTextureView, targetDepthView: GPUTextureView, width: number, height: number) {
        this.targetColorView = targetColorView;
        this.targetDepthView = targetDepthView;
        if (this.dims.width !== width || this.dims.height !== height) {
            this.dims = { width, height };
            // Write dims at offset 64 (after the mat4x4)
            this.device.queue.writeBuffer(this.uniformsBuffer, 64, new Float32Array([width, height]));
        }
    }

    dispatch(commandEncoder: GPUCommandEncoder, clear: boolean = false) {
        if (!this.targetColorView || !this.targetDepthView || !this.bindGroup) return;
        const pass = commandEncoder.beginRenderPass({
            label: "splat forward render pass",
            colorAttachments: [
                {
                    view: this.targetColorView,
                    clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1.0 },
                    loadOp: clear ? "clear" : "load",
                    storeOp: "store",
                },
                {
                    view: this.targetDepthView,
                    clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
                    loadOp: clear ? "clear" : "load",
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
