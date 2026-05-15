import forwardModuleSrc from "./bezier_forward.wgsl?raw";
import { constants, injectWgslConstants } from "../constants";

export class GpuBezierForwardPipelineManager {
    private readonly device: GPUDevice;
    private readonly pipeline1: GPURenderPipeline;
    private readonly pipeline2: GPURenderPipeline;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private bindGroup: GPUBindGroup | null = null;
    private targetView: GPUTextureView | null = null;
    private dims: { width: number, height: number } = { width: 0, height: 0 };
    private readonly bezierBuffer: GPUBuffer;
    private readonly sortOrderBuffer: GPUBuffer;
    private readonly bezierUniformsBuffer: GPUBuffer;
    private readonly numBeziers: number;
    private readonly brushSampler: GPUSampler;
    private readonly brushTextureView: GPUTextureView;

    constructor({
        device,
        numBeziers,
        bezierBuffer,
        sortOrderBuffer,
        brushTexture,
    }: {
        device: GPUDevice,
        numBeziers: number,
        bezierBuffer: GPUBuffer,
        sortOrderBuffer: GPUBuffer,
        brushTexture: GPUTexture,
    }) {
        this.device = device;
        this.bezierBuffer = bezierBuffer;
        this.sortOrderBuffer = sortOrderBuffer;
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

        // mat4x4 (64) + dims vec2 + pad + cam_world vec4 — match splat forward layout.
        this.bezierUniformsBuffer = device.createBuffer({
            label: "bezier forward uniforms buffer",
            size: 96,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bindGroupLayout = device.createBindGroupLayout({
            label: "bezier forward bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
            ],
        });

        const code = injectWgslConstants(forwardModuleSrc, {
            ...constants,
            NUM_BEZIERS: numBeziers,
        });
        const module = device.createShaderModule({ label: "bezier forward", code });
        module.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[bezier_forward] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
        });

        const pipelineLayout = device.createPipelineLayout({ 
            label: "bezier forward pipeline layout",
            bindGroupLayouts: [this.bindGroupLayout] 
        });

        const blend = {
            color: {
                operation: "add" as const,
                srcFactor: "one" as const,
                dstFactor: "one-minus-src-alpha" as const,
            },
            alpha: {
                operation: "add" as const,
                srcFactor: "one" as const,
                dstFactor: "one-minus-src-alpha" as const,
            },
        };

        this.pipeline1 = device.createRenderPipeline({
            label: "bezier forward render pipeline (1 target)",
            layout: pipelineLayout,
            vertex: { module, entryPoint: "vs_main" },
            fragment: {
                module,
                entryPoint: "fs_main",
                targets: [{ format: "rgba8unorm", blend }],
            },
            primitive: { topology: "triangle-strip" },
        });

        this.pipeline2 = device.createRenderPipeline({
            label: "bezier forward render pipeline (2 targets)",
            layout: pipelineLayout,
            vertex: { module, entryPoint: "vs_main" },
            fragment: {
                module,
                entryPoint: "fs_main",
                targets: [
                    { format: "rgba8unorm", blend },
                    { format: "rgba8unorm", writeMask: 0 },
                ],
            },
            primitive: { topology: "triangle-strip" },
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

    writeCameraWorld(x: number, y: number, z: number) {
        this.device.queue.writeBuffer(this.bezierUniformsBuffer, 80, new Float32Array([x, y, z, 1.0]));
    }

    setTarget(targetView: GPUTextureView, width: number, height: number) {
        this.targetView = targetView;
        if (this.dims.width !== width || this.dims.height !== height) {
            this.dims = { width, height };
            this.device.queue.writeBuffer(this.bezierUniformsBuffer, 64, new Float32Array([width, height, 0, 0]));
        }

        this.bindGroup = this.device.createBindGroup({
            label: "bezier forward bind group",
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bezierBuffer } },
                { binding: 1, resource: { buffer: this.bezierUniformsBuffer } },
                { binding: 2, resource: this.brushSampler },
                { binding: 3, resource: this.brushTextureView },
                { binding: 4, resource: { buffer: this.sortOrderBuffer } },
            ],
        });
    }

    render(pass: GPURenderPassEncoder, isDualTarget: boolean = false) {
        if (this.bindGroup) {
            pass.setPipeline(isDualTarget ? this.pipeline2 : this.pipeline1);
            pass.setBindGroup(0, this.bindGroup);
            pass.draw(4, this.numBeziers);
        }
    }

    addDispatches(commandEncoder: GPUCommandEncoder, clear: boolean = true, timestampWrites?: NonNullable<GPURenderPassDescriptor["timestampWrites"]>) {
        if (!this.bindGroup || !this.targetView) return;
        const pass = commandEncoder.beginRenderPass({
            label: "bezier forward pass",
            ...(timestampWrites ? { timestampWrites } : {}),
            colorAttachments: [
                {
                    view: this.targetView,
                    clearValue: clear ? { r: 0.0, g: 0.0, b: 0.0, a: 0.0 } : undefined,
                    loadOp: clear ? "clear" : "load",
                    storeOp: "store",
                },
            ],
        });
        this.render(pass);
        pass.end();
    }

    destroy() {
        this.bezierUniformsBuffer.destroy();
    }
}
