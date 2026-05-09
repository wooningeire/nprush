import forwardModuleSrc from "./splat_forward.wgsl?raw";
import type { Mat4 } from "wgpu-matrix";
import { constants, injectWgslConstants } from "../constants";

export class GpuSplatForwardPipelineManager {
    private readonly device: GPUDevice;
    private readonly pipeline: GPURenderPipeline;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private bindGroup: GPUBindGroup | null = null;
    private dims: { width: number, height: number } = { width: 0, height: 0 };
    private readonly splatBuffer: GPUBuffer;
    private readonly sortOrderBuffer: GPUBuffer;
    private readonly uniformsBuffer: GPUBuffer;
    private readonly numSplats: number;
    private targetColorView: GPUTextureView | null = null;
    private targetDepthView: GPUTextureView | null = null;

    constructor({
        device,
        numSplats,
        splatBuffer,
        sortOrderBuffer,
    }: {
        device: GPUDevice,
        numSplats: number,
        splatBuffer: GPUBuffer,
        sortOrderBuffer: GPUBuffer,
    }) {
        this.device = device;
        this.splatBuffer = splatBuffer;
        this.sortOrderBuffer = sortOrderBuffer;
        this.numSplats = numSplats;

        // mat4x4 (64) + dims vec2 + pad (8) + cam_world vec4 (16) = 96
        this.uniformsBuffer = device.createBuffer({
            label: "splat forward uniforms buffer",
            size: 96,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bindGroupLayout = device.createBindGroupLayout({
            label: "splat forward bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
            ],
        });

        const code = injectWgslConstants(forwardModuleSrc, {
            ...constants,
            NUM_SPLATS: numSplats,
        });
        const module = device.createShaderModule({ label: "splat forward render", code });
        module.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[splat_forward] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
        });

        this.pipeline = device.createRenderPipeline({
            label: "splat forward render pipeline",
            layout: device.createPipelineLayout({ 
                label: "splat forward pipeline layout",
                bindGroupLayouts: [this.bindGroupLayout] 
            }),
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
            label: "splat forward bind group",
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.splatBuffer } },
                { binding: 1, resource: { buffer: this.uniformsBuffer } },
                { binding: 2, resource: { buffer: this.sortOrderBuffer } },
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

    writeCameraWorld(x: number, y: number, z: number) {
        this.device.queue.writeBuffer(this.uniformsBuffer, 80, new Float32Array([x, y, z, 1.0]));
    }

    setTarget(targetColorView: GPUTextureView, targetDepthView: GPUTextureView, width: number, height: number) {
        this.targetColorView = targetColorView;
        this.targetDepthView = targetDepthView;
        if (this.dims.width !== width || this.dims.height !== height) {
            this.dims = { width, height };
            this.device.queue.writeBuffer(this.uniformsBuffer, 64, new Float32Array([width, height, 0, 0]));
        }
    }

    dispatch(commandEncoder: GPUCommandEncoder, clear: boolean = false, draw: boolean = true) {
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
        if (draw) {
            pass.setPipeline(this.pipeline);
            pass.setBindGroup(0, this.bindGroup);
            pass.draw(6, this.numSplats);
        }
        pass.end();
    }

    destroy() {
        this.uniformsBuffer.destroy();
    }
}
