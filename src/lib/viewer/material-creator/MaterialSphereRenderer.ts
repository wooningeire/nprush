import { mat4, type Mat4 } from "wgpu-matrix";
import shaderSource from "./sphere.wgsl?raw";

/**
 * Renders an orthographic unit sphere using a ray-sphere intersection test.
 * Manages its own WebGPU pipeline and uniforms; call `render()` each frame.
 */
export class MaterialSphereRenderer {
    private device: GPUDevice;
    private context: GPUCanvasContext;
    private pipeline: GPURenderPipeline;
    private uniformBuffer: GPUBuffer;
    private bindGroup: GPUBindGroup;

    private constructor(
        device: GPUDevice,
        context: GPUCanvasContext,
        pipeline: GPURenderPipeline,
        uniformBuffer: GPUBuffer,
        bindGroup: GPUBindGroup,
    ) {
        this.device = device;
        this.context = context;
        this.pipeline = pipeline;
        this.uniformBuffer = uniformBuffer;
        this.bindGroup = bindGroup;
    }

    static async create(canvas: HTMLCanvasElement): Promise<MaterialSphereRenderer> {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("WebGPU adapter unavailable");

        const device = await adapter.requestDevice();

        const context = canvas.getContext("webgpu")!;
        const format = navigator.gpu.getPreferredCanvasFormat();
        context.configure({
            device,
            format,
            alphaMode: "opaque",
        });

        const shaderModule = device.createShaderModule({ code: shaderSource });

        const uniformBuffer = device.createBuffer({
            // 2 mat4x4f (128 bytes) + vec2f res + vec2f pad = 144, align to 256
            size: 256,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: { type: "uniform" },
            }],
        });

        const pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout],
        });

        const pipeline = device.createRenderPipeline({
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: "vs",
            },
            fragment: {
                module: shaderModule,
                entryPoint: "fs",
                targets: [{ format }],
            },
            primitive: {
                topology: "triangle-list",
            },
        });

        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [{
                binding: 0,
                resource: { buffer: uniformBuffer },
            }],
        });

        return new MaterialSphereRenderer(device, context, pipeline, uniformBuffer, bindGroup);
    }

    render(viewMat: Mat4, viewInvMat: Mat4) {
        const width = this.context.canvas.width;
        const height = this.context.canvas.height;

        if (width === 0 || height === 0) return;

        // Write uniforms
        const data = new Float32Array(256 / 4);
        data.set(viewMat as Float32Array, 0);          // offset 0:  viewMat (16 floats)
        data.set(viewInvMat as Float32Array, 16);       // offset 64: viewInvMat (16 floats)
        data[32] = width;                                // offset 128: resolution.x
        data[33] = height;                               // offset 132: resolution.y
        // data[34..35] = pad

        this.device.queue.writeBuffer(this.uniformBuffer, 0, data);

        const texture = this.context.getCurrentTexture();
        const encoder = this.device.createCommandEncoder();

        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: texture.createView(),
                loadOp: "clear",
                storeOp: "store",
                clearValue: { r: 0.03, g: 0.03, b: 0.04, a: 1.0 },
            }],
        });

        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.draw(3); // Full-screen triangle
        pass.end();

        this.device.queue.submit([encoder.finish()]);
    }

    destroy() {
        this.uniformBuffer.destroy();
        this.device.destroy();
    }
}
