import type { ImplicitShape } from "./types.ts";
import { requestGpu } from "$/gpu/setup/requestGpu";

const SHADER = /* wgsl */`
struct Uniforms {
    view_proj: mat4x4f,
    light_dir: vec4f,
};

struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) color: vec4f,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs(
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) color: vec4f,
) -> VertexOut {
    var out: VertexOut;
    out.position = uniforms.view_proj * vec4f(position, 1.0);
    out.normal = normalize(normal);
    out.color = color;
    return out;
}

@fragment
fn fs(in: VertexOut) -> @location(0) vec4f {
    let n = normalize(in.normal);
    let l = normalize(uniforms.light_dir.xyz);
    let diffuse = max(dot(n, l), 0.0) * 0.65 + 0.28;
    let rim = pow(1.0 - max(abs(n.z), 0.0), 2.0) * 0.12;
    return vec4f(in.color.rgb * diffuse + rim, in.color.a);
}
`;

const DEPTH_FORMAT: GPUTextureFormat = "depth24plus";

export class ContourMeshRenderer {
    private readonly device: GPUDevice;
    private readonly context: GPUCanvasContext;
    private readonly format: GPUTextureFormat;
    private readonly pipeline: GPURenderPipeline;
    private readonly uniformBuffer: GPUBuffer;
    private readonly bindGroup: GPUBindGroup;
    private vertexBuffer: GPUBuffer | null = null;
    private indexBuffer: GPUBuffer | null = null;
    private indexCount = 0;
    private depthTexture: GPUTexture | null = null;
    private depthWidth = 0;
    private depthHeight = 0;

    private constructor(
        device: GPUDevice,
        context: GPUCanvasContext,
        format: GPUTextureFormat,
        pipeline: GPURenderPipeline,
        uniformBuffer: GPUBuffer,
        bindGroup: GPUBindGroup,
    ) {
        this.device = device;
        this.context = context;
        this.format = format;
        this.pipeline = pipeline;
        this.uniformBuffer = uniformBuffer;
        this.bindGroup = bindGroup;
    }

    static async create(canvas: HTMLCanvasElement): Promise<ContourMeshRenderer> {
        const gpu = await requestGpu({});
        if (!gpu) throw new Error("WebGPU unavailable");

        const context = canvas.getContext("webgpu");
        if (!context) throw new Error("Could not create WebGPU canvas context");

        context.configure({
            device: gpu.device,
            format: gpu.format,
            alphaMode: "opaque",
        });

        const module = gpu.device.createShaderModule({
            label: "contour mesh renderer",
            code: SHADER,
        });
        module.getCompilationInfo().then(info => {
            for (const message of info.messages) {
                console.warn(`[contour_mesh] ${message.type}: ${message.message} (line ${message.lineNum})`);
            }
        });

        const bindGroupLayout = gpu.device.createBindGroupLayout({
            label: "contour mesh renderer bgl",
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            ],
        });

        const pipeline = gpu.device.createRenderPipeline({
            label: "contour mesh renderer pipeline",
            layout: gpu.device.createPipelineLayout({
                label: "contour mesh renderer layout",
                bindGroupLayouts: [bindGroupLayout],
            }),
            vertex: {
                module,
                entryPoint: "vs",
                buffers: [{
                    arrayStride: 48,
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: "float32x3" },
                        { shaderLocation: 1, offset: 12, format: "float32x3" },
                        { shaderLocation: 2, offset: 24, format: "float32x4" },
                        { shaderLocation: 3, offset: 40, format: "float32x2" },
                    ],
                }],
            },
            fragment: {
                module,
                entryPoint: "fs",
                targets: [{ format: gpu.format }],
            },
            primitive: {
                topology: "triangle-list",
                cullMode: "none",
            },
            depthStencil: {
                format: DEPTH_FORMAT,
                depthCompare: "less",
                depthWriteEnabled: true,
            },
        });

        const uniformBuffer = gpu.device.createBuffer({
            label: "contour mesh renderer uniforms",
            size: 80,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const bindGroup = gpu.device.createBindGroup({
            label: "contour mesh renderer bind group",
            layout: bindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
        });

        return new ContourMeshRenderer(gpu.device, context, gpu.format, pipeline, uniformBuffer, bindGroup);
    }

    setShapes(shapes: ImplicitShape[], activeShapeId: string | null) {
        const vertices: number[] = [];
        const indices: number[] = [];
        let vertexOffset = 0;

        for (const shape of shapes) {
            if (!shape.mesh) continue;
            const meshVertices = new Float32Array(shape.mesh.vertices);
            const isActive = shape.id === activeShapeId;

            for (let i = 0; i < meshVertices.length / 12; i++) {
                const base = i * 12;
                if (isActive) {
                    meshVertices[base + 6] = Math.min(1, meshVertices[base + 6] * 1.1 + 0.08);
                    meshVertices[base + 7] = Math.min(1, meshVertices[base + 7] * 1.05 + 0.05);
                    meshVertices[base + 8] = Math.min(1, meshVertices[base + 8] * 0.95 + 0.02);
                } else {
                    meshVertices[base + 6] *= 0.58;
                    meshVertices[base + 7] *= 0.58;
                    meshVertices[base + 8] *= 0.58;
                }
            }

            for (const value of meshVertices) {
                vertices.push(value);
            }
            for (const index of shape.mesh.indices) {
                indices.push(index + vertexOffset);
            }
            vertexOffset += shape.mesh.vertices.length / 12;
        }

        this.vertexBuffer?.destroy();
        this.indexBuffer?.destroy();
        this.indexCount = indices.length;

        if (vertices.length === 0 || indices.length === 0) {
            this.vertexBuffer = null;
            this.indexBuffer = null;
            return;
        }

        const vertexData = new Float32Array(vertices);
        const indexData = new Uint32Array(indices);
        this.vertexBuffer = this.device.createBuffer({
            label: "contour mesh vertices",
            size: vertexData.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.vertexBuffer, 0, vertexData);

        this.indexBuffer = this.device.createBuffer({
            label: "contour mesh indices",
            size: indexData.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.indexBuffer, 0, indexData);
    }

    render(viewProjMat: Float32Array | number[]) {
        const width = Math.floor(this.context.canvas.width);
        const height = Math.floor(this.context.canvas.height);
        if (width <= 0 || height <= 0) return;

        this.ensureDepthTexture(width, height);
        const uniformData = new Float32Array(20);
        uniformData.set(viewProjMat as Float32Array, 0);
        uniformData.set([0.35, 0.8, 0.45, 0], 16);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

        const encoder = this.device.createCommandEncoder({ label: "contour mesh render encoder" });
        const pass = encoder.beginRenderPass({
            label: "contour mesh render pass",
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0.035, g: 0.038, b: 0.042, a: 1 },
                loadOp: "clear",
                storeOp: "store",
            }],
            depthStencilAttachment: {
                view: this.depthTexture!.createView(),
                depthClearValue: 1,
                depthLoadOp: "clear",
                depthStoreOp: "store",
            },
        });

        if (this.vertexBuffer && this.indexBuffer && this.indexCount > 0) {
            pass.setPipeline(this.pipeline);
            pass.setBindGroup(0, this.bindGroup);
            pass.setVertexBuffer(0, this.vertexBuffer);
            pass.setIndexBuffer(this.indexBuffer, "uint32");
            pass.drawIndexed(this.indexCount);
        }

        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }

    destroy() {
        this.vertexBuffer?.destroy();
        this.indexBuffer?.destroy();
        this.depthTexture?.destroy();
        this.uniformBuffer.destroy();
        this.device.destroy();
    }

    private ensureDepthTexture(width: number, height: number) {
        if (this.depthTexture && this.depthWidth === width && this.depthHeight === height) return;
        this.depthTexture?.destroy();
        this.depthTexture = this.device.createTexture({
            label: "contour mesh depth",
            size: [width, height],
            format: DEPTH_FORMAT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.depthWidth = width;
        this.depthHeight = height;
    }
}
