import type { GpuUniformsBufferManager } from "./GpuUniformsBufferManager";
import type { MeshData } from "./loadGlb";
import meshModuleSrc from "./mesh.wgsl?raw";

// Format the caller must use for the depth-stencil attachment paired with this
// pipeline. Exported so the runner allocates a Z-buffer in the matching format.
export const MESH_DEPTH_FORMAT: GPUTextureFormat = "depth24plus";

// Renders an arbitrary indexed triangle mesh into the color + depth attachments
// using the shared scene uniforms. Vertex layout is interleaved [position(vec3),
// normal(vec3)] -- the same layout we used for the procedural sphere this
// replaced -- and indices are uint32 so concatenated GLB scenes with more than
// 65535 vertices work without special casing.
export class GpuMeshRenderPipelineManager {
    readonly renderPipeline: GPURenderPipeline;
    readonly uniformsManager: GpuUniformsBufferManager;

    readonly vertexBuffer: GPUBuffer;
    readonly indexBuffer: GPUBuffer;
    readonly indexCount: number;

    private static readonly STRIDE = 24; // 6 floats * 4 bytes (pos + normal)
    private readonly device: GPUDevice;

    constructor({
        device,
        format,
        uniformsManager,
        mesh,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        uniformsManager: GpuUniformsBufferManager,
        mesh: MeshData,
    }) {
        this.device = device;
        this.uniformsManager = uniformsManager;

        const module = device.createShaderModule({
            label: "mesh module",
            code: meshModuleSrc,
        });

        const pipelineLayout = device.createPipelineLayout({
            label: "mesh render pipeline layout",
            bindGroupLayouts: [
                device.createBindGroupLayout({
                    entries: [
                        {
                            binding: 0,
                            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                            buffer: { type: "uniform" },
                        },
                        {
                            binding: 1,
                            visibility: GPUShaderStage.FRAGMENT,
                            texture: { sampleType: "float" },
                        },
                        {
                            binding: 2,
                            visibility: GPUShaderStage.FRAGMENT,
                            sampler: { type: "filtering" },
                        },
                    ],
                }),
            ],
        });

        this.renderPipeline = device.createRenderPipeline({
            label: "mesh render pipeline",
            layout: pipelineLayout,

            vertex: {
                module,
                entryPoint: "vert",
                buffers: [
                    {
                        arrayStride: GpuMeshRenderPipelineManager.STRIDE,
                        attributes: [
                            {
                                shaderLocation: 0,
                                offset: 0,
                                format: "float32x3",
                            },
                            {
                                shaderLocation: 1,
                                offset: 12,
                                format: "float32x3",
                            },
                        ],
                    },
                ],
            },

            fragment: {
                module,
                entryPoint: "frag",
                targets: [
                    {
                        format,
                    },
                    {
                        format,
                    },
                ],
            },

            primitive: {
                topology: "triangle-list",
                // glTF 2.0 spec mandates CCW front faces. Cull the inside of
                // closed meshes so back-face fragments don't fight front faces
                // for the visible pixel.
                cullMode: "back",
                frontFace: "ccw",
            },

            depthStencil: {
                format: MESH_DEPTH_FORMAT,
                depthWriteEnabled: true,
                depthCompare: "less",
            },
        });

        // Aligned-size requirement: WebGPU requires buffer sizes to be a
        // multiple of 4. Float32 + Uint32 arrays already satisfy this so a
        // direct upload is fine.
        this.vertexBuffer = device.createBuffer({
            label: "mesh vertex buffer",
            size: mesh.vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.vertexBuffer, 0, mesh.vertices);

        this.indexBuffer = device.createBuffer({
            label: "mesh index buffer",
            size: mesh.indices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.indexBuffer, 0, mesh.indices);
        this.indexCount = mesh.indices.length;
    }

    addDraw(renderPassEncoder: GPURenderPassEncoder, matcapTextureView: GPUTextureView) {
        // We create a transient bind group because the texture might change or just for simplicity.
        // Or we could store it if it's static. Let's create it once if possible.
        const bindGroup = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.uniformsManager.uniformsBuffer },
                },
                {
                    binding: 1,
                    resource: matcapTextureView,
                },
                {
                    binding: 2,
                    resource: this.device.createSampler({
                        magFilter: "linear",
                        minFilter: "linear",
                    }),
                },
            ],
        });

        renderPassEncoder.setPipeline(this.renderPipeline);
        renderPassEncoder.setBindGroup(0, bindGroup);
        renderPassEncoder.setVertexBuffer(0, this.vertexBuffer);
        renderPassEncoder.setIndexBuffer(this.indexBuffer, "uint32");
        renderPassEncoder.drawIndexed(this.indexCount);
    }
}
