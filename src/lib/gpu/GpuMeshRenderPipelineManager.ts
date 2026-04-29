import type { GpuUniformsBufferManager } from "./GpuUniformsBufferManager";
import type { MeshData } from "./loadGlb";
import meshModuleSrc from "./mesh.wgsl?raw";

// Format the caller must use for the depth-stencil attachment paired with this
// pipeline. Exported so the runner allocates a Z-buffer in the matching format.
export const MESH_DEPTH_FORMAT: GPUTextureFormat = "depth24plus";

// Renders an arbitrary indexed triangle mesh into the color + depth attachments
// using the shared scene uniforms.
//
// Vertex layout is interleaved [position(vec3), normal(vec3), color(vec4), uv(vec2)]
// — stride 12 floats / 48 bytes. Indices are uint32.
//
// The ground mesh (Plane.001) uses a separate PBR pipeline that samples an
// albedo texture and a tangent-space normal map.
export class GpuMeshRenderPipelineManager {
    readonly renderPipeline: GPURenderPipeline;
    readonly uniformsManager: GpuUniformsBufferManager;

    readonly vertexBuffer: GPUBuffer;
    readonly indexBuffer: GPUBuffer;
    readonly indexCount: number;

    private groundVertexBuffer: GPUBuffer | null = null;
    private groundIndexBuffer: GPUBuffer | null = null;
    private groundIndexCount: number = 0;

    private pbrVertexBuffer: GPUBuffer | null = null;
    private pbrIndexBuffer: GPUBuffer | null = null;
    private pbrIndexCount: number = 0;
    private pbrBindGroup: GPUBindGroup | null = null;
    private pbrPipeline: GPURenderPipeline | null = null;

    // Stride: pos(3) + normal(3) + color(4) + uv(2) = 12 floats × 4 bytes
    private static readonly STRIDE = 48;
    private readonly device: GPUDevice;

    // Store format for use in setGroundMesh (set during construction)
    private _format!: GPUTextureFormat;

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
        this._format = format;

        const module = device.createShaderModule({
            label: "mesh module",
            code: meshModuleSrc,
        });

        const vertexBufferLayout: GPUVertexBufferLayout = {
            arrayStride: GpuMeshRenderPipelineManager.STRIDE,
            attributes: [
                { shaderLocation: 0, offset: 0,  format: "float32x3" }, // position
                { shaderLocation: 1, offset: 12, format: "float32x3" }, // normal
                { shaderLocation: 2, offset: 24, format: "float32x4" }, // color
                { shaderLocation: 3, offset: 40, format: "float32x2" }, // uv
            ],
        };

        // Bind group layout for the standard (matcap) pipeline
        const standardBgl = device.createBindGroupLayout({
            label: "mesh standard bgl",
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
            ],
        });

        const pipelineLayout = device.createPipelineLayout({
            label: "mesh render pipeline layout",
            bindGroupLayouts: [standardBgl],
        });

        const fragmentTargets: GPUColorTargetState[] = [{ format }, { format }];

        const primitiveState: GPUPrimitiveState = {
            topology: "triangle-list",
            cullMode: "back",
            frontFace: "ccw",
        };

        const depthStencil: GPUDepthStencilState = {
            format: MESH_DEPTH_FORMAT,
            depthWriteEnabled: true,
            depthCompare: "less",
        };

        this.renderPipeline = device.createRenderPipeline({
            label: "mesh render pipeline",
            layout: pipelineLayout,
            vertex: { module, entryPoint: "vert", buffers: [vertexBufferLayout] },
            fragment: { module, entryPoint: "frag", targets: fragmentTargets },
            primitive: primitiveState,
            depthStencil,
        });

        this.vertexBuffer = device.createBuffer({
            label: "mesh vertex buffer",
            size: mesh.vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.vertexBuffer, 0, mesh.vertices as any);

        this.indexBuffer = device.createBuffer({
            label: "mesh index buffer",
            size: mesh.indices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.indexBuffer, 0, mesh.indices as any);
        this.indexCount = mesh.indices.length;
    }

    /** Upload a plain ground mesh rendered with the standard matcap pipeline. */
    setGroundMesh(mesh: MeshData) {
        if (this.groundVertexBuffer) this.groundVertexBuffer.destroy();
        if (this.groundIndexBuffer) this.groundIndexBuffer.destroy();

        this.groundVertexBuffer = this.device.createBuffer({
            label: "ground vertex buffer",
            size: mesh.vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.groundVertexBuffer, 0, mesh.vertices as any);

        this.groundIndexBuffer = this.device.createBuffer({
            label: "ground index buffer",
            size: mesh.indices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.groundIndexBuffer, 0, mesh.indices as any);
        this.groundIndexCount = mesh.indices.length;
    }

    /**
     * Upload a PBR mesh (Plane.001) with albedo + normal map textures.
     * albedoTexture: sRGB diffuse map
     * normalTexture: OpenGL-convention tangent-space normal map
     * matcapTexture: shared env/matcap texture (for lighting)
     */
    setPbrMesh(
        mesh: MeshData,
        albedoTexture: GPUTexture,
        normalTexture: GPUTexture,
        matcapTexture: GPUTexture,
    ) {
        if (this.pbrVertexBuffer) this.pbrVertexBuffer.destroy();
        if (this.pbrIndexBuffer) this.pbrIndexBuffer.destroy();

        this.pbrVertexBuffer = this.device.createBuffer({
            label: "pbr vertex buffer",
            size: mesh.vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.pbrVertexBuffer, 0, mesh.vertices as any);

        this.pbrIndexBuffer = this.device.createBuffer({
            label: "pbr index buffer",
            size: mesh.indices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.pbrIndexBuffer, 0, mesh.indices as any);
        this.pbrIndexCount = mesh.indices.length;

        const module = this.device.createShaderModule({
            label: "mesh pbr module",
            code: meshModuleSrc,
        });

        const pbrBgl = this.device.createBindGroupLayout({
            label: "mesh pbr bgl",
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
            ],
        });

        const pbrLayout = this.device.createPipelineLayout({
            label: "mesh pbr pipeline layout",
            bindGroupLayouts: [pbrBgl],
        });

        this.pbrPipeline = this.device.createRenderPipeline({
            label: "ground pbr pipeline",
            layout: pbrLayout,
            vertex: {
                module,
                entryPoint: "vert_pbr",
                buffers: [{
                    arrayStride: GpuMeshRenderPipelineManager.STRIDE,
                    attributes: [
                        { shaderLocation: 0, offset: 0,  format: "float32x3" },
                        { shaderLocation: 1, offset: 12, format: "float32x3" },
                        { shaderLocation: 2, offset: 24, format: "float32x4" },
                        { shaderLocation: 3, offset: 40, format: "float32x2" },
                    ],
                }],
            },
            fragment: {
                module,
                entryPoint: "frag_pbr",
                targets: [{ format: this._format }, { format: this._format }],
            },
            primitive: { topology: "triangle-list", cullMode: "back", frontFace: "ccw" },
            depthStencil: { format: MESH_DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: "less" },
        });

        const pbrSampler = this.device.createSampler({
            magFilter: "linear",
            minFilter: "linear",
            mipmapFilter: "linear",
            addressModeU: "repeat",
            addressModeV: "repeat",
        });

        this.pbrBindGroup = this.device.createBindGroup({
            label: "ground pbr bind group",
            layout: pbrBgl,
            entries: [
                { binding: 0, resource: { buffer: this.uniformsManager.uniformsBuffer } },
                { binding: 1, resource: matcapTexture.createView() },
                { binding: 2, resource: this.device.createSampler({ magFilter: "linear", minFilter: "linear" }) },
                { binding: 3, resource: albedoTexture.createView() },
                { binding: 4, resource: normalTexture.createView() },
                { binding: 5, resource: pbrSampler },
            ],
        });
    }

    addDraw(renderPassEncoder: GPURenderPassEncoder, matcapTextureView: GPUTextureView) {
        const bindGroup = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.uniformsManager.uniformsBuffer } },
                { binding: 1, resource: matcapTextureView },
                { binding: 2, resource: this.device.createSampler({ magFilter: "linear", minFilter: "linear" }) },
            ],
        });

        renderPassEncoder.setPipeline(this.renderPipeline);
        renderPassEncoder.setBindGroup(0, bindGroup);
        renderPassEncoder.setVertexBuffer(0, this.vertexBuffer);
        renderPassEncoder.setIndexBuffer(this.indexBuffer, "uint32");
        renderPassEncoder.drawIndexed(this.indexCount);

        // Draw plain ground mesh with standard matcap pipeline
        if (this.groundVertexBuffer && this.groundIndexBuffer && this.groundIndexCount > 0) {
            renderPassEncoder.setVertexBuffer(0, this.groundVertexBuffer);
            renderPassEncoder.setIndexBuffer(this.groundIndexBuffer, "uint32");
            renderPassEncoder.drawIndexed(this.groundIndexCount);
        }

        // Draw PBR mesh (Plane.001) with PBR pipeline
        if (this.pbrPipeline && this.pbrBindGroup && this.pbrVertexBuffer && this.pbrIndexBuffer && this.pbrIndexCount > 0) {
            renderPassEncoder.setPipeline(this.pbrPipeline);
            renderPassEncoder.setBindGroup(0, this.pbrBindGroup);
            renderPassEncoder.setVertexBuffer(0, this.pbrVertexBuffer);
            renderPassEncoder.setIndexBuffer(this.pbrIndexBuffer, "uint32");
            renderPassEncoder.drawIndexed(this.pbrIndexCount);
        }
    }
}
