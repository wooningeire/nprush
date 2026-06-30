import { requestGpu } from "$/gpu/setup/requestGpu";
import gridShaderSource from "./paint_modeler_grid.wgsl?raw";
import segmentShaderSource from "./paint_modeler_segments.wgsl?raw";
import triangleShaderSource from "./paint_modeler_triangles.wgsl?raw";
import type { RenderPrimitive } from "./types.ts";
import {
    DEPTH_FORMAT,
    GRID_PLANE_Z,
    GRID_UNIFORM_FLOATS,
    MATRIX_FLOATS,
    SEGMENT_UNIFORM_FLOATS,
    TRIANGLE_UNIFORM_FLOATS,
    VERTICES_PER_SEGMENT,
    VERTICES_PER_TRIANGLE,
} from "./renderer/constants.ts";
import {
    createUniformBindGroup,
    createUniformBuffer,
    createVertexBufferState,
    destroyVertexBuffer,
    uploadVertexData,
    type VertexBufferState,
} from "./renderer/buffers.ts";
import {
    createGridPipeline,
    createSegmentPipeline,
    createTrianglePipeline,
} from "./renderer/pipelines.ts";
import {
    createSegmentData,
    createStrokeData,
    createTriangleData,
    isRenderSegment,
    isRenderStroke,
    isRenderTriangle,
    strokeStripVertexCount,
} from "./renderer/vertices.ts";

export class PaintModelingRenderer {
    private readonly device: GPUDevice;
    private readonly context: GPUCanvasContext;
    private readonly gridPipeline: GPURenderPipeline;
    private readonly segmentPipeline: GPURenderPipeline;
    private readonly strokePipeline: GPURenderPipeline;
    private readonly trianglePipeline: GPURenderPipeline;
    private readonly gridUniformBuffer: GPUBuffer;
    private readonly segmentUniformBuffer: GPUBuffer;
    private readonly triangleUniformBuffer: GPUBuffer;
    private readonly gridBindGroup: GPUBindGroup;
    private readonly segmentBindGroup: GPUBindGroup;
    private readonly triangleBindGroup: GPUBindGroup;
    private readonly segmentBuffer: VertexBufferState = createVertexBufferState();
    private readonly strokeBuffer: VertexBufferState = createVertexBufferState();
    private readonly draftSegmentBuffer: VertexBufferState = createVertexBufferState();
    private readonly draftStrokeBuffer: VertexBufferState = createVertexBufferState();
    private readonly triangleBuffer: VertexBufferState = createVertexBufferState();
    private readonly draftTriangleBuffer: VertexBufferState = createVertexBufferState();
    private depthTexture: GPUTexture | null = null;
    private depthWidth = 0;
    private depthHeight = 0;
    private segmentVertexCount = 0;
    private strokeVertexCount = 0;
    private draftSegmentVertexCount = 0;
    private draftStrokeVertexCount = 0;
    private triangleVertexCount = 0;
    private draftTriangleVertexCount = 0;

    static async create(canvas: HTMLCanvasElement): Promise<PaintModelingRenderer> {
        const gpu = await requestGpu({});
        if (!gpu) throw new Error("WebGPU unavailable");

        const context = canvas.getContext("webgpu");
        if (!context) throw new Error("Could not create WebGPU canvas context");

        context.configure({
            device: gpu.device,
            format: gpu.format,
            alphaMode: "opaque",
        });

        return new PaintModelingRenderer(gpu.device, context, gpu.format);
    }

    private constructor(
        device: GPUDevice,
        context: GPUCanvasContext,
        format: GPUTextureFormat,
    ) {
        this.device = device;
        this.context = context;

        const gridModule = createLoggedShaderModule(device, "paint modeler grid shader", gridShaderSource);
        const segmentModule = createLoggedShaderModule(device, "paint modeler segment shader", segmentShaderSource);
        const triangleModule = createLoggedShaderModule(device, "paint modeler triangle shader", triangleShaderSource);

        const uniformBindGroupLayout = device.createBindGroupLayout({
            label: "paint modeler uniform bind group layout",
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: { type: "uniform" },
            }],
        });
        const pipelineLayout = device.createPipelineLayout({
            label: "paint modeler pipeline layout",
            bindGroupLayouts: [uniformBindGroupLayout],
        });

        this.gridPipeline = createGridPipeline(device, pipelineLayout, gridModule, format);
        this.segmentPipeline = createSegmentPipeline(device, pipelineLayout, segmentModule, format, "triangle-list");
        this.strokePipeline = createSegmentPipeline(device, pipelineLayout, segmentModule, format, "triangle-strip");
        this.trianglePipeline = createTrianglePipeline(device, pipelineLayout, triangleModule, format, true);

        this.gridUniformBuffer = createUniformBuffer(device, GRID_UNIFORM_FLOATS, "paint modeler grid uniforms");
        this.segmentUniformBuffer = createUniformBuffer(device, SEGMENT_UNIFORM_FLOATS, "paint modeler segment uniforms");
        this.triangleUniformBuffer = createUniformBuffer(device, TRIANGLE_UNIFORM_FLOATS, "paint modeler triangle uniforms");
        this.gridBindGroup = createUniformBindGroup(
            device,
            uniformBindGroupLayout,
            this.gridUniformBuffer,
            "paint modeler grid bind group",
        );
        this.segmentBindGroup = createUniformBindGroup(
            device,
            uniformBindGroupLayout,
            this.segmentUniformBuffer,
            "paint modeler segment bind group",
        );
        this.triangleBindGroup = createUniformBindGroup(
            device,
            uniformBindGroupLayout,
            this.triangleUniformBuffer,
            "paint modeler triangle bind group",
        );
    }

    setSegments(segments: RenderPrimitive[]) {
        const renderSegments = segments.filter(isRenderSegment);
        const strokes = segments.filter(isRenderStroke);
        const triangles = segments.filter(isRenderTriangle);
        const segmentVertexCount = renderSegments.length * VERTICES_PER_SEGMENT;
        const strokeVertexCount = strokeStripVertexCount(strokes);
        const triangleVertexCount = triangles.length * VERTICES_PER_TRIANGLE;
        const segmentData = createSegmentData(renderSegments);
        const strokeData = createStrokeData(strokes);
        const triangleData = createTriangleData(triangles);

        this.segmentVertexCount = segmentVertexCount;
        this.strokeVertexCount = strokeVertexCount;
        this.triangleVertexCount = triangleVertexCount;

        uploadVertexData(this.device, this.segmentBuffer, segmentData, segmentVertexCount, "paint modeler segments");
        uploadVertexData(this.device, this.strokeBuffer, strokeData, strokeVertexCount, "paint modeler strokes");
        uploadVertexData(this.device, this.triangleBuffer, triangleData, triangleVertexCount, "paint modeler triangles");
    }

    setDraftSegments(segments: RenderPrimitive[]) {
        const renderSegments = segments.filter(isRenderSegment);
        const strokes = segments.filter(isRenderStroke);
        const triangles = segments.filter(isRenderTriangle);
        const segmentVertexCount = renderSegments.length * VERTICES_PER_SEGMENT;
        const strokeVertexCount = strokeStripVertexCount(strokes);
        const triangleVertexCount = triangles.length * VERTICES_PER_TRIANGLE;
        const segmentData = createSegmentData(renderSegments);
        const strokeData = createStrokeData(strokes);
        const triangleData = createTriangleData(triangles);

        this.draftSegmentVertexCount = segmentVertexCount;
        this.draftStrokeVertexCount = strokeVertexCount;
        this.draftTriangleVertexCount = triangleVertexCount;
        uploadVertexData(
            this.device,
            this.draftSegmentBuffer,
            segmentData,
            segmentVertexCount,
            "paint modeler draft segments",
        );
        uploadVertexData(
            this.device,
            this.draftStrokeBuffer,
            strokeData,
            strokeVertexCount,
            "paint modeler draft strokes",
        );
        uploadVertexData(
            this.device,
            this.draftTriangleBuffer,
            triangleData,
            triangleVertexCount,
            "paint modeler draft triangles",
        );
    }

    render(
        viewProjMat: number[] | Float32Array,
        viewProjInvMat: number[] | Float32Array,
        viewMat: number[] | Float32Array,
    ) {
        const width = Math.floor(this.context.canvas.width);
        const height = Math.floor(this.context.canvas.height);
        if (width <= 0 || height <= 0) return;

        this.ensureDepthTexture(width, height);
        this.writeUniforms(viewProjMat, viewProjInvMat, viewMat);

        const encoder = this.device.createCommandEncoder({ label: "paint modeler render encoder" });
        const pass = encoder.beginRenderPass({
            label: "paint modeler render pass",
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0.035, g: 0.043, b: 0.047, a: 1 },
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

        pass.setPipeline(this.gridPipeline);
        pass.setBindGroup(0, this.gridBindGroup);
        pass.draw(3);

        this.drawTriangleBuffer(pass, this.triangleBuffer, this.triangleVertexCount);
        this.drawSegmentBuffer(pass, this.segmentBuffer, this.segmentVertexCount, this.segmentPipeline);
        this.drawSegmentBuffer(pass, this.strokeBuffer, this.strokeVertexCount, this.strokePipeline);
        this.drawTriangleBuffer(pass, this.draftTriangleBuffer, this.draftTriangleVertexCount);
        this.drawSegmentBuffer(pass, this.draftSegmentBuffer, this.draftSegmentVertexCount, this.segmentPipeline);
        this.drawSegmentBuffer(pass, this.draftStrokeBuffer, this.draftStrokeVertexCount, this.strokePipeline);

        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }

    destroy() {
        destroyVertexBuffer(this.segmentBuffer);
        destroyVertexBuffer(this.strokeBuffer);
        destroyVertexBuffer(this.draftSegmentBuffer);
        destroyVertexBuffer(this.draftStrokeBuffer);
        destroyVertexBuffer(this.triangleBuffer);
        destroyVertexBuffer(this.draftTriangleBuffer);
        this.depthTexture?.destroy();
        this.gridUniformBuffer.destroy();
        this.segmentUniformBuffer.destroy();
        this.triangleUniformBuffer.destroy();
        this.device.destroy();
    }

    private writeUniforms(
        viewProjMat: number[] | Float32Array,
        viewProjInvMat: number[] | Float32Array,
        viewMat: number[] | Float32Array,
    ) {
        const canvas = this.context.canvas as HTMLCanvasElement;
        const gridUniformData = new Float32Array(GRID_UNIFORM_FLOATS);
        gridUniformData.set(viewProjInvMat, 0);
        gridUniformData[MATRIX_FLOATS] = GRID_PLANE_Z;
        this.device.queue.writeBuffer(this.gridUniformBuffer, 0, gridUniformData);

        const segmentUniformData = new Float32Array(SEGMENT_UNIFORM_FLOATS);
        segmentUniformData.set(viewProjMat, 0);
        segmentUniformData[MATRIX_FLOATS] = Math.max(
            1,
            canvas.clientWidth || canvas.width,
        );
        segmentUniformData[MATRIX_FLOATS + 1] = Math.max(
            1,
            canvas.clientHeight || canvas.height,
        );
        this.device.queue.writeBuffer(this.segmentUniformBuffer, 0, segmentUniformData);

        const triangleUniformData = new Float32Array(TRIANGLE_UNIFORM_FLOATS);
        triangleUniformData.set(viewProjMat, 0);
        triangleUniformData.set(viewMat, MATRIX_FLOATS);
        this.device.queue.writeBuffer(this.triangleUniformBuffer, 0, triangleUniformData);
    }

    private drawSegmentBuffer(
        pass: GPURenderPassEncoder,
        buffer: VertexBufferState,
        vertexCount: number,
        pipeline: GPURenderPipeline,
    ) {
        if (vertexCount === 0 || !buffer.buffer) return;
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.segmentBindGroup);
        pass.setVertexBuffer(0, buffer.buffer);
        pass.draw(vertexCount);
    }

    private drawTriangleBuffer(
        pass: GPURenderPassEncoder,
        buffer: VertexBufferState,
        vertexCount: number,
    ) {
        if (vertexCount === 0 || !buffer.buffer) return;
        pass.setPipeline(this.trianglePipeline);
        pass.setBindGroup(0, this.triangleBindGroup);
        pass.setVertexBuffer(0, buffer.buffer);
        pass.draw(vertexCount);
    }

    private ensureDepthTexture(width: number, height: number) {
        if (this.depthTexture && this.depthWidth === width && this.depthHeight === height) return;
        this.depthTexture?.destroy();
        this.depthTexture = this.device.createTexture({
            label: "paint modeler depth",
            size: [width, height],
            format: DEPTH_FORMAT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.depthWidth = width;
        this.depthHeight = height;
    }
}

const createLoggedShaderModule = (
    device: GPUDevice,
    label: string,
    code: string,
): GPUShaderModule => {
    const module = device.createShaderModule({ label, code });
    void module.getCompilationInfo().then(info => {
        for (const message of info.messages) {
            console.warn(`[${label}] ${message.type}: ${message.message} (line ${message.lineNum})`);
        }
    });
    return module;
};