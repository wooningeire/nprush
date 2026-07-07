import brushGuideShaderSource from "../paint_brush_guide.wgsl?raw";
import brushPlacementShaderSource from "../paint_brush_placement.wgsl?raw";
import type { PaintRibbon, PaintRibbonVertex, PaintView, RenderRibbon, Vec2, Vec3 } from "../types.ts";
import {
    DEPTH_FORMAT,
    FLOATS_PER_RIBBON_VERTEX,
} from "./constants.ts";

const PLACEMENT_UNIFORM_FLOATS = 56;
const PLACEMENT_RESULT_FLOATS = 16;
const SOURCE_POINT_FLOATS = 4;
const TARGET_INFO_UINTS = 4;
const GUIDE_VERTEX_COUNT = 48 * 2 + 2;
const WORKGROUP_SIZE = 64;

type BrushPlacementInput = {
    point: Vec2,
    brushWidth: number,
    viewportWidth: number,
    viewportHeight: number,
    snapToRibbons: boolean,
};

type BrushPlacementReadback = {
    center: Vec3,
    normal: Vec3,
    depth: number,
    snapped: boolean,
};

type StrokePlacementInput = {
    sourcePoints: Vec2[],
    view: PaintView,
    brushWidth: number,
    snapToRibbons: boolean,
};

export class BrushPlacementManager {
    private readonly device: GPUDevice;
    private readonly computeBindGroupLayout: GPUBindGroupLayout;
    private readonly guideBindGroupLayout: GPUBindGroupLayout;
    private readonly computePipeline: GPUComputePipeline;
    private readonly strokeComputePipeline: GPUComputePipeline;
    private readonly guidePipeline: GPURenderPipeline;
    private readonly hoverUniformBuffer: GPUBuffer;
    private readonly hoverResultBuffer: GPUBuffer;
    private readonly dummyReadBuffer: GPUBuffer;
    private readonly dummyResultBuffer: GPUBuffer;
    private readonly dummyOutputBuffer: GPUBuffer;
    private readonly dummyMetaBuffer: GPUBuffer;
    private targetVertexBuffer: GPUBuffer | null = null;
    private targetInfoBuffer: GPUBuffer | null = null;
    private targetCount = 0;
    private hoverInput: BrushPlacementInput | null = null;
    private hoverComputeBindGroup: GPUBindGroup | null = null;
    private guideBindGroup: GPUBindGroup | null = null;

    constructor(device: GPUDevice, format: GPUTextureFormat) {
        this.device = device;
        this.computeBindGroupLayout = device.createBindGroupLayout({
            label: "paint brush placement bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" },
                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" },
                },
                {
                    binding: 6,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" },
                },
            ],
        });
        this.guideBindGroupLayout = device.createBindGroupLayout({
            label: "paint brush guide bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "uniform" },
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "read-only-storage" },
                },
            ],
        });

        const computePipelineLayout = device.createPipelineLayout({
            label: "paint brush placement pipeline layout",
            bindGroupLayouts: [this.computeBindGroupLayout],
        });
        const guidePipelineLayout = device.createPipelineLayout({
            label: "paint brush guide pipeline layout",
            bindGroupLayouts: [this.guideBindGroupLayout],
        });
        const placementModule = createLoggedShaderModule(
            device,
            "paint brush placement shader",
            brushPlacementShaderSource,
        );
        const guideModule = createLoggedShaderModule(
            device,
            "paint brush guide shader",
            brushGuideShaderSource,
        );

        this.computePipeline = device.createComputePipeline({
            label: "paint brush hover placement pipeline",
            layout: computePipelineLayout,
            compute: {
                module: placementModule,
                entryPoint: "compute_hover",
            },
        });
        this.strokeComputePipeline = device.createComputePipeline({
            label: "paint brush stroke placement pipeline",
            layout: computePipelineLayout,
            compute: {
                module: placementModule,
                entryPoint: "compute_stroke",
            },
        });
        this.guidePipeline = device.createRenderPipeline({
            label: "paint brush guide pipeline",
            layout: guidePipelineLayout,
            vertex: {
                module: guideModule,
                entryPoint: "guide_vertex",
            },
            fragment: {
                module: guideModule,
                entryPoint: "guide_fragment",
                targets: [{
                    format,
                    blend: {
                        color: {
                            operation: "add",
                            srcFactor: "src-alpha",
                            dstFactor: "one-minus-src-alpha",
                        },
                        alpha: {
                            operation: "add",
                            srcFactor: "src-alpha",
                            dstFactor: "one-minus-src-alpha",
                        },
                    },
                }],
            },
            primitive: {
                topology: "line-list",
                cullMode: "none",
            },
            depthStencil: {
                format: DEPTH_FORMAT,
                depthCompare: "less-equal",
                depthWriteEnabled: false,
            },
        });

        this.hoverUniformBuffer = createBuffer(
            device,
            PLACEMENT_UNIFORM_FLOATS * Float32Array.BYTES_PER_ELEMENT,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            "paint brush hover placement uniforms",
        );
        this.hoverResultBuffer = createBuffer(
            device,
            PLACEMENT_RESULT_FLOATS * Float32Array.BYTES_PER_ELEMENT,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            "paint brush hover placement result",
        );
        this.dummyReadBuffer = createBuffer(
            device,
            PLACEMENT_RESULT_FLOATS * Float32Array.BYTES_PER_ELEMENT,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "paint brush placement dummy read storage",
        );
        this.dummyResultBuffer = createBuffer(
            device,
            PLACEMENT_RESULT_FLOATS * Float32Array.BYTES_PER_ELEMENT,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "paint brush placement dummy result storage",
        );
        this.dummyOutputBuffer = createBuffer(
            device,
            PLACEMENT_RESULT_FLOATS * Float32Array.BYTES_PER_ELEMENT,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "paint brush placement dummy output storage",
        );
        this.dummyMetaBuffer = createBuffer(
            device,
            PLACEMENT_RESULT_FLOATS * Float32Array.BYTES_PER_ELEMENT,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "paint brush placement dummy meta storage",
        );
    }

    setBrushPlacementInput(input: BrushPlacementInput | null) {
        this.hoverInput = input;
        if (!input) {
            this.hoverComputeBindGroup = null;
            this.guideBindGroup = null;
        }
    }

    setBrushSnapTargets(ribbons: RenderRibbon[]) {
        this.targetVertexBuffer?.destroy();
        this.targetInfoBuffer?.destroy();
        this.targetVertexBuffer = null;
        this.targetInfoBuffer = null;
        this.targetCount = 0;
        this.hoverComputeBindGroup = null;
        this.guideBindGroup = null;

        const targetInfos: number[] = [];
        const targetVertices: number[] = [];

        for (const ribbon of ribbons) {
            const segmentCount = ribbonSegmentCount(ribbon);
            if (segmentCount === 0) continue;

            targetInfos.push(
                targetVertices.length / FLOATS_PER_RIBBON_VERTEX,
                ribbon.vertices.length,
                ribbon.closed ? 1 : 0,
                0,
            );

            for (const vertex of ribbon.vertices) {
                targetVertices.push(
                    vertex.position[0],
                    vertex.position[1],
                    vertex.position[2],
                    vertex.u,
                    vertex.side[0],
                    vertex.side[1],
                    vertex.side[2],
                    0,
                );
            }
        }

        this.targetCount = targetInfos.length / TARGET_INFO_UINTS;
        if (this.targetCount === 0) return;

        const vertexData = new Float32Array(targetVertices);
        this.targetVertexBuffer = createBuffer(
            this.device,
            vertexData.byteLength,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "paint brush snap target vertices",
        );
        this.device.queue.writeBuffer(this.targetVertexBuffer, 0, vertexData);

        const infoData = new Uint32Array(targetInfos);
        this.targetInfoBuffer = createBuffer(
            this.device,
            infoData.byteLength,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "paint brush snap target infos",
        );
        this.device.queue.writeBuffer(this.targetInfoBuffer, 0, infoData);
    }

    encodeHoverPlacement(
        encoder: GPUCommandEncoder,
        viewProjMat: number[] | Float32Array,
        viewProjInvMat: number[] | Float32Array,
        viewInvMat: number[] | Float32Array,
    ) {
        if (!this.hoverInput) return;

        this.writeUniforms(
            this.hoverUniformBuffer,
            viewProjMat,
            viewProjInvMat,
            viewInvMat,
            this.hoverInput.point,
            this.hoverInput.brushWidth,
            this.hoverInput.viewportWidth,
            this.hoverInput.viewportHeight,
            this.hoverInput.snapToRibbons ? this.targetCount : 0,
            0,
            true,
        );

        this.hoverComputeBindGroup = this.createComputeBindGroup(
            this.hoverUniformBuffer,
            this.hoverResultBuffer,
            this.dummyReadBuffer,
            this.dummyOutputBuffer,
            this.dummyMetaBuffer,
        );
        this.guideBindGroup = this.createGuideBindGroup(this.hoverUniformBuffer, this.hoverResultBuffer);
        const pass = encoder.beginComputePass({ label: "paint brush hover placement pass" });
        pass.setPipeline(this.computePipeline);
        pass.setBindGroup(0, this.hoverComputeBindGroup);
        pass.dispatchWorkgroups(1);
        pass.end();
    }

    drawGuide(pass: GPURenderPassEncoder) {
        if (!this.hoverInput || !this.guideBindGroup) return;
        pass.setPipeline(this.guidePipeline);
        pass.setBindGroup(0, this.guideBindGroup);
        pass.draw(GUIDE_VERTEX_COUNT);
    }

    async readBrushPlacementForTest(
        viewProjMat: number[] | Float32Array,
        viewProjInvMat: number[] | Float32Array,
        viewInvMat: number[] | Float32Array,
    ): Promise<BrushPlacementReadback | null> {
        if (!this.hoverInput) return null;

        const readback = createBuffer(
            this.device,
            PLACEMENT_RESULT_FLOATS * Float32Array.BYTES_PER_ELEMENT,
            GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            "paint brush hover placement readback",
        );
        const encoder = this.device.createCommandEncoder({
            label: "paint brush hover placement readback encoder",
        });

        this.encodeHoverPlacement(encoder, viewProjMat, viewProjInvMat, viewInvMat);
        encoder.copyBufferToBuffer(
            this.hoverResultBuffer,
            0,
            readback,
            0,
            PLACEMENT_RESULT_FLOATS * Float32Array.BYTES_PER_ELEMENT,
        );
        this.device.queue.submit([encoder.finish()]);

        await readback.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(readback.getMappedRange()).slice();
        readback.unmap();
        readback.destroy();

        if (data[3] <= 0.5) return null;
        return {
            center: [data[0], data[1], data[2]],
            normal: [data[4], data[5], data[6]],
            snapped: data[7] > 0.5,
            depth: data[11],
        };
    }

    async buildPlacedRibbonFromSourcePoints({
        sourcePoints,
        view,
        brushWidth,
        snapToRibbons,
    }: StrokePlacementInput): Promise<PaintRibbon | null> {
        if (sourcePoints.length < 2) return null;

        const sourceBuffer = createSourcePointBuffer(this.device, sourcePoints);
        const outputBuffer = createBuffer(
            this.device,
            sourcePoints.length * FLOATS_PER_RIBBON_VERTEX * Float32Array.BYTES_PER_ELEMENT,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            "paint brush placed stroke vertices",
        );
        const metaBuffer = createBuffer(
            this.device,
            TARGET_INFO_UINTS * Uint32Array.BYTES_PER_ELEMENT,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            "paint brush placed stroke meta",
        );
        const outputReadback = createBuffer(
            this.device,
            sourcePoints.length * FLOATS_PER_RIBBON_VERTEX * Float32Array.BYTES_PER_ELEMENT,
            GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            "paint brush placed stroke vertices readback",
        );
        const metaReadback = createBuffer(
            this.device,
            TARGET_INFO_UINTS * Uint32Array.BYTES_PER_ELEMENT,
            GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            "paint brush placed stroke meta readback",
        );
        const uniformBuffer = createBuffer(
            this.device,
            PLACEMENT_UNIFORM_FLOATS * Float32Array.BYTES_PER_ELEMENT,
            GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            "paint brush placed stroke uniforms",
        );

        this.writeUniforms(
            uniformBuffer,
            view.viewProjMat,
            view.viewProjInvMat,
            view.viewInvMat,
            sourcePoints[0],
            brushWidth,
            view.width,
            view.height,
            snapToRibbons ? this.targetCount : 0,
            sourcePoints.length,
            true,
        );

        const bindGroup = this.createComputeBindGroup(
            uniformBuffer,
            this.dummyResultBuffer,
            sourceBuffer,
            outputBuffer,
            metaBuffer,
        );
        const outputBytes = sourcePoints.length
            * FLOATS_PER_RIBBON_VERTEX
            * Float32Array.BYTES_PER_ELEMENT;
        const encoder = this.device.createCommandEncoder({
            label: "paint brush placed stroke encoder",
        });
        const pass = encoder.beginComputePass({ label: "paint brush placed stroke pass" });
        pass.setPipeline(this.strokeComputePipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(sourcePoints.length / WORKGROUP_SIZE));
        pass.end();
        encoder.copyBufferToBuffer(outputBuffer, 0, outputReadback, 0, outputBytes);
        encoder.copyBufferToBuffer(
            metaBuffer,
            0,
            metaReadback,
            0,
            TARGET_INFO_UINTS * Uint32Array.BYTES_PER_ELEMENT,
        );
        this.device.queue.submit([encoder.finish()]);

        await Promise.all([
            outputReadback.mapAsync(GPUMapMode.READ),
            metaReadback.mapAsync(GPUMapMode.READ),
        ]);
        const output = new Float32Array(outputReadback.getMappedRange()).slice();
        const meta = new Uint32Array(metaReadback.getMappedRange()).slice();
        outputReadback.unmap();
        metaReadback.unmap();
        destroyBuffers([
            sourceBuffer,
            outputBuffer,
            metaBuffer,
            outputReadback,
            metaReadback,
            uniformBuffer,
        ]);

        const rows = meta[0];
        if (rows < 2 || rows > sourcePoints.length) return null;

        const vertices: PaintRibbonVertex[] = [];
        for (let index = 0; index < rows; index++) {
            const offset = index * FLOATS_PER_RIBBON_VERTEX;
            vertices.push({
                position: [output[offset], output[offset + 1], output[offset + 2]],
                u: output[offset + 3],
                side: [output[offset + 4], output[offset + 5], output[offset + 6]],
            });
        }

        return {
            closed: meta[1] !== 0,
            vertices,
        };
    }

    destroy() {
        this.targetVertexBuffer?.destroy();
        this.targetInfoBuffer?.destroy();
        this.hoverUniformBuffer.destroy();
        this.hoverResultBuffer.destroy();
        this.dummyReadBuffer.destroy();
        this.dummyResultBuffer.destroy();
        this.dummyOutputBuffer.destroy();
        this.dummyMetaBuffer.destroy();
    }

    private writeUniforms(
        buffer: GPUBuffer,
        viewProjMat: number[] | Float32Array,
        viewProjInvMat: number[] | Float32Array,
        viewInvMat: number[] | Float32Array,
        point: Vec2,
        brushWidth: number,
        viewportWidth: number,
        viewportHeight: number,
        targetCount: number,
        sourceCount: number,
        visible: boolean,
    ) {
        const data = new Float32Array(PLACEMENT_UNIFORM_FLOATS);
        data.set(viewProjMat, 0);
        data.set(viewProjInvMat, 16);
        data.set(viewInvMat, 32);
        data[48] = point.x;
        data[49] = point.y;
        data[50] = brushWidth;
        data[51] = visible ? 1 : 0;
        data[52] = Math.max(1, viewportWidth);
        data[53] = Math.max(1, viewportHeight);
        data[54] = targetCount;
        data[55] = sourceCount;
        this.device.queue.writeBuffer(buffer, 0, data);
    }

    private createComputeBindGroup(
        uniformBuffer: GPUBuffer,
        resultBuffer: GPUBuffer,
        sourceBuffer: GPUBuffer,
        outputBuffer: GPUBuffer,
        metaBuffer: GPUBuffer,
    ): GPUBindGroup {
        return this.device.createBindGroup({
            label: "paint brush placement bind group",
            layout: this.computeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: this.targetVertexBuffer ?? this.dummyReadBuffer } },
                { binding: 2, resource: { buffer: this.targetInfoBuffer ?? this.dummyReadBuffer } },
                { binding: 3, resource: { buffer: sourceBuffer } },
                { binding: 4, resource: { buffer: resultBuffer } },
                { binding: 5, resource: { buffer: outputBuffer } },
                { binding: 6, resource: { buffer: metaBuffer } },
            ],
        });
    }

    private createGuideBindGroup(uniformBuffer: GPUBuffer, resultBuffer: GPUBuffer): GPUBindGroup {
        return this.device.createBindGroup({
            label: "paint brush guide bind group",
            layout: this.guideBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 4, resource: { buffer: resultBuffer } },
            ],
        });
    }
}

const createSourcePointBuffer = (
    device: GPUDevice,
    sourcePoints: Vec2[],
): GPUBuffer => {
    const data = new Float32Array(Math.max(1, sourcePoints.length) * SOURCE_POINT_FLOATS);
    for (let index = 0; index < sourcePoints.length; index++) {
        const offset = index * SOURCE_POINT_FLOATS;
        data[offset] = sourcePoints[index].x;
        data[offset + 1] = sourcePoints[index].y;
    }

    const buffer = createBuffer(
        device,
        data.byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        "paint brush source points",
    );
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
};

const ribbonSegmentCount = (ribbon: RenderRibbon): number => {
    if (ribbon.vertices.length < 2) return 0;
    return ribbon.closed ? ribbon.vertices.length : ribbon.vertices.length - 1;
};

const createBuffer = (
    device: GPUDevice,
    size: number,
    usage: GPUBufferUsageFlags,
    label: string,
): GPUBuffer => device.createBuffer({
    label,
    size: Math.max(16, alignTo(size, 16)),
    usage,
});

const alignTo = (value: number, alignment: number): number => (
    Math.ceil(value / alignment) * alignment
);

const destroyBuffers = (buffers: GPUBuffer[]) => {
    for (const buffer of buffers) {
        buffer.destroy();
    }
};

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