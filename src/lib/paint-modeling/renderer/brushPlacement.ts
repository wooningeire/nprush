import brushGuideShaderSource from "../paint_brush_guide.wgsl?raw";
import brushPlacementKernelShaderSource from "../paint_brush_placement.wgsl?raw";
import brushPlacementPreludeShaderSource from "../paint_brush_placement_prelude.wgsl?raw";

import {
    BrushPlacementMode,
    BrushPlacementProvenance,
    type BrushPlacementMode as BrushPlacementModeValue,
    type BrushPlacementProvenance as BrushPlacementProvenanceValue,
    type ConstructionPlane,
    type PaintRibbon,
    type PaintRibbonVertex,
    type RenderRibbon,
    type Vec2,
} from "../types.ts";
import { FLOATS_PER_RIBBON_VERTEX } from "./constants.ts";
import {
    GUIDE_VERTEX_COUNT,
    PLACEMENT_RESULT_FLOATS,
    PLACEMENT_UNIFORM_FLOATS,
    TARGET_INFO_UINTS,
    WORKGROUP_SIZE,
    placementModeUniformValue,
    placementModeUsesTargets,
    provenanceFromUniformValue,
    type BrushPlacementInput,
    type BrushPlacementReadback,
    type StrokePlacementInput,
} from "./brushPlacementContract.ts";
import {
    createBrushGuidePipeline,
    createBuffer,
    createLoggedShaderModule,
    createSourcePointBuffer,
    destroyBuffers,
} from "./brushPlacementGpu.ts";
import { createBrushSurfaceTargetData } from "./brushPlacementTargets.ts";

const brushPlacementShaderSource = [
    brushPlacementPreludeShaderSource,
    brushPlacementKernelShaderSource,
].join(String.fromCharCode(10));

export class BrushPlacementManager {
    private readonly device: GPUDevice;
    private readonly computeBindGroupLayout: GPUBindGroupLayout;
    private readonly guideBindGroupLayout: GPUBindGroupLayout;
    private readonly computePipeline: GPUComputePipeline;
    private readonly directStrokeComputePipeline: GPUComputePipeline;
    private readonly strokeComputePipeline: GPUComputePipeline;
    private readonly guideXrayPipeline: GPURenderPipeline;
    private readonly guidePipeline: GPURenderPipeline;
    private readonly hoverUniformBuffer: GPUBuffer;
    private readonly hoverResultBuffer: GPUBuffer;
    private readonly dummyReadBuffer: GPUBuffer;

    private readonly dummyOutputBuffer: GPUBuffer;
    private readonly dummyMetaBuffer: GPUBuffer;
    private targetVertexBuffer: GPUBuffer | null = null;
    private targetInfoBuffer: GPUBuffer | null = null;
    private targetCount = 0;
    private hoverInput: BrushPlacementInput | null = null;
    private hoverComputeBindGroup: GPUBindGroup | null = null;
    private guideBindGroup: GPUBindGroup | null = null;
    private lastStrokeProvenance: BrushPlacementProvenanceValue[] = [];

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
        this.directStrokeComputePipeline = device.createComputePipeline({
            label: "paint brush direct surface placement pipeline",
            layout: computePipelineLayout,
            compute: {
                module: placementModule,
                entryPoint: "compute_direct_stroke",
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
        this.guideXrayPipeline = createBrushGuidePipeline(
            device,
            guidePipelineLayout,
            guideModule,
            format,
            "guide_fragment_xray",
            "always",
            "paint construction plane xray guide pipeline",
        );
        this.guidePipeline = createBrushGuidePipeline(
            device,
            guidePipelineLayout,
            guideModule,
            format,
            "guide_fragment",
            "less-equal",
            "paint brush guide pipeline",
        );

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

    setBrushSurfaceTargets(ribbons: RenderRibbon[]) {
        this.targetVertexBuffer?.destroy();
        this.targetInfoBuffer?.destroy();
        this.targetVertexBuffer = null;
        this.targetInfoBuffer = null;
        this.targetCount = 0;
        this.hoverComputeBindGroup = null;
        this.guideBindGroup = null;

        const targets = createBrushSurfaceTargetData(ribbons);
        if (!targets) return;

        this.targetCount = targets.count;
        this.targetVertexBuffer = createBuffer(
            this.device,
            targets.vertices.byteLength,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "paint brush surface target vertices",
        );
        this.device.queue.writeBuffer(this.targetVertexBuffer, 0, targets.vertices);

        this.targetInfoBuffer = createBuffer(
            this.device,
            targets.infos.byteLength,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            "paint brush surface target infos",
        );
        this.device.queue.writeBuffer(this.targetInfoBuffer, 0, targets.infos);
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
            placementModeUsesTargets(this.hoverInput.placementMode) ? this.targetCount : 0,
            0,
            this.hoverInput.placementMode,
            this.hoverInput.constructionPlane,
            this.hoverInput.pointerVisible,
            this.hoverInput.planeSize,
            this.hoverInput.startPoint,
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
        pass.setBindGroup(0, this.guideBindGroup);
        pass.setPipeline(this.guideXrayPipeline);
        pass.draw(GUIDE_VERTEX_COUNT);
        pass.setPipeline(this.guidePipeline);
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
        const provenance = provenanceFromUniformValue(Math.round(data[7]));
        return {
            center: [data[0], data[1], data[2]],
            normal: [data[4], data[5], data[6]],
            tangent: [data[8], data[9], data[10]],
            depth: data[11],
            bitangent: [data[12], data[13], data[14]],
            provenance,
            snapped: provenance === BrushPlacementProvenance.Surface
                || provenance === BrushPlacementProvenance.Bridge
                || provenance === BrushPlacementProvenance.StartDepth
                || provenance === BrushPlacementProvenance.StartPlane,
        };
    }

    async buildPlacedRibbonFromSourcePoints({
        sourcePoints,
        sourceProjection,
        brushWidth,
        placementMode,
        constructionPlane,
    }: StrokePlacementInput): Promise<PaintRibbon | null> {
        this.lastStrokeProvenance = [];
        if (sourcePoints.length < 2) return null;

        const sourceBuffer = createSourcePointBuffer(this.device, sourcePoints);
        const directResultBuffer = createBuffer(
            this.device,
            sourcePoints.length * PLACEMENT_RESULT_FLOATS * Float32Array.BYTES_PER_ELEMENT,
            GPUBufferUsage.STORAGE,
            "paint brush direct surface placement results",
        );
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
            sourceProjection.viewProjMat,
            sourceProjection.viewProjInvMat,
            sourceProjection.viewInvMat,
            sourcePoints[0],
            brushWidth,
            sourceProjection.width,
            sourceProjection.height,
            placementModeUsesTargets(placementMode) ? this.targetCount : 0,
            sourcePoints.length,
            placementMode,
            constructionPlane,
            true,
            // Batch placement does not draw the hover guide, so its guide-only radius is unused.
            1,
            null,
        );

        const bindGroup = this.createComputeBindGroup(
            uniformBuffer,
            directResultBuffer,
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
        const workgroupCount = Math.ceil(sourcePoints.length / WORKGROUP_SIZE);
        const directPass = encoder.beginComputePass({
            label: "paint brush direct surface placement pass",
        });
        directPass.setPipeline(this.directStrokeComputePipeline);
        directPass.setBindGroup(0, bindGroup);
        directPass.dispatchWorkgroups(workgroupCount);
        directPass.end();

        const placementPass = encoder.beginComputePass({
            label: "paint brush resolved stroke placement pass",
        });
        placementPass.setPipeline(this.strokeComputePipeline);
        placementPass.setBindGroup(0, bindGroup);
        placementPass.dispatchWorkgroups(workgroupCount);
        placementPass.end();
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
            directResultBuffer,
            outputBuffer,
            metaBuffer,
            outputReadback,
            metaReadback,
            uniformBuffer,
        ]);

        const rows = meta[0];
        if (rows < 2 || rows > sourcePoints.length) return null;

        const vertices: PaintRibbonVertex[] = [];
        const provenance: BrushPlacementProvenanceValue[] = [];
        for (let index = 0; index < rows; index++) {
            const offset = index * FLOATS_PER_RIBBON_VERTEX;
            vertices.push({
                position: [output[offset], output[offset + 1], output[offset + 2]],
                u: output[offset + 3],
                side: [output[offset + 4], output[offset + 5], output[offset + 6]],
            });
            provenance.push(provenanceFromUniformValue(Math.round(output[offset + 7])));
        }
        this.lastStrokeProvenance = provenance;

        return {
            closed: meta[1] !== 0,
            vertices,
        };
    }

    readLastStrokeProvenanceForTest(): BrushPlacementProvenanceValue[] {
        return [...this.lastStrokeProvenance];
    }

    destroy() {
        this.targetVertexBuffer?.destroy();
        this.targetInfoBuffer?.destroy();
        this.hoverUniformBuffer.destroy();
        this.hoverResultBuffer.destroy();
        this.dummyReadBuffer.destroy();
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
        placementMode: BrushPlacementModeValue,
        constructionPlane: ConstructionPlane,
        pointerVisible: boolean,
        planeSize: number,
        startPoint: Vec2 | null,
    ) {
        const data = new Float32Array(PLACEMENT_UNIFORM_FLOATS);
        data.set(viewProjMat, 0);
        data.set(viewProjInvMat, 16);
        data.set(viewInvMat, 32);
        data[48] = point.x;
        data[49] = point.y;
        data[50] = brushWidth;
        data[51] = pointerVisible ? 1 : 0;
        data[52] = Math.max(1, viewportWidth);
        data[53] = Math.max(1, viewportHeight);
        data[54] = targetCount;
        data[55] = sourceCount;
        data[56] = placementModeUniformValue(placementMode);
        data[57] = placementMode === BrushPlacementMode.ConstructionPlane ? 1 : 0;
        data.set(constructionPlane.origin, 60);
        data[63] = Math.max(0.01, planeSize);
        data.set(constructionPlane.normal, 64);
        if (startPoint) {
            data[68] = startPoint.x;
            data[69] = startPoint.y;
            data[70] = 1;
        }
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
