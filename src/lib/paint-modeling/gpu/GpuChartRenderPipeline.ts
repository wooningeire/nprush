import { DEPTH_FORMAT } from "../renderer/constants.ts";
import {
    COVERAGE_EPSILON,
    SURFACE_FIELD_NORMAL_LENGTH,
    SURFACE_FIELD_STRIDE,
} from "../state/constants.ts";
import type { ChartRole, PaintChart, PaintView, Vec4 } from "../types.ts";
import shaderSource from "./paint_chart_render.wgsl?raw";
import { cameraCenter, viewForward } from "./viewUniforms.ts";

export type GpuChartRenderItem = {
    bindGroup: GPUBindGroup,
    fillVertexCount: number,
    wireVertexCount: number,
    fieldVertexCount: number,
};

const PARAM_FLOATS = 60;
const LINE_STRIDE = 4;

export class GpuChartRenderPipeline {
    readonly bindGroupLayout: GPUBindGroupLayout;
    private readonly device: GPUDevice;
    private readonly fillPipeline: GPURenderPipeline;
    private readonly wirePipeline: GPURenderPipeline;
    private readonly fieldPipeline: GPURenderPipeline;

    constructor(device: GPUDevice, format: GPUTextureFormat) {
        this.device = device;
        this.bindGroupLayout = device.createBindGroupLayout({
            label: "paint chart render bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: { type: "uniform" },
                },
            ],
        });
        const module = device.createShaderModule({
            label: "paint chart render shader",
            code: shaderSource,
        });
        const layout = device.createPipelineLayout({
            label: "paint chart render pipeline layout",
            bindGroupLayouts: [this.bindGroupLayout],
        });
        this.fillPipeline = this.createPipeline(layout, module, format, "chart_fill_vertex", "triangle-list");
        this.wirePipeline = this.createPipeline(layout, module, format, "chart_wire_vertex", "line-list");
        this.fieldPipeline = this.createPipeline(layout, module, format, "chart_field_vertex", "line-list");
    }

    createParamsBuffer(label: string): GPUBuffer {
        return this.device.createBuffer({
            label,
            size: PARAM_FLOATS * Float32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    createBindGroup(fieldBuffer: GPUBuffer, paramsBuffer: GPUBuffer, label: string): GPUBindGroup {
        return this.device.createBindGroup({
            label,
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: fieldBuffer } },
                { binding: 1, resource: { buffer: paramsBuffer } },
            ],
        });
    }

    writeParams(
        paramsBuffer: GPUBuffer,
        chart: PaintChart,
        sourceView: PaintView,
        viewProjMat: number[] | Float32Array,
    ) {
        const data = new Float32Array(PARAM_FLOATS);
        data.set(viewProjMat, 0);
        data.set(sourceView.viewProjInvMat, 16);
        data.set(cameraCenter(sourceView), 32);
        data.set(viewForward(sourceView), 36);
        data[39] = COVERAGE_EPSILON;
        data[40] = chart.width;
        data[41] = chart.height;
        data[42] = LINE_STRIDE;
        data[43] = SURFACE_FIELD_STRIDE;
        data[44] = SURFACE_FIELD_NORMAL_LENGTH;
        data.set(chartLineColor(chart.role), 48);
        data.set(chartFillColor(chart.role), 52);
        data.set(chartNormalColor(chart.role), 56);
        this.device.queue.writeBuffer(paramsBuffer, 0, data);
    }

    draw(pass: GPURenderPassEncoder, item: GpuChartRenderItem) {
        if (item.fillVertexCount > 0) {
            pass.setPipeline(this.fillPipeline);
            pass.setBindGroup(0, item.bindGroup);
            pass.draw(item.fillVertexCount);
        }
        if (item.wireVertexCount > 0) {
            pass.setPipeline(this.wirePipeline);
            pass.setBindGroup(0, item.bindGroup);
            pass.draw(item.wireVertexCount);
        }
        if (item.fieldVertexCount > 0) {
            pass.setPipeline(this.fieldPipeline);
            pass.setBindGroup(0, item.bindGroup);
            pass.draw(item.fieldVertexCount);
        }
    }

    renderItem(
        record: { bindGroup: GPUBindGroup },
        chart: PaintChart,
        showChartWireframe: boolean,
        showSurfaceField: boolean,
    ): GpuChartRenderItem {
        const cellCount = Math.max(0, chart.width - 1) * Math.max(0, chart.height - 1);
        const horizontalRows = Math.floor(Math.max(0, chart.height - 1) / LINE_STRIDE) + 1;
        const verticalColumns = Math.floor(Math.max(0, chart.width - 1) / LINE_STRIDE) + 1;
        const lineCount = horizontalRows * Math.max(0, chart.width - 1)
            + verticalColumns * Math.max(0, chart.height - 1);
        const fieldRows = Math.floor(Math.max(0, chart.height - 1) / SURFACE_FIELD_STRIDE) + 1;
        const fieldColumns = Math.floor(Math.max(0, chart.width - 1) / SURFACE_FIELD_STRIDE) + 1;
        const fieldLineCount = chart.width > 0 && chart.height > 0
            ? fieldRows * fieldColumns
            : 0;
        return {
            bindGroup: record.bindGroup,
            fillVertexCount: showChartWireframe ? cellCount * 6 : 0,
            wireVertexCount: showChartWireframe ? lineCount * 2 : 0,
            fieldVertexCount: showSurfaceField ? fieldLineCount * 2 : 0,
        };
    }

    private createPipeline(
        layout: GPUPipelineLayout,
        module: GPUShaderModule,
        format: GPUTextureFormat,
        entryPoint: string,
        topology: GPUPrimitiveTopology,
    ): GPURenderPipeline {
        return this.device.createRenderPipeline({
            label: `paint chart render ${entryPoint}`,
            layout,
            vertex: { module, entryPoint },
            fragment: {
                module,
                entryPoint: "chart_fragment",
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
                topology,
                cullMode: "none",
            },
            depthStencil: {
                format: DEPTH_FORMAT,
                depthCompare: "less-equal",
                depthWriteEnabled: false,
            },
        });
    }
}

const chartLineColor = (role: ChartRole): Vec4 => {
    if (role === "occluder") return [1, 0.48, 0.32, 0.38];
    if (role === "behind") return [0.46, 0.55, 1, 0.24];
    return [0.44, 0.92, 0.82, 0.18];
};

const chartFillColor = (role: ChartRole): Vec4 => {
    if (role === "occluder") return [1, 0.44, 0.28, 0.1];
    if (role === "behind") return [0.42, 0.5, 1, 0.08];
    return [0.34, 0.82, 0.72, 0.085];
};

const chartNormalColor = (role: ChartRole): Vec4 => {
    if (role === "occluder") return [1, 0.62, 0.42, 0.7];
    if (role === "behind") return [0.58, 0.66, 1, 0.58];
    return [0.74, 1, 0.9, 0.64];
};

