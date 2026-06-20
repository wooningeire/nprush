import { COVERAGE_EPSILON } from "../state/constants.ts";
import type { PaintChart, PaintView, Vec2, Vec3 } from "../types.ts";
import shaderSource from "./paint_chart_raycast.wgsl?raw";
import { cameraCenter, viewForward } from "./viewUniforms.ts";

export type GpuChartRaycastTarget = {
    chart: PaintChart,
    sourceView: PaintView,
    fieldBuffer: GPUBuffer,
};

export type GpuChartRaycastHit = {
    chartIndex: number,
    uv: Vec2,
    world: Vec3,
    viewDepth: number,
};

const WORKGROUP_SIZE = 64;
const PARAM_WORDS = 48;
const PARAM_BYTES = PARAM_WORDS * Uint32Array.BYTES_PER_ELEMENT;
const RESULT_BYTES = 32;
const RESULT_WORDS = RESULT_BYTES / Uint32Array.BYTES_PER_ELEMENT;
const HIT_INFINITY_BITS = 0x7f800000;
const NO_HIT_INDEX = 0xffffffff;

export class GpuChartRaycastPipeline {
    private readonly device: GPUDevice;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private readonly pipeline: GPUComputePipeline;

    constructor(device: GPUDevice) {
        this.device = device;
        this.bindGroupLayout = device.createBindGroupLayout({
            label: "paint chart raycast bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "storage" },
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: { type: "uniform" },
                },
            ],
        });
        const module = device.createShaderModule({
            label: "paint chart raycast shader",
            code: shaderSource,
        });
        this.pipeline = device.createComputePipeline({
            label: "paint chart raycast pipeline",
            layout: device.createPipelineLayout({
                label: "paint chart raycast pipeline layout",
                bindGroupLayouts: [this.bindGroupLayout],
            }),
            compute: {
                module,
                entryPoint: "chart_raycast",
            },
        });
    }

    async raycastCharts(
        targets: GpuChartRaycastTarget[],
        view: PaintView,
        points: Vec2[],
    ): Promise<Array<GpuChartRaycastHit | null>> {
        if (points.length === 0) return [];
        if (targets.length === 0) return points.map(() => null);

        const pointsBuffer = this.createBufferWithData(
            "paint chart raycast points",
            packPoints(points),
            GPUBufferUsage.STORAGE,
        );
        const resultBuffer = this.createBufferWithData(
            "paint chart raycast results",
            initialResults(points.length),
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        );
        const readbackBuffer = this.device.createBuffer({
            label: "paint chart raycast readback",
            size: points.length * RESULT_BYTES,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const paramBuffers: GPUBuffer[] = [];

        const encoder = this.device.createCommandEncoder({ label: "paint chart raycast encoder" });
        const pass = encoder.beginComputePass({ label: "paint chart raycast pass" });
        pass.setPipeline(this.pipeline);

        for (let chartIndex = 0; chartIndex < targets.length; chartIndex++) {
            const target = targets[chartIndex];
            const dispatchCount = triangleInvocationCount(target.chart, points.length);
            if (dispatchCount === 0) continue;

            const paramsBuffer = this.createBufferWithData(
                `paint chart raycast params ${target.chart.id}`,
                packParams(target, view, points.length, chartIndex),
                GPUBufferUsage.UNIFORM,
            );
            paramBuffers.push(paramsBuffer);
            pass.setBindGroup(0, this.createBindGroup(target.fieldBuffer, pointsBuffer, resultBuffer, paramsBuffer));
            pass.dispatchWorkgroups(Math.ceil(dispatchCount / WORKGROUP_SIZE));
        }

        pass.end();
        encoder.copyBufferToBuffer(resultBuffer, 0, readbackBuffer, 0, points.length * RESULT_BYTES);
        this.device.queue.submit([encoder.finish()]);

        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const results = readResults(new DataView(readbackBuffer.getMappedRange()), targets.length, points.length);
        readbackBuffer.unmap();

        pointsBuffer.destroy();
        resultBuffer.destroy();
        readbackBuffer.destroy();
        for (const buffer of paramBuffers) buffer.destroy();

        return results;
    }

    private createBindGroup(
        fieldBuffer: GPUBuffer,
        pointsBuffer: GPUBuffer,
        resultBuffer: GPUBuffer,
        paramsBuffer: GPUBuffer,
    ): GPUBindGroup {
        return this.device.createBindGroup({
            label: "paint chart raycast bind group",
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: fieldBuffer } },
                { binding: 1, resource: { buffer: pointsBuffer } },
                { binding: 2, resource: { buffer: resultBuffer } },
                { binding: 3, resource: { buffer: paramsBuffer } },
            ],
        });
    }

    private createBufferWithData(
        label: string,
        data: ArrayBufferView,
        usage: GPUBufferUsageFlags,
    ): GPUBuffer {
        const buffer = this.device.createBuffer({
            label,
            size: data.byteLength,
            usage,
            mappedAtCreation: true,
        });
        new Uint8Array(buffer.getMappedRange()).set(
            new Uint8Array(data.buffer, data.byteOffset, data.byteLength),
        );
        buffer.unmap();
        return buffer;
    }
}

const packPoints = (points: Vec2[]): Float32Array => {
    const data = new Float32Array(points.length * 2);
    for (let index = 0; index < points.length; index++) {
        data[index * 2] = points[index].x;
        data[index * 2 + 1] = points[index].y;
    }
    return data;
};

const initialResults = (count: number): Uint32Array => {
    const data = new Uint32Array(count * RESULT_WORDS);
    for (let index = 0; index < count; index++) {
        data[index * RESULT_WORDS] = HIT_INFINITY_BITS;
        data[index * RESULT_WORDS + 1] = NO_HIT_INDEX;
    }
    return data;
};

const packParams = (
    target: GpuChartRaycastTarget,
    view: PaintView,
    pointCount: number,
    chartIndex: number,
): Uint32Array => {
    const buffer = new ArrayBuffer(PARAM_BYTES);
    const f32 = new Float32Array(buffer);
    const u32 = new Uint32Array(buffer);
    f32.set(view.viewProjInvMat, 0);
    f32.set(target.sourceView.viewProjInvMat, 16);
    f32.set(cameraCenter(view), 32);
    f32[35] = target.chart.projectionMode === "ray-depth" ? 1 : 0;
    f32.set(cameraCenter(target.sourceView), 36);
    f32[39] = COVERAGE_EPSILON;
    f32.set(viewForward(target.sourceView), 40);
    u32[44] = target.chart.width;
    u32[45] = target.chart.height;
    u32[46] = pointCount;
    u32[47] = chartIndex;
    return u32;
};

const triangleInvocationCount = (chart: PaintChart, pointCount: number): number => {
    const cellCount = Math.max(0, chart.width - 1) * Math.max(0, chart.height - 1);
    return cellCount * 2 * pointCount;
};

const readResults = (
    data: DataView,
    chartCount: number,
    pointCount: number,
): Array<GpuChartRaycastHit | null> => {
    const results: Array<GpuChartRaycastHit | null> = [];
    for (let index = 0; index < pointCount; index++) {
        const offset = index * RESULT_BYTES;
        const chartIndex = data.getUint32(offset + 4, true);
        if (chartIndex === NO_HIT_INDEX || chartIndex >= chartCount) {
            results.push(null);
            continue;
        }
        results.push({
            chartIndex,
            viewDepth: data.getFloat32(offset, true),
            uv: {
                x: data.getFloat32(offset + 8, true),
                y: data.getFloat32(offset + 12, true),
            },
            world: [
                data.getFloat32(offset + 16, true),
                data.getFloat32(offset + 20, true),
                data.getFloat32(offset + 24, true),
            ],
        });
    }
    return results;
};