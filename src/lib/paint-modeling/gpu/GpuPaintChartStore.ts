import type { PaintChart, PaintObject, PaintView, SurfaceHit, Vec2 } from "../types.ts";
import {
    GpuChartPaintPipeline,
    type GpuChartPaintOptions,
    type GpuChartPaintSample,
} from "./GpuChartPaintPipeline.ts";
import {
    GpuChartRenderPipeline,
    type GpuChartRenderItem,
} from "./GpuChartRenderPipeline.ts";
import {
    GpuChartRaycastPipeline,
    type GpuChartRaycastTarget,
} from "./GpuChartRaycastPipeline.ts";

type GpuPaintChartRecord = {
    chartId: string,
    texelCount: number,
    fieldBuffer: GPUBuffer,
    renderParamsBuffer: GPUBuffer,
    bindGroup: GPUBindGroup,
};

const FLOATS_PER_TEXEL = 2;

export class GpuPaintChartStore {
    private readonly device: GPUDevice;
    private readonly paintPipeline: GpuChartPaintPipeline;
    private readonly renderPipeline: GpuChartRenderPipeline;
    private readonly raycastPipeline: GpuChartRaycastPipeline;
    private readonly records = new Map<string, GpuPaintChartRecord>();

    constructor(device: GPUDevice, format: GPUTextureFormat) {
        this.device = device;
        this.paintPipeline = new GpuChartPaintPipeline(device);
        this.renderPipeline = new GpuChartRenderPipeline(device, format);
        this.raycastPipeline = new GpuChartRaycastPipeline(device);
    }

    syncObjects(objects: PaintObject[]) {
        const liveChartIds = new Set<string>();
        for (const object of objects) {
            for (const chart of object.charts) {
                liveChartIds.add(chart.id);
                this.syncChart(chart);
            }
        }

        for (const [chartId, record] of this.records) {
            if (liveChartIds.has(chartId)) continue;
            this.destroyRecord(record);
            this.records.delete(chartId);
        }
    }

    applyPaintRun(
        encoder: GPUCommandEncoder,
        chart: PaintChart,
        samples: GpuChartPaintSample[],
        options: GpuChartPaintOptions,
    ) {
        const texelCount = chart.width * chart.height;
        const record = this.records.get(chart.id);
        if (!record || record.texelCount !== texelCount) {
            this.syncChart(chart);
            return;
        }

        this.paintPipeline.dispatch(
            encoder,
            record.fieldBuffer,
            chart.width,
            chart.height,
            samples,
            options,
        );
    }

    prepareRenderItems(
        objects: PaintObject[],
        views: PaintView[],
        viewProjMat: number[] | Float32Array,
        showChartWireframe: boolean,
        showSurfaceField: boolean,
    ): GpuChartRenderItem[] {
        if (!showChartWireframe && !showSurfaceField) return [];
        const viewById = new Map(views.map(view => [view.id, view]));
        const items: GpuChartRenderItem[] = [];

        for (const object of objects) {
            if (!object.visible) continue;
            for (const chart of object.charts) {
                const sourceView = viewById.get(chart.sourceViewId);
                if (!sourceView) continue;
                const record = this.getOrCreateSyncedRecord(chart);
                this.renderPipeline.writeParams(record.renderParamsBuffer, chart, sourceView, viewProjMat);
                items.push(this.renderPipeline.renderItem(record, chart, showChartWireframe, showSurfaceField));
            }
        }

        return items;
    }

    async raycastObjectSurfaceBatch(
        object: PaintObject,
        views: PaintView[],
        view: PaintView,
        points: Vec2[],
        excludeChartId?: string,
    ): Promise<Array<SurfaceHit | null>> {
        if (!object.visible || points.length === 0) return points.map(() => null);

        const viewById = new Map(views.map(item => [item.id, item]));
        const targets: GpuChartRaycastTarget[] = [];
        for (const chart of object.charts) {
            if (chart.id === excludeChartId) continue;
            const sourceView = viewById.get(chart.sourceViewId);
            if (!sourceView) continue;
            const record = this.getOrCreateSyncedRecord(chart);
            targets.push({ chart, sourceView, fieldBuffer: record.fieldBuffer });
        }

        const hits = await this.raycastPipeline.raycastCharts(targets, view, points);
        return hits.map(hit => {
            if (!hit) return null;
            const target = targets[hit.chartIndex];
            return {
                objectId: object.id,
                chartId: target.chart.id,
                surfaceRef: { chartId: target.chart.id, uv: hit.uv },
                world: hit.world,
                viewDepth: hit.viewDepth,
            };
        });
    }

    drawRenderItems(pass: GPURenderPassEncoder, items: GpuChartRenderItem[]) {
        for (const item of items) {
            this.renderPipeline.draw(pass, item);
        }
    }

    destroy() {
        for (const record of this.records.values()) {
            this.destroyRecord(record);
        }
        this.records.clear();
        this.paintPipeline.destroy();
    }

    private syncChart(chart: PaintChart) {
        const texelCount = chart.width * chart.height;
        const record = this.getOrCreateRecord(chart.id, texelCount);
        this.device.queue.writeBuffer(record.fieldBuffer, 0, packChartFields(chart));
    }

    private getOrCreateSyncedRecord(chart: PaintChart): GpuPaintChartRecord {
        const texelCount = chart.width * chart.height;
        const existing = this.records.get(chart.id);
        if (existing && existing.texelCount === texelCount) return existing;
        this.syncChart(chart);
        return this.records.get(chart.id)!;
    }

    private getOrCreateRecord(chartId: string, texelCount: number): GpuPaintChartRecord {
        const existing = this.records.get(chartId);
        if (existing && existing.texelCount === texelCount) return existing;
        if (existing) this.destroyRecord(existing);

        const fieldBuffer = this.device.createBuffer({
            label: `paint chart fields ${chartId}`,
            size: texelCount * FLOATS_PER_TEXEL * Float32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        const renderParamsBuffer = this.renderPipeline.createParamsBuffer(`paint chart render params ${chartId}`);
        const record = {
            chartId,
            texelCount,
            fieldBuffer,
            renderParamsBuffer,
            bindGroup: this.renderPipeline.createBindGroup(
                fieldBuffer,
                renderParamsBuffer,
                `paint chart render bind group ${chartId}`,
            ),
        };
        this.records.set(chartId, record);
        return record;
    }

    private destroyRecord(record: GpuPaintChartRecord) {
        record.fieldBuffer.destroy();
        record.renderParamsBuffer.destroy();
    }
}

const packChartFields = (chart: PaintChart): Float32Array => {
    const texelCount = chart.width * chart.height;
    const fields = new Float32Array(texelCount * FLOATS_PER_TEXEL);
    for (let index = 0; index < texelCount; index++) {
        fields[index * FLOATS_PER_TEXEL] = chart.depths[index] ?? 0;
        fields[index * FLOATS_PER_TEXEL + 1] = chart.coverage[index] ?? 0;
    }
    return fields;
};