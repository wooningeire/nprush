import { markChartSeamAlongPolyline } from "./chartPainting.ts";
import { raycastPaintObjectSurface } from "./objectRaycast.ts";
import { resamplePaintPolyline } from "./strokeSampling.ts";
import { MAX_EFFECT_SAMPLES, SEAM_BRUSH_RADIUS } from "./constants.ts";
import type {
    PaintChart,
    PaintObject,
    PaintView,
    Vec2,
} from "../types.ts";

type ChartUvRun = {
    chart: PaintChart,
    points: Vec2[],
};

type ChartUvRuns = {
    runs: ChartUvRun[],
    raycastCount: number,
};

export type SeamEditPlan = {
    raycastCount: number,
    hasHits: boolean,
    apply: () => Set<string>,
};

export const planPaintChartSeams = (
    object: PaintObject,
    views: PaintView[],
    view: PaintView,
    points: Vec2[],
): SeamEditPlan => {
    const hitRuns = collectHitRuns(object, views, view, points, MAX_EFFECT_SAMPLES);

    return {
        raycastCount: hitRuns.raycastCount,
        hasHits: hitRuns.runs.length > 0,
        apply: () => {
            const touchedChartIds = new Set<string>();
            for (const run of hitRuns.runs) {
                if (markChartSeamAlongPolyline(run.chart, run.points, SEAM_BRUSH_RADIUS)) {
                    touchedChartIds.add(run.chart.id);
                }
            }
            return touchedChartIds;
        },
    };
};

const collectHitRuns = (
    object: PaintObject,
    views: PaintView[],
    view: PaintView,
    points: Vec2[],
    maxSamples: number,
): ChartUvRuns => {
    const runs: ChartUvRun[] = [];
    let raycastCount = 0;

    for (const point of resamplePaintPolyline(points, maxSamples)) {
        raycastCount += 1;
        const hit = raycastPaintObjectSurface(object, views, view, point);
        if (!hit) continue;

        const chart = object.charts.find(item => item.id === hit.chartId);
        if (!chart) continue;

        appendUvRun(runs, chart, hit.surfaceRef.uv);
    }

    return { runs, raycastCount };
};

const appendUvRun = (
    runs: ChartUvRun[],
    chart: PaintChart,
    point: Vec2,
) => {
    const previous = runs.at(-1);
    if (previous?.chart.id === chart.id) {
        previous.points.push(point);
        return;
    }
    runs.push({ chart, points: [point] });
};
