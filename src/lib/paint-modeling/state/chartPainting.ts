import type { PaintChart, Vec2 } from "../types.ts";
import { COVERAGE_EPSILON, MIN_DEPTH } from "./constants.ts";
import { distance2d } from "./strokeSampling.ts";
import { clamp, lerp } from "./vectorMath.ts";

export type ChartPaintSample = {
    point: Vec2,
    depth: number,
};

export type ChartDepthSculptSample = {
    point: Vec2,
    depthDelta: number,
};

export type DepthWriteMode = "blend" | "replace";

export type ChartPaintRun = {
    chart: PaintChart,
    samples: ChartPaintSample[],
    radius: number,
    requireCoverage: boolean,
    depthWriteMode: DepthWriteMode,
};

export function appendPaintRun(
    runs: ChartPaintRun[],
    chart: PaintChart,
    sample: ChartPaintSample,
    radius: number,
    requireCoverage: boolean,
    depthWriteMode: DepthWriteMode = "blend",
) {
    const previous = runs.at(-1);
    if (
        previous
        && previous.chart.id === chart.id
        && previous.radius === radius
        && previous.requireCoverage === requireCoverage
        && previous.depthWriteMode === depthWriteMode
    ) {
        previous.samples.push(sample);
        return;
    }
    runs.push({ chart, samples: [sample], radius, requireCoverage, depthWriteMode });
}

export function applyStrokeToChartGeometry(
    chart: PaintChart,
    samples: ChartPaintSample[],
    radius: number,
    {
        requireCoverage,
        depthWriteMode = "blend",
    }: {
        requireCoverage: boolean,
        depthWriteMode?: DepthWriteMode,
    },
): boolean {
    if (samples.length === 0) return false;
    let changed = false;

    // Paint color stays in vector stroke samples. The chart grid stores only geometry coverage
    // and depth so brushstrokes do not get baked into a low-resolution color raster.
    forEachGridPoint(chart, (index, uv) => {
        if (requireCoverage && !isGridPointCovered(chart, index)) return;
        const nearest = nearestPaintSampleOnPolyline(samples, uv);
        if (!nearest || nearest.distance > radius) return;
        const t = nearest.distance / Math.max(radius, 1e-5);
        const influence = (1 - t * t) ** 2;

        const depth = Math.max(MIN_DEPTH, nearest.depth);
        chart.depths[index] = depthWriteMode === "replace"
            ? depth
            : lerp(chart.depths[index], depth, influence);
        chart.coverage[index] = Math.max(chart.coverage[index] ?? 0, influence);
        changed = true;
    });
    return changed;
}

export function applyDepthSculptToChartGeometry(
    chart: PaintChart,
    samples: ChartDepthSculptSample[],
    radius: number,
): boolean {
    if (samples.length === 0) return false;
    let changed = false;

    forEachGridPoint(chart, (index, uv) => {
        if (!isGridPointCovered(chart, index)) return;
        const nearest = nearestDepthSculptSampleOnPolyline(samples, uv);
        if (!nearest || nearest.distance > radius) return;
        const t = nearest.distance / Math.max(radius, 1e-5);
        const influence = (1 - t * t) ** 2;
        const nextDepth = Math.max(MIN_DEPTH, chart.depths[index] + nearest.depthDelta * influence);

        if (Math.abs(nextDepth - chart.depths[index]) <= 1e-8) return;
        chart.depths[index] = nextDepth;
        changed = true;
    });
    return changed;
}

export function markChartSeamAlongPolyline(chart: PaintChart, points: Vec2[], radius: number): boolean {
    let changed = false;
    forEachGridPoint(chart, (index, uv) => {
        if (!isGridPointCovered(chart, index)) return;
        const nearest = nearestUvOnPolyline(points, uv);
        if (nearest && nearest.distance <= radius && !chart.seams[index]) {
            chart.seams[index] = true;
            changed = true;
        }
    });
    return changed;
}

export function sampleChartDepth(chart: PaintChart, uv: Vec2): number {
    const x = clamp((uv.x * 0.5 + 0.5) * (chart.width - 1), 0, chart.width - 1);
    const y = clamp((uv.y * 0.5 + 0.5) * (chart.height - 1), 0, chart.height - 1);
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const x1 = Math.min(chart.width - 1, x0 + 1);
    const y1 = Math.min(chart.height - 1, y0 + 1);
    const fx = x - x0;
    const fy = y - y0;
    const a = chart.depths[y0 * chart.width + x0];
    const b = chart.depths[y0 * chart.width + x1];
    const c = chart.depths[y1 * chart.width + x0];
    const d = chart.depths[y1 * chart.width + x1];
    return lerp(lerp(a, b, fx), lerp(c, d, fx), fy);
}

export function chartHasCoverage(chart: PaintChart): boolean {
    return chart.coverage.some(value => value > COVERAGE_EPSILON);
}

export function isGridPointCovered(chart: PaintChart, index: number): boolean {
    return (chart.coverage[index] ?? 0) > COVERAGE_EPSILON;
}

export function isGridEdgeCovered(chart: PaintChart, a: number, b: number): boolean {
    return isGridPointCovered(chart, a) && isGridPointCovered(chart, b);
}

export function isGridTriangleCovered(chart: PaintChart, a: number, b: number, c: number): boolean {
    return isGridPointCovered(chart, a) && isGridPointCovered(chart, b) && isGridPointCovered(chart, c);
}

export function forEachGridPoint(chart: PaintChart, fn: (index: number, uv: Vec2) => void) {
    for (let y = 0; y < chart.height; y++) {
        for (let x = 0; x < chart.width; x++) {
            fn(y * chart.width + x, gridUv(chart, x, y));
        }
    }
}

export function gridUv(chart: PaintChart, x: number, y: number): Vec2 {
    return {
        x: chart.width <= 1 ? 0 : x / (chart.width - 1) * 2 - 1,
        y: chart.height <= 1 ? 0 : y / (chart.height - 1) * 2 - 1,
    };
}

function nearestUvOnPolyline(points: Vec2[], uv: Vec2): { distance: number } | null {
    if (points.length === 0) return null;
    if (points.length === 1) {
        return { distance: distance2d(points[0], uv) };
    }

    let bestDistance = Number.POSITIVE_INFINITY;
    for (let i = 1; i < points.length; i++) {
        const nearest = nearestPointOnSegment(uv, points[i - 1], points[i]);
        if (nearest.distance < bestDistance) {
            bestDistance = nearest.distance;
        }
    }
    return { distance: bestDistance };
}

function nearestPaintSampleOnPolyline(
    samples: ChartPaintSample[],
    uv: Vec2,
): { distance: number; depth: number } | null {
    if (samples.length === 0) return null;
    if (samples.length === 1) {
        return {
            distance: distance2d(samples[0].point, uv),
            depth: samples[0].depth,
        };
    }

    let bestDistance = Number.POSITIVE_INFINITY;
    let bestDepth = samples[0].depth;
    for (let i = 1; i < samples.length; i++) {
        const previous = samples[i - 1];
        const current = samples[i];
        const nearest = nearestPointOnSegment(uv, previous.point, current.point);
        if (nearest.distance < bestDistance) {
            bestDistance = nearest.distance;
            bestDepth = lerp(previous.depth, current.depth, nearest.t);
        }
    }
    return { distance: bestDistance, depth: bestDepth };
}

function nearestDepthSculptSampleOnPolyline(
    samples: ChartDepthSculptSample[],
    uv: Vec2,
): { distance: number; depthDelta: number } | null {
    if (samples.length === 0) return null;
    if (samples.length === 1) {
        return {
            distance: distance2d(samples[0].point, uv),
            depthDelta: samples[0].depthDelta,
        };
    }

    let bestDistance = Number.POSITIVE_INFINITY;
    let bestDepthDelta = samples[0].depthDelta;
    for (let i = 1; i < samples.length; i++) {
        const previous = samples[i - 1];
        const current = samples[i];
        const nearest = nearestPointOnSegment(uv, previous.point, current.point);
        if (nearest.distance < bestDistance) {
            bestDistance = nearest.distance;
            bestDepthDelta = lerp(previous.depthDelta, current.depthDelta, nearest.t);
        }
    }
    return { distance: bestDistance, depthDelta: bestDepthDelta };
}

function nearestPointOnSegment(point: Vec2, a: Vec2, b: Vec2): { distance: number; t: number } {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const lengthSquared = dx * dx + dy * dy;
    const t = lengthSquared <= 1e-10
        ? 0
        : clamp(((point.x - a.x) * dx + (point.y - a.y) * dy) / lengthSquared, 0, 1);
    return {
        t,
        distance: Math.hypot(point.x - (a.x + dx * t), point.y - (a.y + dy * t)),
    };
}