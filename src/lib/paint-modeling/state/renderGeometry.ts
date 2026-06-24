import {
    chartHasCoverage,
    forEachGridPoint,
    gridUv,
    isGridEdgeCovered,
    isGridPointCovered,
    isGridTriangleCovered,
} from "./chartPainting.ts";
import {
    MIN_DEPTH,
    SURFACE_FIELD_NORMAL_LENGTH,
    SURFACE_FIELD_NORMAL_WIDTH,
    SURFACE_FIELD_STRIDE,
} from "./constants.ts";
import {
    cameraCenter,
    viewForward,
    viewPointToWorldAtDepth,
} from "./projection.ts";
import {
    add3,
    clamp,
    cross3,
    distance3,
    dot3,
    normalize3,
    scale3,
    sub3,
} from "./vectorMath.ts";
import type {
    ChartRole,
    PaintChart,
    PaintStroke,
    PaintView,
    RenderPrimitive,
    SurfaceRef,
    Vec2,
    Vec3,
    Vec4,
} from "../types.ts";

export type SurfaceRenderPoint = {
    world: Vec3,
};

type RibbonPoint = {
    left: Vec3,
    right: Vec3,
};

type RibbonSample = {
    point: RibbonPoint,
};

export const appendChartSegments = (
    segments: RenderPrimitive[],
    chart: PaintChart,
    worldAt: (uv: Vec2) => Vec3 | null,
) => {
    if (!chartHasCoverage(chart)) return;

    const gridWorldAt = cachedChartGridWorld(chart, worldAt);
    const color = chart.role === "occluder"
        ? [1, 0.48, 0.32, 0.38] as Vec4
        : chart.role === "behind"
            ? [0.46, 0.55, 1, 0.24] as Vec4
            : [0.44, 0.92, 0.82, 0.18] as Vec4;
    const fillColor = chart.role === "occluder"
        ? [1, 0.44, 0.28, 0.1] as Vec4
        : chart.role === "behind"
            ? [0.42, 0.5, 1, 0.08] as Vec4
            : [0.34, 0.82, 0.72, 0.085] as Vec4;
    const stride = 4;

    for (let y = 1; y < chart.height; y++) {
        for (let x = 1; x < chart.width; x++) {
            const i00 = (y - 1) * chart.width + x - 1;
            const i10 = (y - 1) * chart.width + x;
            const i01 = y * chart.width + x - 1;
            const i11 = y * chart.width + x;

            if (isGridTriangleCovered(chart, i00, i10, i11)) {
                appendWorldTriangle(
                    segments,
                    gridWorldAt(x - 1, y - 1),
                    gridWorldAt(x, y - 1),
                    gridWorldAt(x, y),
                    fillColor,
                );
            }
            if (isGridTriangleCovered(chart, i00, i11, i01)) {
                appendWorldTriangle(
                    segments,
                    gridWorldAt(x - 1, y - 1),
                    gridWorldAt(x, y),
                    gridWorldAt(x - 1, y),
                    fillColor,
                );
            }
        }
    }

    for (let y = 0; y < chart.height; y += stride) {
        for (let x = 1; x < chart.width; x++) {
            const aIndex = y * chart.width + x - 1;
            const bIndex = y * chart.width + x;
            if (!isGridEdgeCovered(chart, aIndex, bIndex)) continue;
            appendWorldSegment(segments, gridWorldAt(x - 1, y), gridWorldAt(x, y), color, 1.15);
        }
    }
    for (let x = 0; x < chart.width; x += stride) {
        for (let y = 1; y < chart.height; y++) {
            const aIndex = (y - 1) * chart.width + x;
            const bIndex = y * chart.width + x;
            if (!isGridEdgeCovered(chart, aIndex, bIndex)) continue;
            appendWorldSegment(segments, gridWorldAt(x, y - 1), gridWorldAt(x, y), color, 1.15);
        }
    }

    forEachGridPoint(chart, (index, uv) => {
        if (!chart.seams[index]) return;
        const a = worldAt({ x: uv.x - 0.018, y: uv.y - 0.018 });
        const b = worldAt({ x: uv.x + 0.018, y: uv.y + 0.018 });
        const c = worldAt({ x: uv.x - 0.018, y: uv.y + 0.018 });
        const d = worldAt({ x: uv.x + 0.018, y: uv.y - 0.018 });
        appendWorldSegment(segments, a, b, [1, 0.16, 0.16, 0.95], 2.4);
        appendWorldSegment(segments, c, d, [1, 0.16, 0.16, 0.95], 2.4);
    });
};

export const appendChartSurfaceFieldSegments = (
    segments: RenderPrimitive[],
    chart: PaintChart,
    sourceView: PaintView,
    worldAt: (uv: Vec2) => Vec3 | null,
) => {
    if (!chartHasCoverage(chart)) return;

    for (let y = 0; y < chart.height; y += SURFACE_FIELD_STRIDE) {
        for (let x = 0; x < chart.width; x += SURFACE_FIELD_STRIDE) {
            const index = y * chart.width + x;
            if (!isGridPointCovered(chart, index)) continue;
            if (chart.seams[index]) continue;

            const world = worldAt(gridUv(chart, x, y));
            if (!world) continue;

            const normal = chartSurfaceNormal(chart, sourceView, worldAt, x, y, world);
            if (!normal) continue;

            const length = clamp(
                distance3(cameraCenter(sourceView), world) * SURFACE_FIELD_NORMAL_LENGTH,
                0.035,
                0.18,
            );
            appendWorldSegment(
                segments,
                world,
                add3(world, scale3(normal, length)),
                surfaceFieldColor(chart.role),
                SURFACE_FIELD_NORMAL_WIDTH,
            );
        }
    }
};

export const appendStrokeRenderSegments = (
    segments: RenderPrimitive[],
    stroke: PaintStroke,
    strokeSourceView: PaintView | null,
    surfacePointForRef: (ref: SurfaceRef) => SurfaceRenderPoint | null,
) => {
    const color = parseColor(stroke.style.color, stroke.style.opacity);
    if ((stroke.style.geometryMode ?? "billboard") === "ribbon" && strokeSourceView) {
        appendRibbonStrokeTriangles(segments, stroke, strokeSourceView, surfacePointForRef, color);
        return;
    }

    appendWorldStrokeRun(
        segments,
        stroke.samples.map(sample => surfacePointForRef(sample.surfaceRef)?.world ?? null),
        color,
        stroke.style.width,
    );
};

const appendRibbonStrokeTriangles = (
    segments: RenderPrimitive[],
    stroke: PaintStroke,
    strokeSourceView: PaintView,
    surfacePointForRef: (ref: SurfaceRef) => SurfaceRenderPoint | null,
    color: Vec4,
) => {
    let previous: RibbonSample | null = null;

    for (let index = 0; index < stroke.samples.length; index++) {
        const sample = stroke.samples[index];
        const center = surfacePointForRef(sample.surfaceRef)?.world ?? null;
        const point = center
            ? ribbonPointAt(stroke.samples, index, strokeSourceView, stroke.style.width, center)
            : null;
        if (!point) {
            previous = null;
            continue;
        }

        if (previous) {
            appendRibbonQuad(segments, previous.point, point, color);
        }
        previous = { point };
    }
};

const ribbonPointAt = (
    samples: PaintStroke["samples"],
    index: number,
    strokeSourceView: PaintView,
    width: number,
    center: Vec3,
): RibbonPoint | null => {
    const sample = samples[index];
    const sideOffset = ribbonSideOffsetAt(samples, index, strokeSourceView, width);
    if (!sideOffset) return null;

    const viewDepth = dot3(sub3(center, cameraCenter(strokeSourceView)), viewForward(strokeSourceView));
    if (!Number.isFinite(viewDepth) || viewDepth <= MIN_DEPTH) return null;

    const sideWorld = viewPointToWorldAtDepth(
        strokeSourceView,
        {
            x: sample.sourcePoint.x + sideOffset.x,
            y: sample.sourcePoint.y + sideOffset.y,
        },
        viewDepth,
    );
    if (!sideWorld) return null;

    const side = sub3(sideWorld, center);
    if (Math.hypot(side[0], side[1], side[2]) <= 1e-8) return null;

    return {
        left: sub3(center, side),
        right: add3(center, side),
    };
};

const ribbonSideOffsetAt = (
    samples: PaintStroke["samples"],
    index: number,
    strokeSourceView: PaintView,
    width: number,
): Vec2 | null => {
    const current = samples[index];
    const previous = index > 0 ? samples[index - 1] : null;
    const next = index < samples.length - 1 ? samples[index + 1] : null;
    if (!previous && !next) return null;

    const start = previous?.sourcePoint ?? current.sourcePoint;
    const end = next?.sourcePoint ?? current.sourcePoint;
    const dxPx = (end.x - start.x) * strokeSourceView.width * 0.5;
    const dyPx = (end.y - start.y) * strokeSourceView.height * 0.5;
    const lengthPx = Math.hypot(dxPx, dyPx);
    if (lengthPx <= 1e-6) return null;

    const halfWidthPx = Math.max(width, 1) * 0.5;
    return {
        x: -dyPx / lengthPx * halfWidthPx * 2 / strokeSourceView.width,
        y: dxPx / lengthPx * halfWidthPx * 2 / strokeSourceView.height,
    };
};

const appendRibbonQuad = (
    segments: RenderPrimitive[],
    a: RibbonPoint,
    b: RibbonPoint,
    color: Vec4,
) => {
    appendWorldTriangle(segments, a.left, a.right, b.right, color);
    appendWorldTriangle(segments, a.left, b.right, b.left, color);
};

export const appendWorldStrokeRun = (
    segments: RenderPrimitive[],
    points: Array<Vec3 | null>,
    color: Vec4,
    width: number,
) => {
    let run: Vec3[] = [];
    const flushRun = () => {
        if (run.length >= 2) {
            segments.push({
                kind: "stroke",
                points: run,
                color,
                width,
            });
        }
        run = [];
    };

    for (const point of points) {
        if (point) {
            run.push(point);
        } else {
            flushRun();
        }
    }
    flushRun();
};

export const parseColor = (color: string, opacity: number): Vec4 => {
    const value = color.startsWith("#") ? color.slice(1) : color;
    const r = parseInt(value.slice(0, 2), 16) / 255;
    const g = parseInt(value.slice(2, 4), 16) / 255;
    const b = parseInt(value.slice(4, 6), 16) / 255;
    return [r, g, b, opacity];
};

const cachedChartGridWorld = (
    chart: PaintChart,
    worldAt: (uv: Vec2) => Vec3 | null,
): (x: number, y: number) => Vec3 | null => {
    const worlds = new Array<Vec3 | null | undefined>(chart.width * chart.height);
    return (x: number, y: number): Vec3 | null => {
        const index = y * chart.width + x;
        if (worlds[index] !== undefined) return worlds[index];
        const world = worldAt(gridUv(chart, x, y));
        worlds[index] = world;
        return world;
    };
};

const chartSurfaceNormal = (
    chart: PaintChart,
    sourceView: PaintView,
    worldAt: (uv: Vec2) => Vec3 | null,
    x: number,
    y: number,
    world: Vec3,
): Vec3 | null => {
    const x0 = Math.max(0, x - 1);
    const x1 = Math.min(chart.width - 1, x + 1);
    const y0 = Math.max(0, y - 1);
    const y1 = Math.min(chart.height - 1, y + 1);
    if (x0 === x1 || y0 === y1) return null;

    const left = worldAt(gridUv(chart, x0, y));
    const right = worldAt(gridUv(chart, x1, y));
    const bottom = worldAt(gridUv(chart, x, y0));
    const top = worldAt(gridUv(chart, x, y1));
    if (!left || !right || !bottom || !top) return null;

    const horizontal = sub3(right, left);
    const vertical = sub3(top, bottom);
    let normal = normalize3(cross3(horizontal, vertical), viewForward(sourceView));
    const towardCamera = sub3(cameraCenter(sourceView), world);
    if (dot3(normal, towardCamera) < 0) normal = scale3(normal, -1);
    return normal;
};

const surfaceFieldColor = (role: ChartRole): Vec4 => {
    if (role === "occluder") return [1, 0.62, 0.42, 0.7];
    if (role === "behind") return [0.58, 0.66, 1, 0.58];
    return [0.74, 1, 0.9, 0.64];
};

const appendWorldSegment = (
    segments: RenderPrimitive[],
    a: Vec3 | null,
    b: Vec3 | null,
    color: Vec4,
    width?: number,
    caps?: { capStart: boolean; capEnd: boolean },
) => {
    if (!a || !b) return;
    segments.push({
        a,
        b,
        color,
        width,
        ...caps,
    });
};

const appendWorldTriangle = (
    segments: RenderPrimitive[],
    a: Vec3 | null,
    b: Vec3 | null,
    c: Vec3 | null,
    color: Vec4,
) => {
    if (!a || !b || !c) return;
    segments.push({
        kind: "triangle",
        a,
        b,
        c,
        color,
    });
};