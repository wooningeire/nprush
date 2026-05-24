import {
    implicitBodyPoint,
    projectPoint,
} from "./implicitBody.ts";
import type {
    ContourStroke,
    ContourStrokeKind,
    ContourView,
    ImplicitBodyParams,
    Vec2,
    Vec3,
} from "./types.ts";

export type CrossViewGuideStyle = "ray" | "proxy" | "surface";

export interface CrossViewGuide {
    id: string;
    strokeId: string;
    kind: ContourStrokeKind;
    sourceViewId: string;
    style: CrossViewGuideStyle;
    points: Vec2[];
    vertices?: CrossViewGuideVertex[];
    depthNdc?: number;
}

export interface CrossViewGuideVertex {
    point: Vec2;
    strokePointIndex: number;
    depthNdc: number;
    depthOffset: number;
    depthDirection?: Vec2;
}

export function buildCrossViewGuides({
    strokes,
    views,
    currentViewProjMat,
    shapeParams = null,
}: {
    strokes: ContourStroke[];
    views: ContourView[];
    currentViewProjMat: number[] | Float32Array;
    shapeParams?: ImplicitBodyParams | null;
}): CrossViewGuide[] {
    const viewById = new Map(views.map(view => [view.id, view]));
    const guides: CrossViewGuide[] = [];
    const surfaceCache = new Map<string, SurfaceProjectionSample[]>();

    for (const stroke of strokes) {
        const sourceView = viewById.get(stroke.viewId);
        if (!sourceView) continue;

        const proxy = projectStrokeAtSourceDepth(stroke, sourceView, currentViewProjMat);
        if (proxy.points.length > 1) {
            guides.push({
                id: `${stroke.id}-proxy`,
                strokeId: stroke.id,
                kind: stroke.kind,
                sourceViewId: sourceView.id,
                style: "proxy",
                points: proxy.points,
                vertices: proxy.vertices,
                depthNdc: averageDepth(proxy.vertices),
            });
        }

        if (shapeParams) {
            const surface = projectStrokeToBodySurface(
                stroke,
                sourceView,
                currentViewProjMat,
                shapeParams,
                surfaceCache,
            );
            if (surface.length > 1) {
                guides.push({
                    id: `${stroke.id}-surface`,
                    strokeId: stroke.id,
                    kind: stroke.kind,
                    sourceViewId: sourceView.id,
                    style: "surface",
                    points: surface,
                    depthNdc: strokeDepthForView(stroke, sourceView),
                });
            }
        }

        const rayPoints = sampleIndexedPolyline(stroke.resampledPoints, 10);
        for (let i = 0; i < rayPoints.length; i++) {
            const ray = projectSourceRay(stroke, rayPoints[i], sourceView, currentViewProjMat);
            if (ray.length === 2) {
                guides.push({
                    id: `${stroke.id}-ray-${i}`,
                    strokeId: stroke.id,
                    kind: stroke.kind,
                    sourceViewId: sourceView.id,
                    style: "ray",
                    points: ray,
                    depthNdc: strokeDepthForView(stroke, sourceView),
                });
            }
        }
    }

    return guides.slice(0, 360);
}

interface SurfaceProjectionSample {
    world: Vec3;
    sourceProjection: Vec2;
}

function projectStrokeAtSourceDepth(
    stroke: ContourStroke,
    sourceView: ContourView,
    currentViewProjMat: number[] | Float32Array,
): { points: Vec2[]; vertices: CrossViewGuideVertex[] } {
    const vertices: CrossViewGuideVertex[] = [];
    for (const sample of sampleIndexedPolyline(stroke.resampledPoints, 72)) {
        const depth = strokeDepthAtIndex(stroke, sourceView, sample.index);
        const world = strokeWorldPointAtIndex(stroke, sourceView, sample.index, sample.point);
        if (!world) continue;
        const projected = projectPoint(currentViewProjMat, world);
        if (isProjectedGuidePoint(projected)) {
            const rayDirection = sourceRayDirection(sourceView.viewProjInvMat, sample.point.x, sample.point.y);
            vertices.push({
                point: projected,
                strokePointIndex: sample.index,
                depthNdc: depth,
                depthOffset: strokeDepthOffsetAtIndex(stroke, sample.index),
                depthDirection: rayDirection
                    ? projectedDepthDirection(world, rayDirection, currentViewProjMat)
                    : undefined,
            });
        }
    }
    return {
        points: vertices.map(vertex => vertex.point),
        vertices,
    };
}

function projectStrokeToBodySurface(
    stroke: ContourStroke,
    sourceView: ContourView,
    currentViewProjMat: number[] | Float32Array,
    shapeParams: ImplicitBodyParams,
    cache: Map<string, SurfaceProjectionSample[]>,
): Vec2[] {
    let samples = cache.get(sourceView.id);
    if (!samples) {
        samples = projectedBodySurfaceSamples(shapeParams, sourceView);
        cache.set(sourceView.id, samples);
    }
    if (samples.length === 0) return [];

    const out: Vec2[] = [];
    for (const point of samplePolyline(stroke.resampledPoints, 54)) {
        const nearest = nearestSurfaceSample(point, samples);
        if (!nearest || nearest.distance2 > 0.012) continue;
        const projected = projectPoint(currentViewProjMat, nearest.sample.world);
        if (isProjectedGuidePoint(projected)) out.push(projected);
    }
    return simplifyProjectedPolyline(out);
}

function projectedBodySurfaceSamples(
    shapeParams: ImplicitBodyParams,
    sourceView: ContourView,
): SurfaceProjectionSample[] {
    const samples: SurfaceProjectionSample[] = [];
    for (let yi = 0; yi <= 16; yi++) {
        const t = yi / 16;
        for (let ti = 0; ti < 40; ti++) {
            const world = implicitBodyPoint(shapeParams, t, ti / 40 * Math.PI * 2);
            const sourceProjection = projectPoint(sourceView.viewProjMat, world);
            if (isProjectedGuidePoint(sourceProjection)) {
                samples.push({ world, sourceProjection });
            }
        }
    }
    return samples;
}

function nearestSurfaceSample(
    point: Vec2,
    samples: SurfaceProjectionSample[],
): { sample: SurfaceProjectionSample; distance2: number } | null {
    let best: SurfaceProjectionSample | null = null;
    let bestDistance2 = Infinity;
    for (const sample of samples) {
        const dx = point.x - sample.sourceProjection.x;
        const dy = point.y - sample.sourceProjection.y;
        const distance2 = dx * dx + dy * dy;
        if (distance2 < bestDistance2) {
            best = sample;
            bestDistance2 = distance2;
        }
    }
    return best ? { sample: best, distance2: bestDistance2 } : null;
}

function projectSourceRay(
    stroke: ContourStroke,
    sample: { point: Vec2; index: number },
    sourceView: ContourView,
    currentViewProjMat: number[] | Float32Array,
): Vec2[] {
    const center = strokeWorldPointAtIndex(stroke, sourceView, sample.index, sample.point);
    const direction = sourceRayDirection(sourceView.viewProjInvMat, sample.point.x, sample.point.y);
    if (!center || !direction) return [];

    const tickLength = Math.max(0.04, Math.min(0.22, sourceView.radius * 0.14));
    const near = add3(center, scale3(direction, -tickLength));
    const far = add3(center, scale3(direction, tickLength));

    const a = projectPoint(currentViewProjMat, near);
    const b = projectPoint(currentViewProjMat, far);
    if (!isProjectedGuidePoint(a) || !isProjectedGuidePoint(b)) return [];

    const length = Math.hypot(a.x - b.x, a.y - b.y);
    if (length <= 0.025) return [];
    if (length <= 0.14) return [a, b];

    const scale = 0.14 / length;
    const screenCenter = {
        x: (a.x + b.x) * 0.5,
        y: (a.y + b.y) * 0.5,
    };
    const half = {
        x: (b.x - a.x) * scale * 0.5,
        y: (b.y - a.y) * scale * 0.5,
    };
    return [
        { x: screenCenter.x - half.x, y: screenCenter.y - half.y },
        { x: screenCenter.x + half.x, y: screenCenter.y + half.y },
    ];
}

function projectedDepthDirection(
    world: Vec3,
    direction: Vec3,
    currentViewProjMat: number[] | Float32Array,
): Vec2 | undefined {
    const tickLength = 0.08;
    const a = projectPoint(currentViewProjMat, add3(world, scale3(direction, -tickLength)));
    const b = projectPoint(currentViewProjMat, add3(world, scale3(direction, tickLength)));
    if (!isProjectedGuidePoint(a) || !isProjectedGuidePoint(b)) return undefined;

    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len = Math.hypot(dx, dy);
    if (!Number.isFinite(len) || len <= 1e-6) return undefined;
    return { x: dx / len, y: dy / len };
}

function samplePolyline(points: Vec2[], count: number): Vec2[] {
    return sampleIndexedPolyline(points, count).map(sample => sample.point);
}

function sampleIndexedPolyline(points: Vec2[], count: number): Array<{ point: Vec2; index: number }> {
    if (points.length <= count) return points.map((point, index) => ({ point, index }));
    const stride = Math.max(1, Math.ceil(points.length / count));
    const sampled = points
        .map((point, index) => ({ point, index }))
        .filter(sample => sample.index % stride === 0);
    const last = { point: points[points.length - 1], index: points.length - 1 };
    if (sampled[sampled.length - 1]?.index !== last.index) sampled.push(last);
    return sampled;
}

function simplifyProjectedPolyline(points: Vec2[]): Vec2[] {
    const out: Vec2[] = [];
    for (const point of points) {
        const last = out[out.length - 1];
        if (!last || Math.hypot(point.x - last.x, point.y - last.y) > 0.01) {
            out.push(point);
        }
    }
    return out;
}

export function strokeDepthForView(stroke: ContourStroke, view: ContourView | { viewProjMat: number[] | Float32Array }): number {
    return clampDepth(stroke.depthNdc ?? ndcDepthAtWorldOrigin(view.viewProjMat));
}

export function strokeDepthAtIndex(
    stroke: ContourStroke,
    view: ContourView | { viewProjMat: number[] | Float32Array },
    index: number,
): number {
    return clampDepth(stroke.depthSamplesNdc?.[index] ?? strokeDepthForView(stroke, view));
}

export function strokeDepthOffsetAtIndex(stroke: ContourStroke, index: number): number {
    return clampDepthOffset(stroke.depthSamplesOffset?.[index] ?? stroke.depthOffset ?? 0);
}

export function strokeWorldPointAtIndex(
    stroke: ContourStroke,
    view: ContourView | {
        viewProjMat: number[] | Float32Array;
        viewProjInvMat?: number[] | Float32Array;
    },
    index: number,
    point = stroke.resampledPoints[index],
): Vec3 | null {
    if (!point || !view.viewProjInvMat) return null;
    const hasOffset = stroke.depthSamplesOffset?.[index] !== undefined || stroke.depthOffset !== undefined;
    if (!hasOffset && stroke.depthSamplesNdc?.[index] !== undefined) {
        return unprojectNdc(view.viewProjInvMat, point.x, point.y, stroke.depthSamplesNdc[index]);
    }

    const base = unprojectNdc(
        view.viewProjInvMat,
        point.x,
        point.y,
        stroke.depthNdc ?? ndcDepthAtWorldOrigin(view.viewProjMat),
    );
    if (!base) return null;

    const offset = strokeDepthOffsetAtIndex(stroke, index);
    if (Math.abs(offset) <= 1e-7) return base;

    const direction = sourceRayDirection(view.viewProjInvMat, point.x, point.y);
    return direction ? add3(base, scale3(direction, offset)) : base;
}

export function ndcDepthAtWorldOrigin(viewProjMat: number[] | Float32Array): number {
    const clipZ = viewProjMat[14];
    const clipW = viewProjMat[15];
    if (!Number.isFinite(clipW) || Math.abs(clipW) <= 1e-6) return 0.5;
    return clampDepth(clipZ / clipW);
}

export function unprojectNdc(
    viewProjInvMat: number[] | Float32Array,
    x: number,
    y: number,
    z: number,
): Vec3 | null {
    const w = 1;
    const outX = viewProjInvMat[0] * x + viewProjInvMat[4] * y + viewProjInvMat[8] * z + viewProjInvMat[12] * w;
    const outY = viewProjInvMat[1] * x + viewProjInvMat[5] * y + viewProjInvMat[9] * z + viewProjInvMat[13] * w;
    const outZ = viewProjInvMat[2] * x + viewProjInvMat[6] * y + viewProjInvMat[10] * z + viewProjInvMat[14] * w;
    const outW = viewProjInvMat[3] * x + viewProjInvMat[7] * y + viewProjInvMat[11] * z + viewProjInvMat[15] * w;
    if (!Number.isFinite(outW) || Math.abs(outW) <= 1e-6) return null;
    return [outX / outW, outY / outW, outZ / outW];
}

function sourceRayDirection(
    viewProjInvMat: number[] | Float32Array,
    x: number,
    y: number,
): Vec3 | null {
    const near = unprojectNdc(viewProjInvMat, x, y, 0.02);
    const far = unprojectNdc(viewProjInvMat, x, y, 0.98);
    if (!near || !far) return null;
    return normalize3([far[0] - near[0], far[1] - near[1], far[2] - near[2]]);
}

function clampDepth(depth: number): number {
    return Math.max(0.02, Math.min(0.98, depth));
}

function clampDepthOffset(offset: number): number {
    return Math.max(-3, Math.min(3, offset));
}

function averageDepth(vertices: CrossViewGuideVertex[]): number | undefined {
    if (vertices.length === 0) return undefined;
    return vertices.reduce((sum, vertex) => sum + vertex.depthNdc, 0) / vertices.length;
}

function add3(a: Vec3, b: Vec3): Vec3 {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function scale3(v: Vec3, scale: number): Vec3 {
    return [v[0] * scale, v[1] * scale, v[2] * scale];
}

function normalize3(v: Vec3): Vec3 | null {
    const len = Math.hypot(v[0], v[1], v[2]);
    if (!Number.isFinite(len) || len <= 1e-8) return null;
    return [v[0] / len, v[1] / len, v[2] / len];
}

function isProjectedGuidePoint(point: Vec2 | null): point is Vec2 {
    return !!point
        && Number.isFinite(point.x)
        && Number.isFinite(point.y)
        && point.x >= -2
        && point.x <= 2
        && point.y >= -2
        && point.y <= 2;
}
