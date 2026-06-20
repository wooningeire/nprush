import { gridUv, isGridTriangleCovered } from "./chartPainting.ts";
import { add3, cross3, dot3, scale3, sub3 } from "./vectorMath.ts";
import type { PaintChart, Vec2, Vec3 } from "../types.ts";

export type ChartRaycastHit = {
    uv: Vec2,
    world: Vec3,
    t: number,
};

type ChartRaycastTriangle = {
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    uv0: Vec2,
    uv1: Vec2,
    uv2: Vec2,
};

export type ChartRaycastCache = {
    triangles: ChartRaycastTriangle[],
};

export const createChartRaycastCache = (
    chart: PaintChart,
    worldAt: (uv: Vec2) => Vec3 | null,
): ChartRaycastCache => {
    const uvs = new Array<Vec2>(chart.width * chart.height);
    const worlds = new Array<Vec3 | null>(chart.width * chart.height);

    for (let y = 0; y < chart.height; y++) {
        for (let x = 0; x < chart.width; x++) {
            const index = y * chart.width + x;
            const uv = gridUv(chart, x, y);
            uvs[index] = uv;
            worlds[index] = worldAt(uv);
        }
    }

    const triangles: ChartRaycastTriangle[] = [];
    for (let y = 1; y < chart.height; y++) {
        for (let x = 1; x < chart.width; x++) {
            const i00 = (y - 1) * chart.width + x - 1;
            const i10 = (y - 1) * chart.width + x;
            const i01 = y * chart.width + x - 1;
            const i11 = y * chart.width + x;

            if (isGridTriangleCovered(chart, i00, i10, i11)) {
                appendCachedTriangle(triangles, worlds, uvs, i00, i10, i11);
            }
            if (isGridTriangleCovered(chart, i00, i11, i01)) {
                appendCachedTriangle(triangles, worlds, uvs, i00, i11, i01);
            }
        }
    }

    return { triangles };
};

export const raycastCachedChart = (
    cache: ChartRaycastCache,
    ray: { origin: Vec3; direction: Vec3 },
): ChartRaycastHit[] => {
    const hits: ChartRaycastHit[] = [];
    for (const triangle of cache.triangles) {
        appendTriangleHit(hits, ray, triangle);
    }
    return hits.sort((a, b) => a.t - b.t);
};

export const raycastChart = (
    chart: PaintChart,
    ray: { origin: Vec3; direction: Vec3 },
    worldAt: (uv: Vec2) => Vec3 | null,
): ChartRaycastHit[] => raycastCachedChart(createChartRaycastCache(chart, worldAt), ray);

const appendCachedTriangle = (
    triangles: ChartRaycastTriangle[],
    worlds: Array<Vec3 | null>,
    uvs: Vec2[],
    i0: number,
    i1: number,
    i2: number,
) => {
    const p0 = worlds[i0];
    const p1 = worlds[i1];
    const p2 = worlds[i2];
    if (!p0 || !p1 || !p2) return;
    triangles.push({
        p0,
        p1,
        p2,
        uv0: uvs[i0],
        uv1: uvs[i1],
        uv2: uvs[i2],
    });
};

const appendTriangleHit = (
    hits: ChartRaycastHit[],
    ray: { origin: Vec3; direction: Vec3 },
    triangle: ChartRaycastTriangle,
) => {
    const hit = intersectRayTriangle(ray.origin, ray.direction, triangle.p0, triangle.p1, triangle.p2);
    if (!hit) return;
    const w0 = 1 - hit.u - hit.v;
    hits.push({
        t: hit.t,
        world: add3(ray.origin, scale3(ray.direction, hit.t)),
        uv: {
            x: triangle.uv0.x * w0 + triangle.uv1.x * hit.u + triangle.uv2.x * hit.v,
            y: triangle.uv0.y * w0 + triangle.uv1.y * hit.u + triangle.uv2.y * hit.v,
        },
    });
};

const intersectRayTriangle = (
    origin: Vec3,
    direction: Vec3,
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
): { t: number; u: number; v: number } | null => {
    const epsilon = 1e-7;
    const edge1 = sub3(p1, p0);
    const edge2 = sub3(p2, p0);
    const h = cross3(direction, edge2);
    const a = dot3(edge1, h);
    if (Math.abs(a) < epsilon) return null;
    const f = 1 / a;
    const s = sub3(origin, p0);
    const u = f * dot3(s, h);
    if (u < 0 || u > 1) return null;
    const q = cross3(s, edge1);
    const v = f * dot3(direction, q);
    if (v < 0 || u + v > 1) return null;
    const t = f * dot3(edge2, q);
    if (t <= epsilon) return null;
    return { t, u, v };
};