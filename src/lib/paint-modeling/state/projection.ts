import { sampleChartDepth } from "./chartPainting.ts";
import { MIN_DEPTH } from "./constants.ts";
import { add3, dot3, normalize3, scale3, sub3 } from "./vectorMath.ts";
import type { PaintChart, PaintView, Vec2, Vec3 } from "../types.ts";

export const chartPointToWorldFromView = (chart: PaintChart, view: PaintView, uv: Vec2): Vec3 | null => {
    return viewPointToWorldAtDepth(view, uv, sampleChartDepth(chart, uv));
};

export const viewPointToWorldAtDepth = (
    view: PaintView,
    point: Vec2,
    depth: number,
): Vec3 | null => {
    const ray = makeViewRay(view, point);
    if (!ray) return null;
    const denominator = dot3(ray.direction, viewForward(view));
    if (Math.abs(denominator) <= 1e-6) return null;
    const distance = Math.max(MIN_DEPTH, depth) / denominator;
    if (!Number.isFinite(distance) || distance <= MIN_DEPTH) return null;
    return add3(ray.origin, scale3(ray.direction, distance));
};

export const viewDepthForDistanceAtPoint = (
    view: PaintView,
    point: Vec2,
    distance: number,
): number => {
    const ray = makeViewRay(view, point);
    if (!ray) return distance;
    return Math.max(MIN_DEPTH, distance * dot3(ray.direction, viewForward(view)));
};

export const viewDepthForWorldPoint = (view: PaintView, point: Vec3): number => {
    return dot3(sub3(point, cameraCenter(view)), viewForward(view));
};

export const makeViewRay = (view: PaintView, point: Vec2): { origin: Vec3; direction: Vec3 } | null => {
    const near = unprojectNdc(view.viewProjInvMat, point.x, point.y, -1);
    const far = unprojectNdc(view.viewProjInvMat, point.x, point.y, 1);
    if (!near || !far) return null;
    return {
        origin: cameraCenter(view),
        direction: normalize3(sub3(far, near), [0, 0, -1]),
    };
};

export const cameraCenter = (view: PaintView): Vec3 => {
    return [view.viewInvMat[12], view.viewInvMat[13], view.viewInvMat[14]];
};

export const viewForward = (view: PaintView): Vec3 => {
    return makeViewRay(view, { x: 0, y: 0 })?.direction ?? [0, 0, -1];
};

export const projectVisiblePoint = (viewProjMat: number[] | Float32Array, p: Vec3): Vec2 | null => {
    const x = viewProjMat[0] * p[0] + viewProjMat[4] * p[1] + viewProjMat[8] * p[2] + viewProjMat[12];
    const y = viewProjMat[1] * p[0] + viewProjMat[5] * p[1] + viewProjMat[9] * p[2] + viewProjMat[13];
    const z = viewProjMat[2] * p[0] + viewProjMat[6] * p[1] + viewProjMat[10] * p[2] + viewProjMat[14];
    const w = viewProjMat[3] * p[0] + viewProjMat[7] * p[1] + viewProjMat[11] * p[2] + viewProjMat[15];
    if (Math.abs(w) <= 1e-8) return null;
    const ndcZ = z / w;
    if (ndcZ < -1 || ndcZ > 1) return null;
    return { x: x / w, y: y / w };
};

const unprojectNdc = (invViewProjMat: number[], x: number, y: number, z: number): Vec3 | null => {
    const px = invViewProjMat[0] * x + invViewProjMat[4] * y + invViewProjMat[8] * z + invViewProjMat[12];
    const py = invViewProjMat[1] * x + invViewProjMat[5] * y + invViewProjMat[9] * z + invViewProjMat[13];
    const pz = invViewProjMat[2] * x + invViewProjMat[6] * y + invViewProjMat[10] * z + invViewProjMat[14];
    const pw = invViewProjMat[3] * x + invViewProjMat[7] * y + invViewProjMat[11] * z + invViewProjMat[15];
    if (Math.abs(pw) <= 1e-8) return null;
    return [px / pw, py / pw, pz / pw];
};