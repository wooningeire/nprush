import { MIN_DEPTH } from "./constants.ts";
import { add3, dot3, normalize3, scale3, sub3 } from "./vectorMath.ts";
import type { ProjectionSnapshot, Vec2, Vec3 } from "../types.ts";

export const viewPointToWorldAtDepth = (
    projection: ProjectionSnapshot,
    point: Vec2,
    depth: number,
): Vec3 | null => {
    const ray = makeViewRay(projection, point);
    if (!ray) return null;
    const denominator = dot3(ray.direction, viewForward(projection));
    if (Math.abs(denominator) <= 1e-6) return null;
    const distance = Math.max(MIN_DEPTH, depth) / denominator;
    if (!Number.isFinite(distance) || distance <= MIN_DEPTH) return null;
    return add3(ray.origin, scale3(ray.direction, distance));
};

export const viewDepthForDistanceAtPoint = (
    projection: ProjectionSnapshot,
    point: Vec2,
    distance: number,
): number => {
    const ray = makeViewRay(projection, point);
    if (!ray) return distance;
    return Math.max(MIN_DEPTH, distance * dot3(ray.direction, viewForward(projection)));
};

export const viewDepthForWorldPoint = (projection: ProjectionSnapshot, point: Vec3): number => {
    return dot3(sub3(point, cameraCenter(projection)), viewForward(projection));
};

export const defaultDepthForProjection = (projection: ProjectionSnapshot): number => {
    return Math.max(MIN_DEPTH, viewDepthForWorldPoint(projection, [0, 0, 0]));
};

export const makeViewRay = (projection: ProjectionSnapshot, point: Vec2): { origin: Vec3; direction: Vec3 } | null => {
    const near = unprojectNdc(projection.viewProjInvMat, point.x, point.y, -1);
    const far = unprojectNdc(projection.viewProjInvMat, point.x, point.y, 1);
    if (!near || !far) return null;
    return {
        origin: cameraCenter(projection),
        direction: normalize3(sub3(far, near), [0, 0, -1]),
    };
};

export const cameraCenter = (projection: ProjectionSnapshot): Vec3 => {
    return [projection.viewInvMat[12], projection.viewInvMat[13], projection.viewInvMat[14]];
};

export const viewForward = (projection: ProjectionSnapshot): Vec3 => {
    return makeViewRay(projection, { x: 0, y: 0 })?.direction ?? [0, 0, -1];
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
