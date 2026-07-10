import type { ConstructionPlane, Vec3 } from "../types.ts";
import { MIN_DEPTH } from "./constants.ts";
import {
    add3,
    dot3,
    normalize3,
    scale3,
    sub3,
} from "./vectorMath.ts";

export const createDefaultConstructionPlane = (): ConstructionPlane => ({
    origin: [0, 0, 0],
    normal: [0, 0, 1],
});

export const cameraOrigin = (viewInvMat: ArrayLike<number>): Vec3 => [
    viewInvMat[12] ?? 0,
    viewInvMat[13] ?? 0,
    viewInvMat[14] ?? 0,
];

export const viewForward = (viewInvMat: ArrayLike<number>): Vec3 => normalize3([
    -(viewInvMat[8] ?? 0),
    -(viewInvMat[9] ?? 0),
    -(viewInvMat[10] ?? 1),
], [0, 0, -1]);

export const viewDepthForPoint = (
    viewInvMat: ArrayLike<number>,
    point: Vec3,
): number => dot3(
    sub3(point, cameraOrigin(viewInvMat)),
    viewForward(viewInvMat),
);

export const movePointToViewDepth = (
    viewInvMat: ArrayLike<number>,
    point: Vec3,
    depth: number,
): Vec3 => {
    const finiteDepth = Number.isFinite(depth) ? depth : MIN_DEPTH;
    const targetDepth = Math.max(MIN_DEPTH, finiteDepth);
    const currentDepth = viewDepthForPoint(viewInvMat, point);
    return add3(point, scale3(viewForward(viewInvMat), targetDepth - currentDepth));
};

export const viewFacingNormal = (viewInvMat: ArrayLike<number>): Vec3 => (
    scale3(viewForward(viewInvMat), -1)
);

export const normalizedPlane = (
    origin: Vec3,
    normal: Vec3,
    fallbackNormal: Vec3 = [0, 0, 1],
): ConstructionPlane => ({
    origin: [...origin],
    normal: normalize3(normal, fallbackNormal),
});