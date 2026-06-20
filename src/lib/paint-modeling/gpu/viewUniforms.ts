import type { PaintView, Vec3 } from "../types.ts";

export const cameraCenter = (view: PaintView): Vec3 => [
    view.viewInvMat[12],
    view.viewInvMat[13],
    view.viewInvMat[14],
];

export const viewForward = (view: PaintView): Vec3 => {
    const near = unprojectNdc(view.viewProjInvMat, 0, 0, -1);
    const far = unprojectNdc(view.viewProjInvMat, 0, 0, 1);
    return normalize3(sub3(far, near));
};

const unprojectNdc = (matrix: number[], x: number, y: number, z: number): Vec3 => {
    const px = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12];
    const py = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13];
    const pz = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14];
    const pw = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15];
    return [px / pw, py / pw, pz / pw];
};

const sub3 = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

const normalize3 = (v: Vec3): Vec3 => {
    const length = Math.hypot(v[0], v[1], v[2]);
    if (length <= 1e-8) return [0, 0, -1];
    return [v[0] / length, v[1] / length, v[2] / length];
};