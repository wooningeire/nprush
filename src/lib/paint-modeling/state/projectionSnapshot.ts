import type { ProjectionSnapshot } from "../types.ts";

type ProjectionCameraSource = {
    viewProjMat: ArrayLike<number>,
    viewProjInvMat: ArrayLike<number>,
    viewInvMat: ArrayLike<number>,
};

export const captureProjectionSnapshot = (
    width: number,
    height: number,
    camera: ProjectionCameraSource,
): ProjectionSnapshot => ({
    width: Math.max(1, width),
    height: Math.max(1, height),
    // GPU placement completes asynchronously, so a stroke must not reread a moved live camera.
    viewProjMat: Array.from(camera.viewProjMat),
    viewProjInvMat: Array.from(camera.viewProjInvMat),
    viewInvMat: Array.from(camera.viewInvMat),
});
