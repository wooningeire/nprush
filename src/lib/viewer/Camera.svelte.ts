import { mat4, type Mat4 } from "wgpu-matrix";

export interface CameraControlScheme {
    viewMat(): Mat4;
    viewInvMat(): Mat4;
}

export interface CameraScreenDims {
    width(): number,
    height(): number,
}

export class Camera {
    private readonly controlScheme: CameraControlScheme;
    readonly screenDims: CameraScreenDims;

    zNear = $state(0.01);
    zFar = $state(100);
    fov = $state(Math.PI / 2);

    // Fixed 1:1 aspect so the projection is independent of the browser window size.
    // Both the optim textures (OPTIM_SHORT × OPTIM_SHORT) and the full-res display
    // textures use this same projection, keeping optimization stable across resizes.
    readonly aspect = 1.0;

    readonly proj = $derived.by(() => mat4.perspective(this.fov, this.aspect, this.zNear, this.zFar));
    readonly viewMat = $derived.by(() => this.controlScheme.viewMat());
    readonly viewInvMat = $derived.by(() => this.controlScheme.viewInvMat());
    readonly viewProjMat = $derived.by(() => mat4.mul(this.proj, this.viewMat));
    readonly viewProjInvMat = $derived.by(() => mat4.inverse(this.viewProjMat));

    constructor({
        controlScheme,
        screenDims,
    }: {
        controlScheme: CameraControlScheme,
        screenDims: CameraScreenDims,
    }) {
        this.controlScheme = controlScheme;
        this.screenDims = screenDims;
    }
}