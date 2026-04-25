import { mat4, type Mat4 } from "wgpu-matrix";
import { STRIP_HEIGHT_FRAC } from "$/util";

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

    // Splats are displayed in the right half of the canvas, above the debug strip.
    // The projection aspect must match this visible region so the sphere rendered into
    // the optim texture has matching pixel-width/height (i.e. is actually circular).
    readonly aspect = $derived.by(
        () => (this.screenDims.width() / 2) / (this.screenDims.height() * (1 - STRIP_HEIGHT_FRAC))
    );

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