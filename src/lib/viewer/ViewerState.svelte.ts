import { onDestroy, onMount } from "svelte";
import { Camera } from "./Camera.svelte.ts";
import { CameraOrbit } from "./CameraOrbit.svelte.ts";
import { requestGpu } from "$/gpu/requestGpu";
import { GpuRunner } from "./GpuRunner.svelte.ts";
import { GPU_CONSTANTS } from "$/gpu/constants";
import { loadGlb } from "../gpu/io/loadGlb.ts";
import artelorianUrl from "$/assets/artelorian.glb?url";
import groundUrl from "$/assets/ground.glb?url";
import hdrUrl from "$/assets/lakeside_sunrise_2k.hdr?url";
import brushUrl from "$/assets/brush.png?url";
import groundAlbedoUrl from "$/assets/brown_mud_03_diff_2k.jpg?url";
import groundNormalUrl from "$/assets/brown_mud_03_nor_gl_2k.png?url";
import { loadHdrTexture } from "../gpu/io/loadHdrTexture.ts";
import { loadTexture } from "../gpu/io/loadTexture.ts";
import { buildBvh, raycastBvh, type BvhResult } from "../gpu/bvh.ts";
import { vec4 } from "wgpu-matrix";
import { STRIP_HEIGHT_FRAC } from "$/util";
import { downloadBlob, encodeFramesToVideo } from "../util/export.ts";

export class ViewerState {
    width = $state(300);
    height = $state(150);
    edgeBeziersEnabled = $state(false);
    baseColorBeziersEnabled = $state(true);
    colorBeziersEnabled = $state(true);
    splatTrainingPaused = $state(false);
    edgeBezierTrainingPaused = $state(false);
    baseColorBezierTrainingPaused = $state(false);
    colorBezierTrainingPaused = $state(false);
    compareBlurred = $state(true);
    blurRadius = $state(16);
    meshSplatsEnabled = $state(true);
    
    meshVerts: Float32Array | null = null;
    meshBvh: BvhResult | null = null;
    isCapturing = $state(false);
    
    runner = $state<GpuRunner | null>(null);
    
    readonly orbit = new CameraOrbit();
    readonly camera = new Camera({
        controlScheme: this.orbit,
        screenDims: { width: () => this.width, height: () => this.height },
    });

    onPaintDrag(x: number, y: number) {
        if (!this.meshBvh || !this.meshVerts || !this.runner || !this.meshSplatsEnabled) {
            return;
        }
        
        const targetWidth = this.width / 2;
        const targetHeight = this.height * (1 - STRIP_HEIGHT_FRAC);
        
        // Ignore clicks in the bottom strip
        if (y > targetHeight) return;
        
        // Map x to either the left or right half
        const localX = x % targetWidth;
        
        const ndcX = (localX / targetWidth) * 2 - 1;
        const ndcY = -((y / targetHeight) * 2 - 1);
        
        const originNdC = [ndcX, ndcY, 0, 1];
        const targetNdC = [ndcX, ndcY, 1, 1];
        
        const originWorldW = vec4.transformMat4(originNdC, this.camera.viewProjInvMat);
        const targetWorldW = vec4.transformMat4(targetNdC, this.camera.viewProjInvMat);
        
        const origin = [originWorldW[0]/originWorldW[3], originWorldW[1]/originWorldW[3], originWorldW[2]/originWorldW[3]] as [number, number, number];
        const target = [targetWorldW[0]/targetWorldW[3], targetWorldW[1]/targetWorldW[3], targetWorldW[2]/targetWorldW[3]] as [number, number, number];
        
        const dir = [target[0]-origin[0], target[1]-origin[1], target[2]-origin[2]] as [number, number, number];
        const len = Math.sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
        dir[0] /= len; dir[1] /= len; dir[2] /= len;
        
        const hit = raycastBvh(this.meshBvh, this.meshVerts, origin, dir);
        if (hit) {
            // Paint a splat with a random color and size
            const radius = 0.02 + Math.random() * 0.08;
            const color = [Math.random(), Math.random(), Math.random()] as [number, number, number];
            this.runner.meshRenderPipelineManager.addSplat(hit.p, radius, color);
        }
    }

    async takeScreenshot() {
        if (!this.runner || this.isCapturing) return;
        this.isCapturing = true;
        try {
            const blob = await this.runner.takeScreenshot();
            downloadBlob(blob, `nprush-render-${Date.now()}.png`);
        } catch (e) {
            console.error("Failed to take screenshot", e);
        } finally {
            this.isCapturing = false;
        }
    }

    // --- Turntable Animation ---
    turntableFrameCount = $state(120);
    turntableStepsPerFrame = $state(10);
    turntableProgress = $state(0);
    isTurntableRendering = $state(false);
    turntableCanceled = false;

    // Multi-view training: when enabled, each frame trains from a random
    // camera angle sampled from the turntable animation path.
    turntableTraining = $state(false);

    // Time-varying path parameters.
    // t ∈ [0, 1] → full revolution.
    // long(t) = baseLong + t * 2π
    // lat(t)  = latCenter + latAmplitude * sin(t * 2π * latCycles)
    // radius(t) = radiusCenter + radiusAmplitude * sin(t * 2π * radiusCycles)
    turntableLatCenter = $state(Math.PI * 1 / 4);
    turntableLatAmplitude = $state(0);
    turntableLatCycles = $state(1);
    turntableRadiusCenter = $state(1);
    turntableRadiusAmplitude = $state(0);
    turntableRadiusCycles = $state(2);

    /**
     * Evaluate the turntable animation path at a normalized time t ∈ [0, 1].
     */
    evaluatePath(t: number, baseLong: number): { long: number; lat: number; radius: number } {
        const TWO_PI = Math.PI * 2;
        return {
            long: baseLong + t * TWO_PI,
            lat: this.turntableLatCenter + this.turntableLatAmplitude * Math.sin(t * TWO_PI * this.turntableLatCycles),
            radius: this.turntableRadiusCenter + this.turntableRadiusAmplitude * Math.sin(t * TWO_PI * this.turntableRadiusCycles),
        };
    }

    /** Saved longitude origin for multi-view training. */
    private turntableBaseLong = 0;

    /**
     * Toggle multi-view turntable training. When enabled, each training
     * frame will use a randomly sampled camera angle from the animation path.
     */
    toggleTurntableTraining() {
        this.turntableTraining = !this.turntableTraining;
        if (this.turntableTraining) {
            // Snapshot current camera as the base reference
            this.turntableBaseLong = this.orbit.long;
            this.turntableLatCenter = this.orbit.lat;
            this.turntableRadiusCenter = this.orbit.radius;
        }
    }

    /**
     * Called each frame by the render loop while turntableTraining is enabled.
     * Sets the camera to a random view from the turntable path so the
     * optimizer sees all angles.
     */
    tickTurntableTraining() {
        if (!this.turntableTraining) return;
        const t = Math.random();
        const p = this.evaluatePath(t, this.turntableBaseLong);
        this.orbit.long = p.long;
        this.orbit.lat = p.lat;
        this.orbit.radius = p.radius;
    }

    cancelTurntable() {
        this.turntableCanceled = true;
    }

    /**
     * Render the turntable animation:
     * 1. Deterministically sweep through the animation path
     * 2. At each frame position, run a few training steps
     * 3. Capture the composited frame
     * 4. Encode all frames into a video and download
     */
    async renderTurntable() {
        if (!this.runner || this.isTurntableRendering) return;

        this.isTurntableRendering = true;
        this.turntableCanceled = false;
        this.turntableProgress = 0;

        const origLong = this.orbit.long;
        const origLat = this.orbit.lat;
        const origRadius = this.orbit.radius;

        const baseLong = this.turntableTraining ? this.turntableBaseLong : origLong;

        const totalFrames = this.turntableFrameCount;
        const stepsPerFrame = this.turntableStepsPerFrame;

        const frames: ImageData[] = [];

        try {
            for (let frame = 0; frame < totalFrames; frame++) {
                if (this.turntableCanceled) break;

                const t = frame / totalFrames;
                const p = this.evaluatePath(t, baseLong);
                this.orbit.long = p.long;
                this.orbit.lat = p.lat;
                this.orbit.radius = p.radius;

                for (let step = 0; step < stepsPerFrame; step++) {
                    if (this.turntableCanceled) break;
                    await new Promise<void>(r => requestAnimationFrame(() => r()));
                }

                if (this.turntableCanceled) break;

                const imageData = await this.runner.captureTurntableFrame();
                frames.push(imageData);
                this.turntableProgress = (frame + 1) / totalFrames;
            }

            if (!this.turntableCanceled && frames.length > 0) {
                await this.encodeFramesToVideo(frames);
            }
        } catch (e) {
            console.error("Turntable render failed", e);
        } finally {
            this.orbit.long = origLong;
            this.orbit.lat = origLat;
            this.orbit.radius = origRadius;
            this.isTurntableRendering = false;
            this.turntableProgress = 0;
        }
    }

    private async encodeFramesToVideo(frames: ImageData[]) {
        if (frames.length === 0) return;

        try {
            const { blob, mimeType } = await encodeFramesToVideo(frames, 30);
            const extension = mimeType.includes("mp4") ? "mp4" : "webm";
            downloadBlob(blob, `nprush-turntable-${Date.now()}.${extension}`);
        } catch (e) {
            console.error("Failed to encode video", e);
        }
    }

    static mount({
        canvasPromise,
        numSplats = GPU_CONSTANTS.NUM_GAUSSIAN_SPLATS,
    }: {
        canvasPromise: Promise<HTMLCanvasElement>,
        numSplats?: number,
    }) {
        const state = new ViewerState();
        
        let stopLoop: (() => void) | null = null;
        onMount(async () => {
            // Kick off mesh load and gpu request concurrently; both are awaited
            // before we build the runner since the mesh is a constructor input.
            const [gpu, mesh, groundMesh, groundPbrMesh] = await Promise.all([
                requestGpu({ canvas: await canvasPromise }),
                loadGlb(artelorianUrl),
                loadGlb(groundUrl, false, [1, 1, 1, 0]),           // Plane — specular mirror, world-space
                loadGlb(groundUrl, false, [1, 1, 1, 1], 'Plane.001'), // Plane.001 — PBR textured
            ]);
            if (!gpu) return;

            state.meshVerts = new Float32Array(mesh.vertices);
            state.meshBvh = buildBvh(state.meshVerts, new Uint32Array(mesh.indices));

            const [envTexture, brushTexture, groundAlbedoTexture, groundNormalTexture] = await Promise.all([
                loadHdrTexture(gpu.device, hdrUrl),
                loadTexture(gpu.device, brushUrl),
                loadTexture(gpu.device, groundAlbedoUrl),
                loadTexture(gpu.device, groundNormalUrl),
            ]);

            const gpuRunner = new GpuRunner({
                device: gpu.device,
                context: gpu.context,
                format: gpu.format,
                camera: state.camera,
                viewerState: state,
                mesh,
                groundMesh,
                groundPbrMesh,
                matcapTexture: envTexture,
                brushTexture,
                groundAlbedoTexture,
                groundNormalTexture,
                numSplats,
            });
            state.runner = gpuRunner;

            stopLoop = gpuRunner.loop();
        });

        onDestroy(() => {
            stopLoop?.();
            state.runner?.destroy();
        });

        return state;
    }
}