import { onDestroy, onMount } from "svelte";
import { Camera } from "./Camera.svelte.ts";
import { CameraOrbit } from "./CameraOrbit.svelte.ts";
import { requestGpu } from "$/gpu/requestGpu";
import { GpuRunner } from "./GpuRunner.svelte.ts";
import { GPU_CONSTANTS } from "$/gpu/constants";
import { loadGlb } from "$/gpu/loadGlb";
import artelorianUrl from "$/assets/artelorian.glb?url";
import groundUrl from "$/assets/ground.glb?url";
import hdrUrl from "$/assets/lakeside_sunrise_2k.hdr?url";
import brushUrl from "$/assets/brush.png?url";
import groundAlbedoUrl from "$/assets/brown_mud_03_diff_2k.jpg?url";
import groundNormalUrl from "$/assets/brown_mud_03_nor_gl_2k.png?url";
import { loadHdrTexture } from "$/gpu/loadHdrTexture";
import { loadTexture } from "$/gpu/loadTexture";
import { buildBvh, raycastBvh, type BvhResult } from "$/gpu/bvh";
import { vec4 } from "wgpu-matrix";
import { STRIP_HEIGHT_FRAC } from "$/util";

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
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `nprush-render-${Date.now()}.png`;
            a.click();
            URL.revokeObjectURL(url);
        } catch (e) {
            console.error("Failed to take screenshot", e);
        } finally {
            this.isCapturing = false;
        }
    }

    // --- Turntable Animation ---
    turntableFrameCount = $state(120);
    turntableStepsPerFrame = $state(50);
    turntableProgress = $state(0);
    isTurntableRendering = $state(false);
    turntableCanceled = false;

    cancelTurntable() {
        this.turntableCanceled = true;
    }

    /**
     * Render a turntable animation:
     * 1. Save current camera state
     * 2. For each frame, set camera longitude, run N training steps, capture composited frame
     * 3. Encode all frames into a WebM video via MediaRecorder
     * 4. Download the video
     * 5. Restore camera state
     */
    async renderTurntable() {
        if (!this.runner || this.isTurntableRendering) return;

        this.isTurntableRendering = true;
        this.turntableCanceled = false;
        this.turntableProgress = 0;

        // Save original camera state
        const origLong = this.orbit.long;
        const origLat = this.orbit.lat;
        const origRadius = this.orbit.radius;

        const totalFrames = this.turntableFrameCount;
        const stepsPerFrame = this.turntableStepsPerFrame;

        // We'll collect frames first, then encode
        const frames: ImageData[] = [];

        try {
            for (let frame = 0; frame < totalFrames; frame++) {
                if (this.turntableCanceled) break;

                // Set camera to the turntable angle for this frame
                const t = frame / totalFrames;
                this.orbit.long = origLong + t * Math.PI * 2;

                // Run N training steps at this camera angle by waiting for rAF ticks.
                // Each rAF tick runs one optimization step in the loop.
                for (let step = 0; step < stepsPerFrame; step++) {
                    if (this.turntableCanceled) break;
                    await new Promise<void>(r => requestAnimationFrame(() => r()));
                }

                if (this.turntableCanceled) break;

                // Capture the composited frame
                const imageData = await this.runner.captureTurntableFrame();
                frames.push(imageData);
                this.turntableProgress = (frame + 1) / totalFrames;
            }

            if (!this.turntableCanceled && frames.length > 0) {
                // Encode frames into a WebM video using canvas + MediaRecorder
                await this.encodeFramesToVideo(frames);
            }
        } catch (e) {
            console.error("Turntable render failed", e);
        } finally {
            // Restore camera
            this.orbit.long = origLong;
            this.orbit.lat = origLat;
            this.orbit.radius = origRadius;
            this.isTurntableRendering = false;
            this.turntableProgress = 0;
        }
    }

    private async encodeFramesToVideo(frames: ImageData[]) {
        if (frames.length === 0) return;

        const w = frames[0].width;
        const h = frames[0].height;
        const fps = 30;
        const frameDuration = 1000 / fps;

        const canvas = document.createElement("canvas");
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d")!;

        const stream = canvas.captureStream(fps); 
        
        const mimeType = [
            "video/webm;codecs=vp9",
            "video/webm;codecs=vp8",
            "video/webm",
            "video/mp4",
        ].find(t => MediaRecorder.isTypeSupported(t)) || "video/webm";

        const recorder = new MediaRecorder(stream, {
            mimeType,
            videoBitsPerSecond: 8_000_000,
        });

        const chunks: Blob[] = [];
        recorder.ondataavailable = (e) => {
            if (e.data.size > 0) chunks.push(e.data);
        };

        const done = new Promise<void>((resolve, reject) => {
            recorder.onstop = () => resolve();
            recorder.onerror = (e) => reject(e);
        });

        recorder.start();
        
        // Let the recorder "warm up"
        await new Promise(r => setTimeout(r, 100));

        for (const frame of frames) {
            ctx.putImageData(frame, 0, 0);
            // With fixed FPS captureStream, the recorder pulls from the canvas automatically.
            // We just need to wait long enough for it to see the frame.
            await new Promise(r => setTimeout(r, frameDuration));
        }

        // Small trailing buffer
        await new Promise(r => setTimeout(r, 100));

        recorder.stop();
        await done;

        const blob = new Blob(chunks, { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        const extension = mimeType.includes("mp4") ? "mp4" : "webm";
        a.download = `nprush-turntable-${Date.now()}.${extension}`;
        a.click();
        URL.revokeObjectURL(url);
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