import { onDestroy, onMount } from "svelte";
import { Camera } from "./Camera.svelte.ts";
import { CameraOrbit } from "./CameraOrbit.svelte.ts";
import { requestGpu } from "$/gpu/requestGpu";
import { GpuRunner } from "./GpuRunner.svelte.ts";
import { constants } from "$/gpu/constants";
import { loadGlb, parseGlbBuffer } from "../gpu/file-load/loadGlb.ts";
import artelorianUrl from "$/assets/artelorian.glb?url";
import groundUrl from "$/assets/ground.glb?url";
import hdrUrl from "$/assets/lakeside_sunrise_2k.hdr?url";
import brushUrl from "$/assets/brush.png?url";
import groundAlbedoUrl from "$/assets/brown_mud_03_diff_2k.jpg?url";
import groundNormalUrl from "$/assets/brown_mud_03_nor_gl_2k.png?url";
import { loadHdrTexture } from "../gpu/file-load/loadHdrTexture.ts";
import { loadTexture } from "../gpu/file-load/loadTexture.ts";
import { buildBvh, raycastBvh, type BvhResult } from "../gpu/bvh.ts";
import { vec4 } from "wgpu-matrix";
import { STRIP_HEIGHT_FRAC } from "$/util";
import { downloadBlob, openFrameWriter } from "../util/export.ts";
import {
    RENDER_MODE_MULTIVIEW,
    RENDER_MODE_SINGLE_VIEW_REALTIME,
    type RenderMode,
} from "./renderMode.ts";
import { evaluateTurntablePath, type TurntablePathParams } from "./turntable/turntablePath.ts";
import { runTurntableExport } from "./turntable/turntableExport.ts";
import { GPU_PROFILER_PAIR_COUNT, GPU_PROFILER_HISTORY_FRAMES } from "$/gpu/performanceMeasurement/gpuProfilerPairs";
import { showToast, dismissToast } from "./toast.svelte.ts";

export class ViewerState {
    width = $state(300);
    height = $state(150);

    // Output render resolution — independent of the browser window size.
    // These drive the full-res textures used for visualization and capture.
    renderWidth = $state(256);
    renderHeight = $state(256);

    edgeBeziersEnabled = $state(false);
    baseColorBeziersEnabled = $state(true);
    colorBeziersEnabled = $state(true);
    splatsEnabled = $state(true);
    splatTrainingPaused = $state(false);
    edgeBezierTrainingPaused = $state(false);
    baseColorBezierTrainingPaused = $state(false);
    colorBezierTrainingPaused = $state(false);
    compareBlurred = $state(true);
    blurRadius = $state(16);
    meshSplatsEnabled = $state(false);
    
    meshVerts: Float32Array | null = null;
    meshBvh: BvhResult | null = null;
    isCapturing = $state(false);
    
    renderMode = $state<RenderMode>(RENDER_MODE_SINGLE_VIEW_REALTIME);
    viewportRenderingFrozen = $state(false);

    gpuTimestampQuerySupported = $state(false);
    gpuProfilingEnabled = $state(false);
    gpuProfilingMs = $state<(number | null)[]>(Array(GPU_PROFILER_PAIR_COUNT).fill(null));
    gpuProfilingHistoryFrames = $state<(number | null)[][]>([]);

    setGpuProfilingFrameMs(msPerPair: readonly (number | null)[]) {
        this.gpuProfilingMs = [...msPerPair];
        const next = [...this.gpuProfilingHistoryFrames, [...msPerPair]];
        const cap = GPU_PROFILER_HISTORY_FRAMES;
        this.gpuProfilingHistoryFrames = next.length > cap ? next.slice(-cap) : next;
    }

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
            showToast(`Screenshot failed: ${(e as Error)?.message ?? e}`, "error");
        } finally {
            this.isCapturing = false;
        }
    }

    // --- Turntable Animation ---
    turntableFrameCount = $state(120);
    /** rAF ticks to wait before each PNG capture during turntable export (not GPU work). */
    turntableStepsPerFrame = $state(10);
    /**
     * Hold the sampled prerendered dataset view / camera across this many browser
     * frames before picking another slot at random. (PT samples/view is unrelated — that is prerender convergence only.)
     */
    multiviewDisplayFramesPerView = $state(1);
    turntableProgress = $state(0);
    isTurntableRendering = $state(false);
    turntableCanceled = false;

    // Multi-view training: trains from fixed prerendered views; GpuRunner lingers each
    // sampled slot for {@link multiviewDisplayFramesPerView} display frames unless set to 1.
    turntableTraining = $state(false);

    // Minimum path-tracer samples accumulated when building each prerendered dataset slot.
    turntableMinSamplesPerView = $state(32);

    /** Frames accumulated on the current turntable training view. */
    private turntableViewFrames = 0;

    // --- Prerendered multiview dataset ---
    // Number of views to prerender into the dataset.
    multiviewNumViews = $state(32);
    // True while the prerender pass is running.
    multiviewPrerendering = $state(false);
    multiviewPrerenderProgress = $state(0);
    // True once the dataset is ready; training uses it instead of live PT.
    multiviewDatasetReady = $state(false);

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

    /** Saved longitude origin for multi-view training / turntable export while training. */
    private _turntableBaseLong = 0;

    get turntableBaseLong(): number {
        return this._turntableBaseLong;
    }

    getTurntablePathParams(): TurntablePathParams {
        return {
            latCenter: this.turntableLatCenter,
            latAmplitude: this.turntableLatAmplitude,
            latCycles: this.turntableLatCycles,
            radiusCenter: this.turntableRadiusCenter,
            radiusAmplitude: this.turntableRadiusAmplitude,
            radiusCycles: this.turntableRadiusCycles,
        };
    }

    setRenderMode(mode: RenderMode) {
        if (this.renderMode === mode) return;
        if (mode === RENDER_MODE_SINGLE_VIEW_REALTIME && this.turntableTraining) {
            this.turntableTraining = false;
        }
        this.renderMode = mode;
    }

    /**
     * Toggle multi-view turntable training. When enabled, kicks off a
     * prerender pass to build the dataset, then trains from it.
     */
    toggleTurntableTraining() {
        if (this.turntableTraining) {
            // Stop training
            this.turntableTraining = false;
            return;
        }
        // Snapshot current camera as the base reference
        this._turntableBaseLong = this.orbit.long;
        this.turntableLatCenter = this.orbit.lat;
        this.turntableRadiusCenter = this.orbit.radius;
        this.turntableViewFrames = 0;
        this.multiviewDatasetReady = false;
        this.turntableTraining = true;
        // Kick off prerender — runner will detect turntableTraining=true and
        // multiviewDatasetReady=false and run the prerender pass.
        if (this.runner) {
            this.runner.prerenderDataset().catch(e => {
                console.error("Prerender failed", e);
                showToast(`Prerender failed: ${(e as Error)?.message ?? e}`, "error");
            });
        }
    }

    /**
     * Called each frame by the render loop while in Multiview mode
     * AND the dataset is ready. No-op — GpuRunner samples a dataset slot and may
     * hold it for {@link multiviewDisplayFramesPerView} frames.
     */
    tickAnimationMode() {
        // Dataset-driven: GpuRunner selects a dataset slot (held for displayFramesPerView)
        // and writes its matrices directly to GPU buffers, without reactive orbit drift.
    }

    cancelTurntable() {
        this.turntableCanceled = true;
    }

    /**
     * Render the turntable animation:
     * 1. Ask the user to pick an output folder (File System Access API).
     * 2. Deterministically sweep through the animation path.
     * 3. At each frame position, run a few render steps then capture the
     *    composited frame and write it directly to the folder as a PNG.
     *
     * Frames are written one-by-one so memory usage stays flat regardless
     * of frame count, and there is no dependency on the unreliable
     * MediaRecorder / captureStream video encoding path.
     */
    async renderTurntable() {
        if (!this.runner || this.isTurntableRendering) return;

        // Ask for the output folder before locking the UI — if the user
        // cancels the picker we bail out cleanly with no state change.
        const writer = await openFrameWriter();
        if (!writer) return;

        this.isTurntableRendering = true;
        this.turntableCanceled = false;
        this.turntableProgress = 0;

        const origLong = this.orbit.long;
        const origLat = this.orbit.lat;
        const origRadius = this.orbit.radius;

        const baseLong = this.turntableTraining ? this.turntableBaseLong : origLong;
        const turntableParams = this.getTurntablePathParams();

        try {
            await runTurntableExport({
                totalFrames: this.turntableFrameCount,
                stepsPerFrame: this.renderMode === RENDER_MODE_SINGLE_VIEW_REALTIME ? this.turntableStepsPerFrame : 1,
                isCanceled: () => this.turntableCanceled,
                captureFrame: () => this.runner!.captureTurntableFrame(),
                writer,
                orbit: this.orbit,
                restoreOrbit: { long: origLong, lat: origLat, radius: origRadius },
                evalAtT: t => {
                    const p = evaluateTurntablePath(t, baseLong, turntableParams);
                    // Orbit.lat can differ from turntableLatCenter while multiview uses the dataset.
                    return { ...p, lat: origLat + (p.lat - turntableParams.latCenter) };
                },
                onProgress: p => {
                    this.turntableProgress = p;
                },
            });
        } catch (e) {
            showToast(`Turntable render failed: ${(e as Error)?.message ?? e}`, "error");
        } finally {
            this.isTurntableRendering = false;
            this.turntableProgress = 0;
        }
    }

    async loadModelFromFile(file: File) {
        if (!this.runner) {
            showToast("renderer not ready yet", "error");
            return;
        }
        const t = showToast(`loading ${file.name}…`, "info", 0);
        try {
            const buffer = await file.arrayBuffer();
            const mesh = parseGlbBuffer(buffer);
            // Flip Z: negate position Z (offset 2) and normal Z (offset 5) per vertex.
            const STRIDE = 12;
            for (let i = 0; i < mesh.vertices.length / STRIDE; i++) {
                mesh.vertices[i * STRIDE + 2] *= -1;
                mesh.vertices[i * STRIDE + 5] *= -1;
            }
            const t2 = showToast("building BVH…", "info", 0);
            this.meshVerts = new Float32Array(mesh.vertices);
            this.meshBvh = buildBvh(this.meshVerts, new Uint32Array(mesh.indices));
            dismissToast(t2);
            this.runner.replaceMesh(mesh);
            dismissToast(t);
            showToast(`loaded ${file.name}`, "success");
        } catch (e) {
            dismissToast(t);
            console.error("Failed to load model", e);
            showToast(`failed to load model: ${(e as Error)?.message ?? e}`, "error", 0);
        }
    }

    static mount({
        canvasesPromise,
    }: {
        canvasesPromise: Promise<Record<string, HTMLCanvasElement>>,
    }) {
        const state = new ViewerState();
        
        let stopLoop: (() => void) | null = null;
        onMount(async () => {
            // Kick off mesh load and gpu request concurrently; both are awaited
            // before we build the runner since the mesh is a constructor input.
            const t0 = showToast("loading meshes & gpu…", "info", 0);
            const [gpu, mesh, groundMesh, groundPbrMesh] = await Promise.all([
                requestGpu({
                    onStatusChange: (text) => showToast(text, "info", 2500),
                    onErr: (text) => showToast(text, "error", 0),
                }),
                loadGlb(artelorianUrl).then(r => { showToast("mesh loaded", "info", 2000); return r; }),
                loadGlb(groundUrl, false, [1, 1, 1, 0]),
                loadGlb(groundUrl, false, [1, 1, 1, 1], 'Plane.001'),
            ]);
            dismissToast(t0);
            if (!gpu) return;

            state.gpuTimestampQuerySupported = gpu.supportsTimestamp;

            const t1 = showToast("building BVH…", "info", 0);
            state.meshVerts = new Float32Array(mesh.vertices);
            state.meshBvh = buildBvh(state.meshVerts, new Uint32Array(mesh.indices));
            dismissToast(t1);
            showToast("BVH ready", "info", 2000);

            const t2 = showToast("loading textures…", "info", 0);
            const [envTexture, brushTexture, groundAlbedoTexture, groundNormalTexture] = await Promise.all([
                loadHdrTexture(gpu.device, hdrUrl).then(r => { showToast("environment loaded", "info", 2000); return r; }),
                loadTexture(gpu.device, brushUrl),
                loadTexture(gpu.device, groundAlbedoUrl),
                loadTexture(gpu.device, groundNormalUrl).then(r => { showToast("textures loaded", "info", 2000); return r; }),
            ]);
            dismissToast(t2);

            const gpuRunner = new GpuRunner({
                device: gpu.device,
                canvases: await canvasesPromise,
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
                gpuTimestampSupported: gpu.supportsTimestamp,
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