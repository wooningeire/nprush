import { onDestroy, onMount } from "svelte";
import { Camera } from "./Camera.svelte.ts";
import { CameraOrbit } from "./CameraOrbit.svelte.ts";
import { GpuRunner } from "./GpuRunner.svelte.ts";
import { parseGlbBuffer } from "../gpu/file-load/loadGlb.ts";
import { buildBvh, raycastBvh, type BvhResult } from "../gpu/bvh.ts";
import { vec4 } from "wgpu-matrix";
import { downloadBlob, openFrameWriter } from "../util/export.ts";
import {
    RENDER_MODE_SINGLE_VIEW_REALTIME,
    type RenderMode,
} from "./renderMode.ts";
import { evaluateTurntablePath, type TurntablePathParams } from "./turntable/turntablePath.ts";
import { runTurntableExport } from "./turntable/turntableExport.ts";
import { showToast, dismissToast } from "./toast.svelte.ts";
import { loadInitialAssetsAndGpu } from "../gpu/setup/loadInitialAssetsAndGpu.ts";

export class ViewerState {
    width = $state(300);
    height = $state(150);

    // Output render resolution — independent of the browser window size.
    // These drive the full-res textures used for visualization and capture.
    renderWidth = $state(256);
    renderHeight = $state(256);

    edgeBeziersEnabled = $state(false);
    coarseColorBeziersEnabled = $state(true);
    fineColorBeziersEnabled = $state(true);
    splatsEnabled = $state(true);
    splatTrainingPaused = $state(false);
    edgeBezierTrainingPaused = $state(false);
    coarseColorBezierTrainingPaused = $state(false);
    fineColorBezierTrainingPaused = $state(false);
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
    gpuProfilingEntries = $state<{label: string, ms: number | null}[]>([]);
    gpuProfilingHistoryFrames = $state<number[]>([]);

    setGpuProfilingFrameMs(entries: {label: string, ms: number | null}[]) {
        this.gpuProfilingEntries = entries;
        const total = entries.reduce((sum, e) => sum + (e.ms ?? 0), 0);
        const next = [...this.gpuProfilingHistoryFrames, total];
        const cap = 180;
        this.gpuProfilingHistoryFrames = next.length > cap ? next.slice(-cap) : next;
    }

    runner = $state<GpuRunner | null>(null);
    
    readonly viewportOrbit = new CameraOrbit();
    readonly viewportCamera = new Camera({
        controlScheme: this.viewportOrbit,
        screenDims: { width: () => this.width, height: () => this.height },
    });
    
    readonly backendOrbit = new CameraOrbit();
    readonly backendCamera = new Camera({
        controlScheme: this.backendOrbit,
        screenDims: { width: () => this.renderWidth, height: () => this.renderHeight },
    });

    onPaintDrag(x: number, y: number, targetWidth: number, targetHeight: number) {
        if (!this.meshBvh || !this.meshVerts || !this.runner || !this.meshSplatsEnabled) {
            return;
        }
        
        const ndcX = (x / targetWidth) * 2 - 1;
        const ndcY = -((y / targetHeight) * 2 - 1);
        
        if (ndcX < -1 || ndcX > 1 || ndcY < -1 || ndcY > 1) {
            return;
        }

        
        const originNdC = [ndcX, ndcY, 0, 1];
        const targetNdC = [ndcX, ndcY, 1, 1];
        
        const originWorldW = vec4.transformMat4(originNdC, this.viewportCamera.viewProjInvMat);
        const targetWorldW = vec4.transformMat4(targetNdC, this.viewportCamera.viewProjInvMat);
        
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
            showToast(`Screenshot failed: ${(e as Error)?.message ?? e}`, "warning");
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
        this._turntableBaseLong = this.viewportOrbit.long;
        this.turntableLatCenter = this.viewportOrbit.lat;
        this.turntableRadiusCenter = this.viewportOrbit.radius;
        
        this.backendOrbit.long = this.viewportOrbit.long;
        this.backendOrbit.lat = this.viewportOrbit.lat;
        this.backendOrbit.radius = this.viewportOrbit.radius;
        
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

        const origLong = this.backendOrbit.long;
        const origLat = this.backendOrbit.lat;
        const origRadius = this.backendOrbit.radius;

        const baseLong = this.turntableTraining ? this.turntableBaseLong : origLong;
        const turntableParams = this.getTurntablePathParams();

        try {
            await runTurntableExport({
                totalFrames: this.turntableFrameCount,
                stepsPerFrame: this.renderMode === RENDER_MODE_SINGLE_VIEW_REALTIME ? this.turntableStepsPerFrame : 1,
                isCanceled: () => this.turntableCanceled,
                captureFrame: () => this.runner!.captureTurntableFrame(),
                writer,
                orbit: this.backendOrbit,
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
            const initialLoadResult = await loadInitialAssetsAndGpu(state);
            if (initialLoadResult === null) return;

            const {
                gpu,
                primaryMesh,
                groundMesh,
                groundPbrMesh,
                envTexture,
                coarseBrushTexture,
                fineBrushTexture,
                groundAlbedoTexture,
                groundNormalTexture,
            } = initialLoadResult;

    


            const canvases = await canvasesPromise;
            const contexts: Record<string, GPUCanvasContext> = {};

            for (const [id, canvas] of Object.entries(canvases)) {
                const webgpuContext = canvas.getContext("webgpu");
                if (webgpuContext === null) {
                    showToast("couldn't attach WebGPU to a canvas", "error", 0);
                    return;
                }

                webgpuContext.configure({
                    device: gpu.device,
                    format: gpu.format,
                    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
                    alphaMode: "premultiplied",
                });
                contexts[id] = webgpuContext;
            }


            const gpuRunner = new GpuRunner({
                device: gpu.device,
                contexts,
                format: gpu.format,
                viewportCamera: state.viewportCamera,
                backendCamera: state.backendCamera,
                viewerState: state,
                mesh: primaryMesh,
                groundMesh,
                groundPbrMesh,
                matcapTexture: envTexture,
                coarseBrushTexture,
                fineBrushTexture,
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