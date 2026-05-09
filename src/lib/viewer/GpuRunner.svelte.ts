import type { Camera } from "./Camera.svelte";
import { GpuUniformsBufferManager } from "$/gpu/GpuUniformsBufferManager";
import { GpuMeshRenderPipelineManager, MESH_DEPTH_FORMAT } from "$/gpu/GpuMeshRenderPipelineManager";
import { GpuSplatOptimizerManager } from "../gpu/splat/GpuSplatOptimizerManager.ts";
import { GpuBezierOptimizerManager } from "../gpu/bezier/GpuBezierOptimizerManager.ts";
import { GpuSplatForwardPipelineManager } from "../gpu/splat/GpuSplatForwardPipelineManager.ts";
import { GpuBezierForwardPipelineManager } from "../gpu/bezier/GpuBezierForwardPipelineManager.ts";
import { GpuBlurPipelineManager } from "../gpu/blur/GpuBlurPipelineManager.ts";
import { GpuDepthAwareBlurPipelineManager } from "../gpu/blur/GpuDepthAwareBlurPipelineManager.ts";
import { GpuEnvmapPipelineManager } from "../gpu/envmap/GpuEnvmapPipelineManager.ts";
import { GpuPathTracePipelineManager } from "../gpu/pathtrace/GpuPathTracePipelineManager.ts";
import type { MeshData } from "../gpu/io/loadGlb.ts";
import type { ViewerState } from "./ViewerState.svelte.ts";
import type { Mat4 } from "wgpu-matrix";
import { constants } from "$/gpu/constants";
import { computeOptimTextureSize } from "$/gpu/optimTextureSize.ts";
import { readTextureToImageData, imageDataToBlob } from "$/gpu/io/readback.ts";

const OPTIM_SHORT = constants.OPTIM_SHORT;

// The edge layer is now cubic bezier curves. A handful is enough since each
// curve is a 1D primitive that natively traces a contour.
const NUM_EDGE_LAYER_BEZIERS = constants.NUM_EDGE_LAYER_BEZIERS;

/**
 * Fixed prerendered multiview dataset.
 *
 * Holds N converged path-traced images (one per camera view) plus the
 * corresponding camera matrices. Each texture is rgba8unorm at optim-res.
 * Memory: N × W × H × 4 bytes — 32 views at 128×128 ≈ 2 MB.
 */
class MultiviewDataset {
    readonly textures: GPUTexture[];
    readonly textureViews: GPUTextureView[];
    /** Per-view viewProjMat (16 f32 each). */
    readonly viewProjMats: Float32Array[];
    /** Per-view viewMat (16 f32 each). */
    readonly viewMats: Float32Array[];
    /** Per-view invViewProjMat (16 f32 each). */
    readonly invViewProjMats: Float32Array[];
    readonly numViews: number;

    constructor(device: GPUDevice, numViews: number, width: number, height: number) {
        this.numViews = numViews;
        this.textures = [];
        this.textureViews = [];
        this.viewProjMats = [];
        this.viewMats = [];
        this.invViewProjMats = [];
        for (let i = 0; i < numViews; i++) {
            const tex = device.createTexture({
                label: `multiview dataset slot ${i}`,
                size: [width, height],
                format: "rgba8unorm",
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
            this.textures.push(tex);
            this.textureViews.push(tex.createView({ label: `multiview view ${i}` }));
            this.viewProjMats.push(new Float32Array(16));
            this.viewMats.push(new Float32Array(16));
            this.invViewProjMats.push(new Float32Array(16));
        }
    }

    destroy() {
        for (const tex of this.textures) tex.destroy();
    }
}

export class GpuRunner {
    private readonly device: GPUDevice;
    private readonly context: GPUCanvasContext;
    private readonly format: GPUTextureFormat;
    private readonly camera: Camera;
    private readonly viewerState: ViewerState;

    readonly uniformsManager: GpuUniformsBufferManager;
    readonly meshRenderPipelineManager: GpuMeshRenderPipelineManager;
    readonly splatOptimizerManager: GpuSplatOptimizerManager;
    // The edge layer is a separate optimizer of cubic bezier curves trained
    // against the depth-edge texture. Curves natively represent 1D contours,
    // which is a much better fit for the silhouette target than gaussians.
    readonly edgeLayerBezierManager: GpuBezierOptimizerManager;
    readonly baseColorLayerBezierManager: GpuBezierOptimizerManager;
    readonly colorLayerBezierManager: GpuBezierOptimizerManager;
    readonly splatForwardManager: GpuSplatForwardPipelineManager;
    readonly bezierForwardManager: GpuBezierForwardPipelineManager;
    readonly baseColorBezierForwardManager: GpuBezierForwardPipelineManager;
    readonly colorBezierForwardManager: GpuBezierForwardPipelineManager;
    private readonly blurManager: GpuBlurPipelineManager;
    private readonly depthAwareBlurManager: GpuDepthAwareBlurPipelineManager;
    private readonly matcapTexture: GPUTexture;
    private readonly matcapTextureView: GPUTextureView;
    private readonly envmapPipelineManager: GpuEnvmapPipelineManager;
    readonly pathTracePipelineManager: GpuPathTracePipelineManager;

    // Full-res textures (sized to the visible main panel area: half-width x height-minus-strip).
    // These match the camera projection aspect so the rendered model has the same pixel
    // proportions as a square-rendered version (no horizontal/vertical squash).
    //
    // Note: targetDepthTexture is an RGBA8 *visualization* depth (linear view-space depth
    // remapped to grayscale, used by Sobel for edge detection). It is NOT the hardware
    // Z-buffer. The hardware Z-buffer is targetZTexture below; without it, triangles draw
    // in submission order which causes back-face leakage on overlapping geometry.
    private targetTexture: GPUTexture | null = null;
    private targetTextureView: GPUTextureView | null = null;
    private targetDepthTexture: GPUTexture | null = null;
    private targetDepthTextureView: GPUTextureView | null = null;
    private targetZTexture: GPUTexture | null = null;
    private targetZTextureView: GPUTextureView | null = null;
    private fullEdgeTexture: GPUTexture | null = null;
    private fullEdgeTextureView: GPUTextureView | null = null;
    private fullSplatTexture: GPUTexture | null = null;
    private fullSplatTextureView: GPUTextureView | null = null;
    private fullSplatDepthTexture: GPUTexture | null = null;
    private fullSplatDepthTextureView: GPUTextureView | null = null;
    private fullBezierTexture: GPUTexture | null = null;
    private fullBezierTextureView: GPUTextureView | null = null;
    private fullBaseColorBezierTexture: GPUTexture | null = null;
    private fullBaseColorBezierTextureView: GPUTextureView | null = null;
    private fullColorBezierTexture: GPUTexture | null = null;
    private fullColorBezierTextureView: GPUTextureView | null = null;
    private targetBlurredTexture: GPUTexture | null = null;
    private targetBlurredTextureView: GPUTextureView | null = null;
    private targetTempTexture: GPUTexture | null = null;
    private targetTempTextureView: GPUTextureView | null = null;
    private fullWidth = 0;
    private fullHeight = 0;

    // Optim-res textures (aspect-matched to half-screen)
    private optimTexture: GPUTexture | null = null;
    private optimTextureView: GPUTextureView | null = null;
    private optimDepthTexture: GPUTexture | null = null;
    private optimDepthTextureView: GPUTextureView | null = null;
    private optimZTexture: GPUTexture | null = null;
    private optimZTextureView: GPUTextureView | null = null;
    private optimEdgeTexture: GPUTexture | null = null;
    private optimEdgeTextureView: GPUTextureView | null = null;
    private optimSplatTexture: GPUTexture | null = null;
    private optimSplatTextureView: GPUTextureView | null = null;
    private optimSplatDepthTexture: GPUTexture | null = null;
    private optimSplatDepthTextureView: GPUTextureView | null = null;
    private dummyTexture: GPUTexture | null = null;
    private dummyTextureView: GPUTextureView | null = null;
    private optimBlurredTexture: GPUTexture | null = null;
    private optimBlurredTextureView: GPUTextureView | null = null;
    private optimDepthAwareBlurredTexture: GPUTexture | null = null;
    private optimDepthAwareBlurredTextureView: GPUTextureView | null = null;
    private optimBlurredDepthTexture: GPUTexture | null = null;
    private optimBlurredDepthTextureView: GPUTextureView | null = null;
    private optimTempTexture: GPUTexture | null = null;
    private optimTempTextureView: GPUTextureView | null = null;
    private targetNormalTexture: GPUTexture | null = null;
    private targetNormalTextureView: GPUTextureView | null = null;
    private optimNormalTexture: GPUTexture | null = null;
    private optimNormalTextureView: GPUTextureView | null = null;
    private optimWidth = 0;
    private optimHeight = 0;

    private capturePromise: { resolve: (blob: Blob) => void, reject: (err: Error) => void } | null = null;

    // Prerendered multiview dataset — null until prerenderDataset() completes.
    private multiviewDataset: MultiviewDataset | null = null;

    readonly destroy: () => void;

    constructor({
        device,
        context,
        format,
        camera,
        viewerState,
        mesh,
        groundMesh,
        groundPbrMesh,
        matcapTexture,
        brushTexture,
        groundAlbedoTexture,
        groundNormalTexture,
        numSplats = constants.NUM_GAUSSIAN_SPLATS,
    }: {
        device: GPUDevice,
        context: GPUCanvasContext,
        format: GPUTextureFormat,
        camera: Camera,
        viewerState: any,
        mesh: MeshData,
        groundMesh: MeshData | null,
        groundPbrMesh: MeshData | null,
        matcapTexture: GPUTexture,
        brushTexture: GPUTexture,
        groundAlbedoTexture?: GPUTexture,
        groundNormalTexture?: GPUTexture,
        numSplats?: number,
    }) {
        this.device = device;
        this.context = context;
        this.format = format;
        this.camera = camera;
        this.viewerState = viewerState;
        this.matcapTexture = matcapTexture;
        this.matcapTextureView = matcapTexture.createView();

        this.uniformsManager = new GpuUniformsBufferManager({ device });

        this.meshRenderPipelineManager = new GpuMeshRenderPipelineManager({
            device,
            format,
            uniformsManager: this.uniformsManager,
            mesh,
        });

        if (groundMesh) {
            this.meshRenderPipelineManager.setGroundMesh(groundMesh);
        }
        if (groundPbrMesh && groundAlbedoTexture && groundNormalTexture) {
            this.meshRenderPipelineManager.setPbrMesh(groundPbrMesh, groundAlbedoTexture, groundNormalTexture, matcapTexture);
        }

        this.envmapPipelineManager = new GpuEnvmapPipelineManager({
            device,
            format,
            uniformsManager: this.uniformsManager,
            envTexture: matcapTexture,
        });

        this.pathTracePipelineManager = new GpuPathTracePipelineManager({
            device,
            envTexture: matcapTexture,
        });
        // Upload scene geometry for path tracing
        const ptMeshes: MeshData[] = [mesh];
        if (groundMesh) ptMeshes.push(groundMesh);
        if (groundPbrMesh) ptMeshes.push(groundPbrMesh);
        this.pathTracePipelineManager.setMeshes(ptMeshes);

        // The color-layer instance owns the visualization render pipeline, which
        // composites both layers. We tell it the edge layer's bezier count via
        // numBeziers so its render shader sizes the bezier loop correctly.
        this.splatOptimizerManager = new GpuSplatOptimizerManager({
            device,
            format,
            numSplats,
            numBeziers: NUM_EDGE_LAYER_BEZIERS,
        });

        this.edgeLayerBezierManager = new GpuBezierOptimizerManager({
            device,
            numBeziers: NUM_EDGE_LAYER_BEZIERS,
        });

        this.baseColorLayerBezierManager = new GpuBezierOptimizerManager({
            device,
            numBeziers: NUM_EDGE_LAYER_BEZIERS,
        });

        this.colorLayerBezierManager = new GpuBezierOptimizerManager({
            device,
            numBeziers: NUM_EDGE_LAYER_BEZIERS,
        });

        this.splatForwardManager = new GpuSplatForwardPipelineManager({
            device,
            numSplats,
            splatBuffer: this.splatOptimizerManager.splatBuffer,
            sortOrderBuffer: this.splatOptimizerManager.sortIndicesBuffer,
        });

        this.bezierForwardManager = new GpuBezierForwardPipelineManager({
            device,
            numBeziers: NUM_EDGE_LAYER_BEZIERS,
            bezierBuffer: this.edgeLayerBezierManager.bezierBuffer,
            sortOrderBuffer: this.edgeLayerBezierManager.sortIndicesBuffer,
            brushTexture,
        });

        this.baseColorBezierForwardManager = new GpuBezierForwardPipelineManager({
            device,
            numBeziers: NUM_EDGE_LAYER_BEZIERS,
            bezierBuffer: this.baseColorLayerBezierManager.bezierBuffer,
            sortOrderBuffer: this.baseColorLayerBezierManager.sortIndicesBuffer,
            brushTexture,
        });

        this.colorBezierForwardManager = new GpuBezierForwardPipelineManager({
            device,
            numBeziers: NUM_EDGE_LAYER_BEZIERS,
            bezierBuffer: this.colorLayerBezierManager.bezierBuffer,
            sortOrderBuffer: this.colorLayerBezierManager.sortIndicesBuffer,
            brushTexture,
        });
        
        this.blurManager = new GpuBlurPipelineManager(device);
        this.depthAwareBlurManager = new GpuDepthAwareBlurPipelineManager(device);

        this.destroy = $effect.root(() => {
            $effect(() => this.uniformsManager.writeViewProjMat(this.camera.viewProjMat));
            $effect(() => this.uniformsManager.writeViewMat(this.camera.viewMat));
            $effect(() => this.uniformsManager.writeInvViewProjMat(this.camera.viewProjInvMat));
            $effect(() => {
                // Write invViewProjMat to path tracer and reset accumulation on camera change
                this.pathTracePipelineManager.writeInvViewProjMat(this.camera.viewProjInvMat as Float32Array);
                this.pathTracePipelineManager.reset();
            });
            $effect(() => {
                this.pathTracePipelineManager.setSplatResources(
                    this.meshRenderPipelineManager.meshSplatsBuffer,
                    this.meshRenderPipelineManager.meshUniformsBuffer
                );
            });
            $effect(() => {
                // Reset path tracer when mesh splats enabled state or count changes
                this.viewerState.meshSplatsEnabled;
                this.meshRenderPipelineManager.numMeshSplats;
                this.pathTracePipelineManager.reset();
            });
            $effect(() => this.meshRenderPipelineManager.writeMeshSplatsEnabled(this.viewerState.meshSplatsEnabled));
            $effect(() => this.splatOptimizerManager.writeRenderUniforms(
                this.viewerState.edgeBeziersEnabled,
                this.viewerState.baseColorBeziersEnabled,
                this.viewerState.colorBeziersEnabled,
                this.viewerState.meshSplatsEnabled,
                this.viewerState.splatsEnabled,
            ));
            $effect(() => {
                if (this.viewerState.renderMode !== 'animation') {
                    this.splatOptimizerManager.writeSplatVPMatrix(this.camera.viewProjMat, this.camera.viewProjInvMat, this.viewerState.compareBlurred);
                }
            });
            $effect(() => {
                // Reset Adam momentum on camera change so stale cross-view gradients
                // don't corrupt the step for the new viewpoint during turntable training.
                // Mirrors the pathtracer's reset() call on camera change.
                this.camera.viewProjMat;
                this.splatOptimizerManager.resetAdam();
                this.edgeLayerBezierManager.resetAdam();
                this.baseColorLayerBezierManager.resetAdam();
                this.colorLayerBezierManager.resetAdam();
            });
            $effect(() => this.splatForwardManager.writeVPMatrix(this.camera.viewProjMat));
            $effect(() => {
                if (this.viewerState.renderMode !== 'animation') {
                    this.edgeLayerBezierManager.writeVPMatrix(this.camera.viewProjMat);
                    this.baseColorLayerBezierManager.writeVPMatrix(this.camera.viewProjMat);
                    this.colorLayerBezierManager.writeVPMatrix(this.camera.viewProjMat);
                    this.edgeLayerBezierManager.writeVPInvMatrix(this.camera.viewProjInvMat);
                    this.baseColorLayerBezierManager.writeVPInvMatrix(this.camera.viewProjInvMat);
                    this.colorLayerBezierManager.writeVPInvMatrix(this.camera.viewProjInvMat);
                }
            });
            $effect(() => this.bezierForwardManager.writeVPMatrix(this.camera.viewProjMat));
            $effect(() => this.baseColorBezierForwardManager.writeVPMatrix(this.camera.viewProjMat));
            $effect(() => this.colorBezierForwardManager.writeVPMatrix(this.camera.viewProjMat));
            $effect(() => {
                this.edgeLayerBezierManager.writeMode(0); // Edge mode
                this.baseColorLayerBezierManager.writeMode(1); // Color+Depth mode
                this.colorLayerBezierManager.writeMode(1); // Color+Depth mode
                this.colorLayerBezierManager.writeMaxWidth(0.03); // finer strokes on second color layer
                // Fine color layer: less aggressive killing so thin strokes survive,
                // but background penalty enabled to kill off-model curves.
                this.colorLayerBezierManager.writeKillThresholds(0.0001, 0.0001);
                this.colorLayerBezierManager.writeBgPenalty(0.0);
                // Base color layer: no background penalty (blurred target bleeds into bg).
                // Enable no_kill so broad strokes aren't pruned before they settle —
                // the ADC stuck+loss kill was the main source of base-layer jitter.
                // Longer ADC period reduces clone/kill churn on broad strokes.
                this.baseColorLayerBezierManager.writeMaxWidth(2);
                this.baseColorLayerBezierManager.writeKillThresholds(0.0001, 0.0001);
                this.baseColorLayerBezierManager.setAdcPeriod(150);
            });
            $effect(() => {
                // During animation mode the camera rotates to random angles each
                // view, so edge-layer curves that are off-screen from the current angle
                // must not be killed — they will be visible again from other views.
                // Setting no_kill suppresses both the loss-based and offscreen kills in
                // bezier_adc.wgsl and the per-step offscreen cull in bezier_step.wgsl.
                this.edgeLayerBezierManager.writeNoKill(this.viewerState.renderMode === 'animation');
            });

            return () => {
                // Cleanup managers
                this.uniformsManager.destroy();
                this.meshRenderPipelineManager.destroy();
                this.splatOptimizerManager.destroy();
                this.edgeLayerBezierManager.destroy();
                this.baseColorLayerBezierManager.destroy();
                this.colorLayerBezierManager.destroy();
                this.splatForwardManager.destroy();
                this.bezierForwardManager.destroy();
                this.baseColorBezierForwardManager.destroy();
                this.colorBezierForwardManager.destroy();
                this.blurManager.destroy();
                this.depthAwareBlurManager.destroy();
                this.pathTracePipelineManager.destroy();
                this.envmapPipelineManager.destroy?.();

                // Cleanup all textures owned by runner
                this.targetTexture?.destroy();
                this.targetDepthTexture?.destroy();
                this.targetZTexture?.destroy();
                this.targetNormalTexture?.destroy();
                this.fullEdgeTexture?.destroy();
                this.fullSplatTexture?.destroy();
                this.fullSplatDepthTexture?.destroy();
                this.fullBezierTexture?.destroy();
                this.fullBaseColorBezierTexture?.destroy();
                this.fullColorBezierTexture?.destroy();
                this.targetBlurredTexture?.destroy();
                this.targetTempTexture?.destroy();

                this.optimTexture?.destroy();
                this.optimDepthTexture?.destroy();
                this.optimZTexture?.destroy();
                this.optimNormalTexture?.destroy();
                this.optimEdgeTexture?.destroy();
                this.optimSplatTexture?.destroy();
                this.optimSplatDepthTexture?.destroy();
                this.optimBlurredTexture?.destroy();
                this.optimDepthAwareBlurredTexture?.destroy();
                this.optimBlurredDepthTexture?.destroy();
                this.optimTempTexture?.destroy();
                this.dummyTexture?.destroy();

                this.multiviewDataset?.destroy();
            };
        });
    }

    async takeScreenshot(): Promise<Blob> {
        return new Promise((resolve, reject) => {
            this.capturePromise = { resolve, reject };
        });
    }

    /**
     * Captures a single clean composited frame (splats + bezier layers, no debug UI)
     * at the current camera angle.  The returned ImageData has dimensions
     * [fullW × fullH] matching the right-half panel.
     *
     * The promise resolves on the next rAF tick after the GPU readback completes.
     */
    captureTurntableFrame(): Promise<ImageData> {
        return new Promise((resolve, reject) => {
            this.turntableFrameRequest = { resolve, reject };
        });
    }

    private turntableFrameRequest: {
        resolve: (img: ImageData) => void;
        reject: (err: Error) => void;
    } | null = null;

    private async readTextureToImageData(
        texture: GPUTexture,
        width: number,
        height: number,
    ): Promise<ImageData> {
        return readTextureToImageData(this.device, texture, width, height, texture.format);
    }

    /**
     * Build a prerendered multiview dataset.
     *
     * For each of `numViews` camera positions sampled from the turntable path:
     *   1. Set the camera to that position.
     *   2. Accumulate `minSamplesPerView` PT frames (one per rAF tick).
     *   3. Copy the resolved PT output into a dedicated GPUTexture slot.
     *
     * Memory: numViews × (ow × oh × 4 bytes) ≈ 32 × 128×128×4 = 2 MB.
     * After this returns, `multiviewDataset` is populated and training can
     * use it as a fixed dataset — no live PT dispatch needed per frame.
     */
    async prerenderDataset(): Promise<void> {
        const vs = this.viewerState;
        const numViews = vs.multiviewNumViews as number;
        const samplesPerView = vs.turntableMinSamplesPerView as number;

        // Wait until optim textures are ready (loop may not have run yet).
        while (!this.optimTextureView || this.optimWidth === 0) {
            await new Promise<void>(r => requestAnimationFrame(() => r()));
        }

        const ow = this.optimWidth;
        const oh = this.optimHeight;

        // Destroy any previous dataset and allocate fresh slots.
        this.multiviewDataset?.destroy();
        this.multiviewDataset = null;
        const dataset = new MultiviewDataset(this.device, numViews, ow, oh);

        vs.multiviewPrerendering = true;
        vs.multiviewPrerenderProgress = 0;
        vs.multiviewDatasetReady = false;

        try {
            for (let i = 0; i < numViews; i++) {
                if (!vs.turntableTraining) break; // canceled

                // Sample a deterministic view position spread evenly around the path.
                const t = i / numViews;
                const p = vs.evaluatePath(t, (vs as any).turntableBaseLong);
                vs.orbit.long = p.long;
                vs.orbit.lat = p.lat;
                vs.orbit.radius = p.radius;

                // Reset PT accumulation for this new view.
                this.pathTracePipelineManager.reset();

                // Accumulate PT samples — one per rAF tick.
                for (let s = 0; s < samplesPerView; s++) {
                    if (!vs.turntableTraining) break;
                    await new Promise<void>(r => requestAnimationFrame(() => r()));
                }

                if (!vs.turntableTraining) break;

                // Copy the resolved PT output into the dataset slot.
                // The PT output texture is already at optim-res (ow × oh).
                const ptTex = this.pathTracePipelineManager.outputTexture;
                if (ptTex) {
                    const enc = this.device.createCommandEncoder({ label: `prerender copy view ${i}` });
                    enc.copyTextureToTexture(
                        { texture: ptTex },
                        { texture: dataset.textures[i] },
                        [ow, oh, 1],
                    );
                    this.device.queue.submit([enc.finish()]);
                }

                // Store the camera matrices for this view.
                dataset.viewProjMats[i].set(this.camera.viewProjMat as Float32Array);
                dataset.viewMats[i].set(this.camera.viewMat as Float32Array);
                dataset.invViewProjMats[i].set(this.camera.viewProjInvMat as Float32Array);

                vs.multiviewPrerenderProgress = (i + 1) / numViews;
            }
        } finally {
            vs.multiviewPrerendering = false;
            if (vs.turntableTraining) {
                this.multiviewDataset = dataset;
                vs.multiviewDatasetReady = true;
                // Reset Adam so training starts fresh from the new dataset.
                this.splatOptimizerManager.resetAdam();
                this.edgeLayerBezierManager.resetAdam();
                this.baseColorLayerBezierManager.resetAdam();
                this.colorLayerBezierManager.resetAdam();
                this.edgeLayerBezierManager.resetAdcState();
                this.baseColorLayerBezierManager.resetAdcState();
                this.colorLayerBezierManager.resetAdcState();
            } else {
                dataset.destroy();
            }
        }
    }

    private rebuildOptimTextures(panelAspect: number) {
        // Size optim textures to match the visible panel aspect ratio so the model
        // rendered into them has matching pixel proportions for the gradient pass.
        const { width: ow, height: oh } = computeOptimTextureSize(OPTIM_SHORT, panelAspect);

        if (ow === this.optimWidth && oh === this.optimHeight) return;
        this.optimWidth = ow;
        this.optimHeight = oh;

        // Resize path tracer output to match optim resolution
        this.pathTracePipelineManager.setOutputSize(ow, oh);

        if (this.optimTexture) this.optimTexture.destroy();
        if (this.optimNormalTexture) this.optimNormalTexture.destroy();
        if (this.optimDepthTexture) this.optimDepthTexture.destroy();
        if (this.optimZTexture) this.optimZTexture.destroy();
        if (this.optimEdgeTexture) this.optimEdgeTexture.destroy();
        if (this.optimSplatTexture) this.optimSplatTexture.destroy();
        if (this.optimSplatDepthTexture) this.optimSplatDepthTexture.destroy();
        if (this.optimBlurredTexture) this.optimBlurredTexture.destroy();
        if (this.optimDepthAwareBlurredTexture) this.optimDepthAwareBlurredTexture.destroy();
        if (this.optimBlurredDepthTexture) this.optimBlurredDepthTexture.destroy();
        if (this.optimTempTexture) this.optimTempTexture.destroy();

        this.optimTexture = this.device.createTexture({
            label: "optimization target texture",
            size: [ow, oh],
            format: this.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimTextureView = this.optimTexture.createView();
        
        this.optimNormalTexture = this.device.createTexture({
            label: "optimization normal texture",
            size: [ow, oh],
            format: this.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimNormalTextureView = this.optimNormalTexture.createView();

        this.optimDepthTexture = this.device.createTexture({
            label: "optimization depth visualization",
            size: [ow, oh],
            format: this.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimDepthTextureView = this.optimDepthTexture.createView();

        this.optimZTexture = this.device.createTexture({
            label: "optimization z-buffer",
            size: [ow, oh],
            format: MESH_DEPTH_FORMAT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.optimZTextureView = this.optimZTexture.createView();

        this.optimEdgeTexture = this.device.createTexture({
            label: "optimization edge map",
            size: [ow, oh],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimEdgeTextureView = this.optimEdgeTexture.createView();

        this.optimSplatTexture = this.device.createTexture({
            label: "optimization splat view",
            size: [ow, oh],
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimSplatTextureView = this.optimSplatTexture.createView();

        this.optimSplatDepthTexture = this.device.createTexture({
            label: "optimization splat depth",
            size: [ow, oh],
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimSplatDepthTextureView = this.optimSplatDepthTexture.createView();

        this.optimBlurredTexture = this.device.createTexture({
            label: "optimization blurred target",
            size: [ow, oh],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimBlurredTextureView = this.optimBlurredTexture.createView();

        this.optimDepthAwareBlurredTexture = this.device.createTexture({
            label: "optimization depth-aware blurred target",
            size: [ow, oh],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimDepthAwareBlurredTextureView = this.optimDepthAwareBlurredTexture.createView();

        this.optimBlurredDepthTexture = this.device.createTexture({
            label: "optimization blurred depth",
            size: [ow, oh],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimBlurredDepthTextureView = this.optimBlurredDepthTexture.createView();

        this.optimTempTexture = this.device.createTexture({
            label: "optimization blur temp",
            size: [ow, oh],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimTempTextureView = this.optimTempTexture.createView();

        if (!this.dummyTexture) {
            this.dummyTexture = this.device.createTexture({
                label: "dummy 1x1 texture",
                size: [1, 1],
                format: "rgba8unorm",
                usage: GPUTextureUsage.TEXTURE_BINDING,
            });
            this.dummyTextureView = this.dummyTexture.createView();
        }

        // Rebind
        this.splatOptimizerManager.setEdgeTarget(this.optimDepthTextureView, this.optimEdgeTextureView);
        this.splatOptimizerManager.setBackwardTarget(this.optimTextureView, this.optimDepthTextureView, ow, oh);

        // Edge layer: target IS the edge texture itself.
        this.edgeLayerBezierManager.setBackwardTarget(
            this.optimEdgeTextureView,
            this.optimEdgeTextureView,
            this.dummyTextureView!,
            this.dummyTextureView!,
            this.optimTextureView!,
            ow, oh
        );

        // Base Color layer: target is depth-aware blurred color + sharp depth, background is splat output
        this.baseColorLayerBezierManager.setBackwardTarget(
            this.optimDepthAwareBlurredTextureView!,
            this.optimDepthTextureView!,
            this.optimSplatTextureView!,
            this.optimSplatDepthTextureView!,
            this.optimTextureView!,
            ow, oh
        );

        // Color layer: target is sharp color + depth, background is splat output
        this.colorLayerBezierManager.setBackwardTarget(
            this.optimTextureView!,
            this.optimDepthTextureView!,
            this.optimSplatTextureView!,
            this.optimSplatDepthTextureView!,
            this.optimTextureView!,
            ow, oh
        );
    }

    loop() {
        let handle = 0;
        let canceled = false;

        const loop = () => {
            const currentTexture = this.context.getCurrentTexture();
            const width = currentTexture.width;
            const height = currentTexture.height;

            // Optim textures use a fixed 1:1 aspect so optimization is independent
            // of the browser window size. Only the full-res display textures track
            // the visible panel dimensions.
            this.rebuildOptimTextures(1.0);

            // Full-res render textures use a fixed resolution from viewerState,
            // independent of the browser window size.
            const fullW = Math.max(1, this.viewerState.renderWidth);
            const fullH = Math.max(1, this.viewerState.renderHeight);
            if (!this.targetTexture || this.fullWidth !== fullW || this.fullHeight !== fullH) {
                if (this.targetTexture) this.targetTexture.destroy();
                if (this.targetNormalTexture) this.targetNormalTexture.destroy();
                if (this.targetDepthTexture) this.targetDepthTexture.destroy();
                if (this.targetZTexture) this.targetZTexture.destroy();
                if (this.fullEdgeTexture) this.fullEdgeTexture.destroy();
                if (this.fullSplatTexture) this.fullSplatTexture.destroy();
                if (this.fullSplatDepthTexture) this.fullSplatDepthTexture.destroy();
                if (this.fullBezierTexture) this.fullBezierTexture.destroy();
                if (this.fullBaseColorBezierTexture) this.fullBaseColorBezierTexture.destroy();
                if (this.fullColorBezierTexture) this.fullColorBezierTexture.destroy();
                if (this.targetBlurredTexture) this.targetBlurredTexture.destroy();
                if (this.targetTempTexture) this.targetTempTexture.destroy();

                this.fullWidth = fullW;
                this.fullHeight = fullH;

                this.targetTexture = this.device.createTexture({
                    label: "full-res target texture",
                    size: [fullW, fullH],
                    format: this.format,
                    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
                });
                this.targetTextureView = this.targetTexture.createView();
                
                this.targetNormalTexture = this.device.createTexture({
                    label: "full-res normal texture",
                    size: [fullW, fullH],
                    format: this.format,
                    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
                });
                this.targetNormalTextureView = this.targetNormalTexture.createView();

                this.targetDepthTexture = this.device.createTexture({
                    label: "full-res depth visualization",
                    size: [fullW, fullH],
                    format: this.format,
                    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
                });
                this.targetDepthTextureView = this.targetDepthTexture.createView();

                this.targetZTexture = this.device.createTexture({
                    label: "full-res z-buffer",
                    size: [fullW, fullH],
                    format: MESH_DEPTH_FORMAT,
                    usage: GPUTextureUsage.RENDER_ATTACHMENT,
                });
                this.targetZTextureView = this.targetZTexture.createView();

                this.fullEdgeTexture = this.device.createTexture({
                    label: "full-res edge map",
                    size: [fullW, fullH],
                    format: "rgba8unorm",
                    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
                });
                this.fullEdgeTextureView = this.fullEdgeTexture.createView();

                this.fullSplatTexture = this.device.createTexture({
                    label: "full-res splat view",
                    size: [fullW, fullH],
                    format: "rgba8unorm",
                    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
                });
                this.fullSplatTextureView = this.fullSplatTexture.createView();

                this.fullSplatDepthTexture = this.device.createTexture({
                    label: "full-res splat depth",
                    size: [fullW, fullH],
                    format: "rgba8unorm",
                    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
                });
                this.fullSplatDepthTextureView = this.fullSplatDepthTexture.createView();

                this.fullBezierTexture = this.device.createTexture({
                    label: "full-res bezier view",
                    size: [fullW, fullH],
                    format: "rgba8unorm",
                    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
                });
                this.fullBezierTextureView = this.fullBezierTexture.createView();

                this.fullBaseColorBezierTexture = this.device.createTexture({
                    label: "full-res base color bezier view",
                    size: [fullW, fullH],
                    format: "rgba8unorm",
                    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
                });
                this.fullBaseColorBezierTextureView = this.fullBaseColorBezierTexture.createView();

                this.fullColorBezierTexture = this.device.createTexture({
                    label: "full-res color bezier view",
                    size: [fullW, fullH],
                    format: "rgba8unorm",
                    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
                });
                this.fullColorBezierTextureView = this.fullColorBezierTexture.createView();

                this.targetBlurredTexture = this.device.createTexture({
                    label: "full-res blurred target",
                    size: [fullW, fullH],
                    format: "rgba8unorm",
                    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
                });
                this.targetBlurredTextureView = this.targetBlurredTexture.createView();

                this.targetTempTexture = this.device.createTexture({
                    label: "full-res blur temp",
                    size: [fullW, fullH],
                    format: "rgba8unorm",
                    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
                });
                this.targetTempTextureView = this.targetTempTexture.createView();

                this.splatForwardManager.setTarget(this.fullSplatTextureView, this.fullSplatDepthTextureView!, fullW, fullH);
                this.bezierForwardManager.setTarget(this.fullBezierTextureView, fullW, fullH);
                this.baseColorBezierForwardManager.setTarget(this.fullBaseColorBezierTextureView!, fullW, fullH);
                this.colorBezierForwardManager.setTarget(this.fullColorBezierTextureView, fullW, fullH);

                this.splatOptimizerManager.setRenderTarget(
                    this.targetTextureView!,
                    this.fullSplatTextureView,
                    this.targetDepthTextureView!,
                    this.fullEdgeTextureView!,
                    this.fullBezierTextureView,
                    this.fullBaseColorBezierTextureView!,
                    this.fullColorBezierTextureView,
                    this.dummyTextureView!, // PT not ready yet at setup time
                );
            }

            if (!this.targetTextureView || !this.targetDepthTextureView || !this.optimTextureView) {
                if (!canceled) requestAnimationFrame(loop);
                return;
            }

            // In animation mode, we might randomize camera each frame
            if (this.viewerState.renderMode === 'animation') {
                this.viewerState.tickAnimationMode();
            }

            const commandEncoder = this.device.createCommandEncoder({
                label: "runner loop command encoder",
            });

            // When the prerendered dataset is ready, pick a random view and
            // write its stored matrices to the uniforms buffers so the mesh
            // render and optimizers see the correct camera for this frame.
            // This replaces the live PT dispatch for the optimizer target.
            let datasetView: GPUTextureView | null = null;
            // Depth sort keys must use the same view-projection as the optimizer
            // uniforms for this frame (dataset random view vs live camera).
            let sortVp: Mat4 = this.camera.viewProjMat as Mat4;
            if (this.viewerState.turntableTraining && this.viewerState.multiviewDatasetReady && this.multiviewDataset) {
                const ds = this.multiviewDataset;
                const idx = Math.floor(Math.random() * ds.numViews);
                datasetView = ds.textureViews[idx];
                sortVp = ds.viewProjMats[idx] as Mat4;
                // Write camera matrices so the mesh render pass uses this view.
                this.uniformsManager.writeViewProjMat(ds.viewProjMats[idx]);
                this.uniformsManager.writeViewMat(ds.viewMats[idx]);
                this.uniformsManager.writeInvViewProjMat(ds.invViewProjMats[idx]);
                // Write VP to all optimizer uniforms.
                this.splatOptimizerManager.writeSplatVPMatrix(ds.viewProjMats[idx], ds.invViewProjMats[idx], this.viewerState.compareBlurred);
                this.edgeLayerBezierManager.writeVPMatrix(ds.viewProjMats[idx]);
                this.baseColorLayerBezierManager.writeVPMatrix(ds.viewProjMats[idx]);
                this.colorLayerBezierManager.writeVPMatrix(ds.viewProjMats[idx]);
                this.edgeLayerBezierManager.writeVPInvMatrix(ds.invViewProjMats[idx]);
                this.baseColorLayerBezierManager.writeVPInvMatrix(ds.invViewProjMats[idx]);
                this.colorLayerBezierManager.writeVPInvMatrix(ds.invViewProjMats[idx]);
            }

            // 1a. Render the model into the full-res target + depth textures (for visualization).
            if (!this.viewerState.viewportRenderingFrozen) {
                const spherePassEncoder = commandEncoder.beginRenderPass({
                    label: "mesh render pass (full res)",
                    colorAttachments: [
                        {
                            clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1.0 },
                            loadOp: "clear",
                            storeOp: "store",
                            view: this.targetTextureView,
                        },
                        {
                            clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
                            loadOp: "clear",
                            storeOp: "store",
                            view: this.targetDepthTextureView!,
                        },
                        {
                            clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 1.0 },
                            loadOp: "clear",
                            storeOp: "store",
                            view: this.targetNormalTextureView!,
                        },
                    ],
                    depthStencilAttachment: {
                        view: this.targetZTextureView!,
                        depthClearValue: 1.0,
                        depthLoadOp: "clear",
                        depthStoreOp: "store",
                    },
                });
                this.envmapPipelineManager.addDraw(spherePassEncoder);
                this.meshRenderPipelineManager.addDraw(spherePassEncoder, this.matcapTextureView);
                spherePassEncoder.end();
            }

            // 1b. Render the model into the optim-res (aspect-matched) textures for gradient computation.
            const optimPassEncoder = commandEncoder.beginRenderPass({
                label: "mesh render pass (optim res)",
                colorAttachments: [
                    {
                        clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1.0 },
                        loadOp: "clear",
                        storeOp: "store",
                        view: this.optimTextureView!,
                    },
                    {
                        clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
                        loadOp: "clear",
                        storeOp: "store",
                        view: this.optimDepthTextureView!,
                    },
                    {
                        clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 1.0 },
                        loadOp: "clear",
                        storeOp: "store",
                        view: this.optimNormalTextureView!,
                    },
                ],
                depthStencilAttachment: {
                    view: this.optimZTextureView!,
                    depthClearValue: 1.0,
                    depthLoadOp: "clear",
                    depthStoreOp: "store",
                },
            });
            this.envmapPipelineManager.addDraw(optimPassEncoder);
            this.meshRenderPipelineManager.addDraw(optimPassEncoder, this.matcapTextureView);
            optimPassEncoder.end();

            // 1b.5 Path trace pass — accumulates one sample per pixel into the PT output texture.
            // The PT output is used as the target for the splat/bezier optimizers instead of
            // the rasterized mesh render, giving a more physically-based training signal.
            // Skip during dataset-driven training — the prerendered textures are used directly.
            if (!datasetView) {
                this.pathTracePipelineManager.dispatch(commandEncoder);
            }

            // Use dataset view if available, else PT output, else raster fallback.
            const ptOutputView = this.pathTracePipelineManager.outputTextureView;
            const optimTargetView = datasetView ?? ptOutputView ?? this.optimTextureView!;

            // 1c. Run separable blur on targets if enabled
            if (this.viewerState.compareBlurred) {
                this.blurManager.blur(
                    commandEncoder,
                    optimTargetView,
                    this.optimBlurredTextureView!,
                    this.optimTempTextureView!,
                    this.optimWidth,
                    this.optimHeight,
                    this.viewerState.blurRadius,
                    this.viewerState.blurRadius / 2,
                    true // isSrgb
                );
                this.blurManager.blur(
                    commandEncoder,
                    this.optimDepthTextureView!,
                    this.optimBlurredDepthTextureView!,
                    this.optimTempTextureView!,
                    this.optimWidth,
                    this.optimHeight,
                    this.viewerState.blurRadius,
                    this.viewerState.blurRadius / 2,
                    false // isSrgb
                );
            }
            
            if (this.viewerState.baseColorBeziersEnabled) {
                this.depthAwareBlurManager.blur(
                    commandEncoder,
                    optimTargetView,
                    this.optimDepthTextureView!,
                    this.optimNormalTextureView!,
                    this.optimDepthAwareBlurredTextureView!,
                    this.optimWidth,
                    this.optimHeight,
                    15
                );
            }

            // 2. Run edge detection on optim-res depth (always use sharp for beziers)
            this.splatOptimizerManager.setEdgeTarget(
                this.optimDepthTextureView!, 
                this.optimEdgeTextureView!
            );
            this.splatOptimizerManager.dispatchEdge(commandEncoder, this.optimWidth, this.optimHeight);

            // 3. Dispatch Splat Optimizer Compute Passes (uses optim-res texture + edge map)
            this.splatOptimizerManager.setBackwardTarget(
                this.viewerState.compareBlurred ? this.optimBlurredTextureView! : optimTargetView,
                this.viewerState.compareBlurred ? this.optimBlurredDepthTextureView! : this.optimDepthTextureView!,
                this.optimWidth,
                this.optimHeight
            );
            const defaultPause = this.viewerState.renderMode === 'animation' && (!this.viewerState.turntableTraining || !this.viewerState.multiviewDatasetReady);
            if (this.viewerState.splatsEnabled && !(this.viewerState.splatTrainingPaused || defaultPause)) {
                this.splatOptimizerManager.dispatch(commandEncoder);
            }

            // 3.1 Sort splats by depth for correct alpha blending order
            if (this.viewerState.splatsEnabled) {
                this.splatOptimizerManager.dispatchSort(commandEncoder, sortVp);
            }

            // 3.1b Render current splats at optim-res to use as background for color beziers.
            this.splatForwardManager.setTarget(
                this.optimSplatTextureView!,
                this.optimSplatDepthTextureView!,
                this.optimWidth,
                this.optimHeight
            );
            this.splatForwardManager.dispatch(commandEncoder, true, this.viewerState.splatsEnabled);

            // 3.2 Restore full-res target for visualization later
            if (!this.viewerState.viewportRenderingFrozen) {
                this.splatForwardManager.setTarget(this.fullSplatTextureView!, this.fullSplatDepthTextureView!, fullW, fullH);
            }

            // 3b. Train the bezier edge layer: its target is the freshly-computed
            // edge texture, so the curves learn to trace the depth silhouette.
            if (this.viewerState.edgeBeziersEnabled) {
                if (!(this.viewerState.edgeBezierTrainingPaused || defaultPause)) {
                    this.edgeLayerBezierManager.dispatch(commandEncoder);
                }
                this.edgeLayerBezierManager.dispatchSort(commandEncoder, sortVp);
            }

            // Train base color beziers against depth-aware blurred target
            if (this.viewerState.baseColorBeziersEnabled) {
                // Background is pure splats
                this.baseColorLayerBezierManager.setBackwardTarget(
                    this.optimDepthAwareBlurredTextureView!,
                    this.optimDepthTextureView!,
                    this.optimSplatTextureView!,
                    this.optimSplatDepthTextureView!,
                    this.optimTextureView!,
                    this.optimWidth,
                    this.optimHeight,
                );
                if (!(this.viewerState.baseColorBezierTrainingPaused || defaultPause)) {
                    this.baseColorLayerBezierManager.dispatch(commandEncoder);
                }
                this.baseColorLayerBezierManager.dispatchSort(commandEncoder, sortVp);

                // Render base color beziers into optimSplatTextureView (loadOp: "load")
                // This makes it the background for the NEXT layer!
                this.baseColorBezierForwardManager.setTarget(this.optimSplatTextureView!, this.optimWidth, this.optimHeight);
                this.baseColorBezierForwardManager.dispatch(commandEncoder, false);
            }

            // Train sharp color beziers against sharp target
            if (this.viewerState.colorBeziersEnabled) {
                // Background is now splats OR splats+base (if base was enabled)
                this.colorLayerBezierManager.setBackwardTarget(
                    optimTargetView,
                    this.optimDepthTextureView!,
                    this.optimSplatTextureView!,
                    this.optimSplatDepthTextureView!,
                    this.optimTextureView!,
                    this.optimWidth,
                    this.optimHeight,
                );
                if (!(this.viewerState.colorBezierTrainingPaused || defaultPause)) {
                    this.colorLayerBezierManager.dispatch(commandEncoder);
                }
                this.colorLayerBezierManager.dispatchSort(commandEncoder, sortVp);
            }

            // 4. Run edge detection on full-res depth (for display)
            if (!this.viewerState.viewportRenderingFrozen) {
                this.splatOptimizerManager.setEdgeTarget(this.targetDepthTextureView!, this.fullEdgeTextureView!);
                this.splatOptimizerManager.dispatchEdge(commandEncoder, fullW, fullH);
                // Restore optim-res edge bind group for next frame
                this.splatOptimizerManager.setEdgeTarget(this.optimDepthTextureView!, this.optimEdgeTextureView!);

                // 4.5. Compute views into textures
                this.splatForwardManager.dispatch(commandEncoder, true, this.viewerState.splatsEnabled);
                if (this.viewerState.edgeBeziersEnabled) {
                    this.bezierForwardManager.dispatch(commandEncoder, true);
                }
                if (this.viewerState.baseColorBeziersEnabled) {
                    // For the full-res visualizer, we just want the base layer isolated, not drawn over splats
                    this.baseColorBezierForwardManager.setTarget(this.fullBaseColorBezierTextureView!, fullW, fullH);
                    this.baseColorBezierForwardManager.dispatch(commandEncoder, true);
                }
                if (this.viewerState.colorBeziersEnabled) {
                    this.colorBezierForwardManager.dispatch(commandEncoder, true);
                }
            }

            // 5. Render Splat Visualization to Screen View (uses full-res textures)
            const ptView = this.pathTracePipelineManager.outputTextureView ?? this.dummyTextureView!;
            this.splatOptimizerManager.setRenderTarget(
                this.targetTextureView!,
                this.fullSplatTextureView!,
                this.targetDepthTextureView!,
                this.fullEdgeTextureView!,
                this.fullBezierTextureView!,
                this.fullBaseColorBezierTextureView!,
                this.fullColorBezierTextureView!,
                ptView,
            );

            const screenView = currentTexture.createView();
            const finalPassEncoder = commandEncoder.beginRenderPass({
                label: "final render pass",
                colorAttachments: [
                    {
                        clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1.0 },
                        loadOp: "clear",
                        storeOp: "store",
                        view: screenView,
                    },
                ],
            });
            this.splatOptimizerManager.addDraw(finalPassEncoder);
            finalPassEncoder.end();

            this.device.queue.submit([commandEncoder.finish()]);

            if (canceled) return;

            if (this.capturePromise) {
                const { resolve, reject } = this.capturePromise;
                this.capturePromise = null;

                const texture = currentTexture;
                const width = texture.width;
                const height = texture.height;

                readTextureToImageData(this.device, texture, width, height, this.format)
                    .then(imageDataToBlob)
                    .then(resolve)
                    .catch(reject);
            }

            // Turntable frame capture — read back composited layers
            if (this.turntableFrameRequest) {
                const { resolve, reject } = this.turntableFrameRequest;
                this.turntableFrameRequest = null;

                (async () => {
                    try {
                        const w = this.fullWidth;
                        const h = this.fullHeight;
                        if (!w || !h || !this.fullSplatTexture) {
                            reject(new Error("Textures not ready"));
                            return;
                        }

                        // Read back the layers we need
                        const splat = await this.readTextureToImageData(this.fullSplatTexture!, w, h);
                        const baseColorBezier = this.viewerState.baseColorBeziersEnabled && this.fullBaseColorBezierTexture
                            ? await this.readTextureToImageData(this.fullBaseColorBezierTexture, w, h)
                            : null;
                        const colorBezier = this.viewerState.colorBeziersEnabled && this.fullColorBezierTexture
                            ? await this.readTextureToImageData(this.fullColorBezierTexture, w, h)
                            : null;
                        const edgeBezier = this.viewerState.edgeBeziersEnabled && this.fullBezierTexture
                            ? await this.readTextureToImageData(this.fullBezierTexture, w, h)
                            : null;

                        // Composite: same logic as splat_render.wgsl right-half
                        const result = new ImageData(w, h);
                        for (let i = 0; i < w * h; i++) {
                            const o = i * 4;
                            let r = splat.data[o] / 255;
                            let g = splat.data[o + 1] / 255;
                            let b = splat.data[o + 2] / 255;

                            // Base color bezier: premultiplied alpha over
                            if (baseColorBezier) {
                                const ba = baseColorBezier.data[o + 3] / 255;
                                r = r * (1 - ba) + baseColorBezier.data[o] / 255;
                                g = g * (1 - ba) + baseColorBezier.data[o + 1] / 255;
                                b = b * (1 - ba) + baseColorBezier.data[o + 2] / 255;
                            }

                            // Color bezier: premultiplied alpha over
                            if (colorBezier) {
                                const ca = colorBezier.data[o + 3] / 255;
                                r = r * (1 - ca) + colorBezier.data[o] / 255;
                                g = g * (1 - ca) + colorBezier.data[o + 1] / 255;
                                b = b * (1 - ca) + colorBezier.data[o + 2] / 255;
                            }

                            // Edge bezier: grayscale mix toward white
                            if (edgeBezier) {
                                const e = Math.min(1, edgeBezier.data[o] / 255);
                                r = r + (1 - r) * e;
                                g = g + (1 - g) * e;
                                b = b + (1 - b) * e;
                            }

                            result.data[o] = Math.min(255, Math.round(r * 255));
                            result.data[o + 1] = Math.min(255, Math.round(g * 255));
                            result.data[o + 2] = Math.min(255, Math.round(b * 255));
                            result.data[o + 3] = 255;
                        }
                        resolve(result);
                    } catch (e) {
                        reject(e as Error);
                    }
                })();
            }

            handle = requestAnimationFrame(loop);
        };

        handle = requestAnimationFrame(loop);

        return () => {
            cancelAnimationFrame(handle);
            canceled = true;
        };
    }
}