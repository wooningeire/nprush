import type { Camera } from "./Camera.svelte.ts";
import { GpuUniformsBufferManager } from "$/gpu/GpuUniformsBufferManager.ts";
import { GpuMeshRenderPipelineManager, MESH_DEPTH_FORMAT } from "$/gpu/GpuMeshRenderPipelineManager.ts";
import { GpuSplatOptimizerManager } from "../gpu/splat/GpuSplatOptimizerManager.ts";
import { GpuBezierOptimizerManager } from "../gpu/bezier/GpuBezierOptimizerManager.ts";
import { GpuSplatForwardPipelineManager } from "../gpu/splat/GpuSplatForwardPipelineManager.ts";
import { GpuBezierForwardPipelineManager } from "../gpu/bezier/GpuBezierForwardPipelineManager.ts";
import { GpuBlurPipelineManager } from "../gpu/blur/GpuBlurPipelineManager.ts";
import { GpuDepthAwareBlurPipelineManager } from "../gpu/blur/GpuDepthAwareBlurPipelineManager.ts";
import { GpuEnvmapPipelineManager } from "../gpu/envmap/GpuEnvmapPipelineManager.ts";
import { GpuPathTracePipelineManager } from "../gpu/pathtrace/GpuPathTracePipelineManager.ts";
import type { MeshData } from "../gpu/file-load/loadGlb.ts";
import type { ViewerState } from "./ViewerState.svelte.ts";
import { RENDER_MODE_MULTIVIEW } from "./renderMode.ts";
import { TurntableController } from "./turntable/TurntableController.ts";
import { GpuPerformanceMeasurementBufferManager } from "$/gpu/performanceMeasurement/GpuPerformanceMeasurementBufferManager.ts";
import { GpuProfilingPair, GPU_PROFILER_PAIR_COUNT } from "$/gpu/performanceMeasurement/gpuProfilerPairs.ts";
import { vec3, type Mat4 } from "wgpu-matrix";
import { constants } from "$/gpu/constants.ts";
import { computeOptimTextureSize } from "$/gpu/optimTextureSize.ts";
import { readTextureToImageData, imageDataToBlob } from "$/gpu/file-save/readback.ts";

const OPTIM_SHORT = constants.OPTIM_SHORT;

const NUM_EDGE_LAYER_BEZIERS = constants.NUM_EDGE_LAYER_BEZIERS;


export class GpuRunner {
    private readonly device: GPUDevice;
    private readonly format: GPUTextureFormat;
    private readonly camera: Camera;
    private readonly viewerState: ViewerState;
    private readonly contexts: Record<string, GPUCanvasContext>;

    readonly uniformsManager: GpuUniformsBufferManager;
    readonly meshRenderPipelineManager: GpuMeshRenderPipelineManager;
    readonly splatOptimizerManager: GpuSplatOptimizerManager;
    // The edge layer is a separate optimizer of cubic bezier curves trained
    // against the depth-edge texture. Curves natively represent 1D contours,
    // which is a much better fit for the silhouette target than gaussians.
    readonly edgeLayerBezierManager: GpuBezierOptimizerManager;
    readonly coarseColorLayerBezierManager: GpuBezierOptimizerManager;
    readonly fineColorLayerBezierManager: GpuBezierOptimizerManager;
    readonly splatForwardManager: GpuSplatForwardPipelineManager;
    readonly bezierForwardManager: GpuBezierForwardPipelineManager;
    readonly baseColorBezierForwardManager: GpuBezierForwardPipelineManager;
    readonly colorBezierForwardManager: GpuBezierForwardPipelineManager;
    private readonly blurManager: GpuBlurPipelineManager;
    private readonly depthAwareBlurManager: GpuDepthAwareBlurPipelineManager;
    private readonly matcapTextureView: GPUTextureView;
    private readonly envmapPipelineManager: GpuEnvmapPipelineManager;
    readonly pathTracePipelineManager: GpuPathTracePipelineManager;
    private gpuPerfBuffers: GpuPerformanceMeasurementBufferManager | null = null;

    // Full-res textures (sized to the visible main panel area: half-width x height-minus-strip).
    // These match the camera projection aspect so the rendered model has the same pixel
    // proportions as a square-rendered version (no horizontal/vertical squash).
    //
    // Note: targetDepthTexture is an R16Float depth (linear view-space depth remapped via
    // 1 - DEPTH_NEAR/d, used by Sobel for edge detection). It is NOT the hardware Z-buffer.
    // The hardware Z-buffer is targetZTexture below; without it, triangles draw in submission
    // order which causes back-face leakage on overlapping geometry.
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
    private targetTempTexture: GPUTexture | null = null;
    private fullWidth = 0;
    private fullHeight = 0;

    // Optim-res textures (aspect-matched to half-screen)
    private splatTargetColorTexture: GPUTexture | null = null;
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
    readonly turntable: TurntableController;

    readonly destroy: () => void;

    constructor({
        device,
        contexts,
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
        gpuTimestampSupported,
    }: {
        device: GPUDevice,
        contexts: Record<string, GPUCanvasContext>,
        format: GPUTextureFormat,
        camera: Camera,
        viewerState: ViewerState,
        mesh: MeshData,
        groundMesh: MeshData | null,
        groundPbrMesh: MeshData | null,
        matcapTexture: GPUTexture,
        brushTexture: GPUTexture,
        groundAlbedoTexture?: GPUTexture,
        groundNormalTexture?: GPUTexture,
        gpuTimestampSupported: boolean,
    }) {
        this.device = device;
        this.contexts = contexts;
        this.format = format;
        this.camera = camera;
        this.viewerState = viewerState;
        this.matcapTextureView = matcapTexture.createView();

        this.gpuPerfBuffers = gpuTimestampSupported
            ? new GpuPerformanceMeasurementBufferManager({ device })
            : null;

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
            numSplats: constants.NUM_GAUSSIAN_SPLATS,
        });

        this.edgeLayerBezierManager = new GpuBezierOptimizerManager({
            device,
            numBeziers: NUM_EDGE_LAYER_BEZIERS,
        });

        this.coarseColorLayerBezierManager = new GpuBezierOptimizerManager({
            device,
            numBeziers: NUM_EDGE_LAYER_BEZIERS,
        });

        this.fineColorLayerBezierManager = new GpuBezierOptimizerManager({
            device,
            numBeziers: NUM_EDGE_LAYER_BEZIERS,
        });

        this.splatForwardManager = new GpuSplatForwardPipelineManager({
            device,
            numSplats: constants.NUM_GAUSSIAN_SPLATS,
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
            bezierBuffer: this.coarseColorLayerBezierManager.bezierBuffer,
            sortOrderBuffer: this.coarseColorLayerBezierManager.sortIndicesBuffer,
            brushTexture,
        });

        this.colorBezierForwardManager = new GpuBezierForwardPipelineManager({
            device,
            numBeziers: NUM_EDGE_LAYER_BEZIERS,
            bezierBuffer: this.fineColorLayerBezierManager.bezierBuffer,
            sortOrderBuffer: this.fineColorLayerBezierManager.sortIndicesBuffer,
            brushTexture,
        });
        
        this.blurManager = new GpuBlurPipelineManager(device);
        this.depthAwareBlurManager = new GpuDepthAwareBlurPipelineManager(device);

        this.turntable = new TurntableController({
            device,
            camera,
            viewerState,
            managers: {
                uniformsManager: this.uniformsManager,
                pathTracePipelineManager: this.pathTracePipelineManager,
                splatOptimizerManager: this.splatOptimizerManager,
                edgeLayerBezierManager: this.edgeLayerBezierManager,
                coarseColorLayerBezierManager: this.coarseColorLayerBezierManager,
                fineColorLayerBezierManager: this.fineColorLayerBezierManager,
                bezierForwardManager: this.bezierForwardManager,
                baseColorBezierForwardManager: this.baseColorBezierForwardManager,
                colorBezierForwardManager: this.colorBezierForwardManager,
            },
        });

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
                this.viewerState.coarseColorBeziersEnabled,
                this.viewerState.fineColorBeziersEnabled,
                this.viewerState.meshSplatsEnabled,
                this.viewerState.splatsEnabled,
                this.getCanvasAspects()
            ));
            /** Splat view-projection + eye position (for SH) are written each frame inside loop(). */
            $effect(() => {
                // Reset Adam on camera change for ordinary navigation. Multiview uses
                // dataset views without moving the orbit. Single-view turntable export
                // moves the camera each frame but should accumulate optimizer state.
                this.camera.viewProjMat;
                if (this.viewerState.isTurntableRendering) return;
                this.splatOptimizerManager.resetAdam();
                this.edgeLayerBezierManager.resetAdam();
                this.coarseColorLayerBezierManager.resetAdam();
                this.fineColorLayerBezierManager.resetAdam();
            });
            /** Bezier optimizers skip multiview; turntable+dataset assigns their VP inside loop(). */
            $effect(() => {
                if (this.viewerState.renderMode !== RENDER_MODE_MULTIVIEW) {
                    this.edgeLayerBezierManager.writeVPMatrix(this.camera.viewProjMat);
                    this.coarseColorLayerBezierManager.writeVPMatrix(this.camera.viewProjMat);
                    this.fineColorLayerBezierManager.writeVPMatrix(this.camera.viewProjMat);
                    this.edgeLayerBezierManager.writeVPInvMatrix(this.camera.viewProjInvMat);
                    this.coarseColorLayerBezierManager.writeVPInvMatrix(this.camera.viewProjInvMat);
                    this.fineColorLayerBezierManager.writeVPInvMatrix(this.camera.viewProjInvMat);
                }
            });
            $effect(() => this.bezierForwardManager.writeVPMatrix(this.camera.viewProjMat));
            $effect(() => this.baseColorBezierForwardManager.writeVPMatrix(this.camera.viewProjMat));
            $effect(() => this.colorBezierForwardManager.writeVPMatrix(this.camera.viewProjMat));
            $effect(() => {
                // mode=2: coverage loss (positions on edges) + color loss from normalTex weighted by edge strength
                this.edgeLayerBezierManager.writeMode(2);
                this.edgeLayerBezierManager.writeMaxWidth(0.005);
                this.edgeLayerBezierManager.writeKillThresholds(0.0001, 0.0001);
                this.edgeLayerBezierManager.writeBgPenalty(0.0);
                this.coarseColorLayerBezierManager.writeMode(1); // Color+Depth mode
                this.fineColorLayerBezierManager.writeMode(1); // Color+Depth mode
                this.fineColorLayerBezierManager.writeMaxWidth(0.005); // finer strokes on fine bezier layer
                // Fine bezier layer: less aggressive killing so thin strokes survive,
                // but background penalty enabled to kill off-model curves.
                this.fineColorLayerBezierManager.writeKillThresholds(0.0001, 0.0001);
                this.fineColorLayerBezierManager.writeBgPenalty(0.0);
                // Coarse bezier layer: no background penalty (blurred target bleeds into bg).
                // Multiview/turntable sets adam no_kill separately (below) so off-frustum kills
                // don't remove curves visible from other views. Longer ADC period here dampens
                // clone/kill churn in ordinary single-view orbit where no_kill stays off.
                this.coarseColorLayerBezierManager.writeMaxWidth(0.1);
                this.coarseColorLayerBezierManager.writeKillThresholds(0.0001, 0.0001);
                this.coarseColorLayerBezierManager.setAdcPeriod(150);
            });
            $effect(() => {
                // Multiview rotates the effective view each frame; single-view turntable
                // export also moves the camera. Skip step + ADC kills that treat
                // "off this view's frustum" as dead so curves survive for other angles.
                const noKillMv =
                    this.viewerState.renderMode === RENDER_MODE_MULTIVIEW || this.viewerState.isTurntableRendering;
                this.splatOptimizerManager.writeNoKill(noKillMv);
                this.edgeLayerBezierManager.writeNoKill(noKillMv);
                this.coarseColorLayerBezierManager.writeNoKill(noKillMv);
                this.fineColorLayerBezierManager.writeNoKill(noKillMv);
            });

            return () => {
                // Cleanup managers
                this.uniformsManager.destroy();
                this.meshRenderPipelineManager.destroy();
                this.splatOptimizerManager.destroy();
                this.edgeLayerBezierManager.destroy();
                this.coarseColorLayerBezierManager.destroy();
                this.fineColorLayerBezierManager.destroy();
                this.splatForwardManager.destroy();
                this.bezierForwardManager.destroy();
                this.baseColorBezierForwardManager.destroy();
                this.colorBezierForwardManager.destroy();
                this.blurManager.destroy();
                this.depthAwareBlurManager.destroy();
                this.pathTracePipelineManager.destroy();
                this.gpuPerfBuffers?.destroy();

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

                this.splatTargetColorTexture?.destroy();
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

                this.turntable.destroy();
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
        return this.turntable.captureTurntableFrame();
    }

    replaceMesh(mesh: MeshData) {
        this.meshRenderPipelineManager.replaceMesh(mesh);
        const ptMeshes: MeshData[] = [mesh];
        this.pathTracePipelineManager.setMeshes(ptMeshes);
        this.pathTracePipelineManager.reset();
    }

    async prerenderDataset(): Promise<void> {
        // Wait until optim textures are ready (loop may not have run yet).
        while (!this.optimTextureView || this.optimWidth === 0) {
            await new Promise<void>(r => requestAnimationFrame(() => r()));
        }
        return this.turntable.prerenderDataset(this.optimWidth, this.optimHeight);
    }

    private recreateOptimizationTextures(panelAspect: number) {
        // Size optim textures to match the visible panel aspect ratio so the model
        // rendered into them has matching pixel proportions for the gradient pass.
        const { width: optimizationTargetWidth, height: optimizationTargetHeight } = computeOptimTextureSize(OPTIM_SHORT, panelAspect);

        if (optimizationTargetWidth === this.optimWidth && optimizationTargetHeight === this.optimHeight) return;
        this.optimWidth = optimizationTargetWidth;
        this.optimHeight = optimizationTargetHeight;

        // Resize path tracer output to match optim resolution
        this.pathTracePipelineManager.setOutputSize(optimizationTargetWidth, optimizationTargetHeight);

        if (this.splatTargetColorTexture) this.splatTargetColorTexture.destroy();
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

        this.splatTargetColorTexture = this.device.createTexture({
            label: "optimization target texture",
            size: [optimizationTargetWidth, optimizationTargetHeight],
            format: this.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimTextureView = this.splatTargetColorTexture.createView();
        
        this.optimNormalTexture = this.device.createTexture({
            label: "optimization normal texture",
            size: [optimizationTargetWidth, optimizationTargetHeight],
            format: this.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimNormalTextureView = this.optimNormalTexture.createView();

        this.optimDepthTexture = this.device.createTexture({
            label: "optimization depth visualization",
            size: [optimizationTargetWidth, optimizationTargetHeight],
            format: "r16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimDepthTextureView = this.optimDepthTexture.createView();

        this.optimZTexture = this.device.createTexture({
            label: "optimization z-buffer",
            size: [optimizationTargetWidth, optimizationTargetHeight],
            format: MESH_DEPTH_FORMAT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.optimZTextureView = this.optimZTexture.createView();

        this.optimEdgeTexture = this.device.createTexture({
            label: "optimization edge map",
            size: [optimizationTargetWidth, optimizationTargetHeight],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimEdgeTextureView = this.optimEdgeTexture.createView();

        this.optimSplatTexture = this.device.createTexture({
            label: "optimization splat view",
            size: [optimizationTargetWidth, optimizationTargetHeight],
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimSplatTextureView = this.optimSplatTexture.createView();

        this.optimSplatDepthTexture = this.device.createTexture({
            label: "optimization splat depth",
            size: [optimizationTargetWidth, optimizationTargetHeight],
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimSplatDepthTextureView = this.optimSplatDepthTexture.createView();

        this.optimBlurredTexture = this.device.createTexture({
            label: "optimization blurred target",
            size: [optimizationTargetWidth, optimizationTargetHeight],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimBlurredTextureView = this.optimBlurredTexture.createView();

        this.optimDepthAwareBlurredTexture = this.device.createTexture({
            label: "optimization depth-aware blurred target",
            size: [optimizationTargetWidth, optimizationTargetHeight],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimDepthAwareBlurredTextureView = this.optimDepthAwareBlurredTexture.createView();

        this.optimBlurredDepthTexture = this.device.createTexture({
            label: "optimization blurred depth",
            size: [optimizationTargetWidth, optimizationTargetHeight],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimBlurredDepthTextureView = this.optimBlurredDepthTexture.createView();

        this.optimTempTexture = this.device.createTexture({
            label: "optimization blur temp",
            size: [optimizationTargetWidth, optimizationTargetHeight],
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
        this.splatOptimizerManager.setEdgeTarget(this.optimDepthTextureView, this.optimEdgeTextureView, this.optimNormalTextureView);
        this.splatOptimizerManager.setBackwardTarget(this.optimTextureView, this.optimDepthTextureView, optimizationTargetWidth, optimizationTargetHeight);

        // Edge layer: color target = edge map, depth = real depth, background = black (dummy).
        // Mode=1 color loss drives beziers white on edges, transparent off edges.
        this.edgeLayerBezierManager.setBackwardTarget(
            this.optimEdgeTextureView,
            this.optimDepthTextureView,
            this.dummyTextureView!,
            this.optimTextureView!,
            optimizationTargetWidth,
            optimizationTargetHeight,
        );

        this.coarseColorLayerBezierManager.setBackwardTarget(
            this.optimDepthAwareBlurredTextureView!,
            this.optimDepthTextureView!,
            this.optimSplatTextureView!,
            this.optimTextureView!,
            optimizationTargetWidth, optimizationTargetHeight
        );

        this.fineColorLayerBezierManager.setBackwardTarget(
            this.optimTextureView!,
            this.optimDepthTextureView!,
            this.optimSplatTextureView!,
            this.optimTextureView!,
            optimizationTargetWidth, optimizationTargetHeight
        );
    }

    loop() {
        let handle = 0;
        let canceled = false;

        const loop = async () => {
            this.recreateOptimizationTextures(1);

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
                    format: "r16float",
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
                    label: "full-res coarse bezier view",
                    size: [fullW, fullH],
                    format: "rgba8unorm",
                    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
                });
                this.fullBaseColorBezierTextureView = this.fullBaseColorBezierTexture.createView();

                this.fullColorBezierTexture = this.device.createTexture({
                    label: "full-res fine bezier view",
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

                this.targetTempTexture = this.device.createTexture({
                    label: "full-res blur temp",
                    size: [fullW, fullH],
                    format: "rgba8unorm",
                    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
                });

                this.splatForwardManager.setTarget(this.fullSplatTextureView, this.fullSplatDepthTextureView!, fullW, fullH);
                this.bezierForwardManager.setTarget(this.fullBezierTextureView!, fullW, fullH);
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
                if (!canceled) requestAnimationFrame(() => void loop());
                return;
            }

            // In animation mode, we might randomize camera each frame
            if (this.viewerState.renderMode === RENDER_MODE_MULTIVIEW) {
                this.viewerState.tickAnimationMode();
            }

            const commandEncoder = this.device.createCommandEncoder({
                label: "runner loop command encoder",
            });

            const activeProfilerIndices = new Set<number>();
            const recordGpu = !!(this.gpuPerfBuffers && this.viewerState.gpuProfilingEnabled);
            const profWrites = (idx: number) => {
                if (recordGpu) activeProfilerIndices.add(idx);
                return recordGpu ? this.gpuPerfBuffers!.writes(idx) : undefined;
            };

            // Frozen viewport skips full-resolution splat/bezier passes unless we must
            // refresh those textures for turntable PNG export (readback next enqueue).
            const needsTurntableExportLayers = this.turntable.hasPendingCapture();

            // Resolve the current frame's view (dataset slot or live camera).
            const frameView = this.turntable.resolveFrameView();
            const { datasetView, sortVp, vpInv: vpInvForSplat, invView: invViewForCam } = frameView;

            const camWorld = vec3.transformMat4(vec3.fromValues(0, 0, 0), invViewForCam);
            this.splatOptimizerManager.writeSplatVPMatrix(sortVp, vpInvForSplat, this.viewerState.compareBlurred, [
                camWorld[0],
                camWorld[1],
                camWorld[2],
            ]);
            this.splatForwardManager.writeVPMatrix(sortVp);
            this.splatForwardManager.writeCameraWorld(camWorld[0], camWorld[1], camWorld[2]);
            this.bezierForwardManager.writeVPMatrix(sortVp);
            this.baseColorBezierForwardManager.writeVPMatrix(sortVp);
            this.colorBezierForwardManager.writeVPMatrix(sortVp);

            this.edgeLayerBezierManager.writeCamWorld(camWorld[0], camWorld[1], camWorld[2]);
            this.coarseColorLayerBezierManager.writeCamWorld(camWorld[0], camWorld[1], camWorld[2]);
            this.fineColorLayerBezierManager.writeCamWorld(camWorld[0], camWorld[1], camWorld[2]);
            const cx = camWorld[0];
            const cy = camWorld[1];
            const cz = camWorld[2];
            this.bezierForwardManager.writeCameraWorld(cx, cy, cz);
            this.baseColorBezierForwardManager.writeCameraWorld(cx, cy, cz);
            this.colorBezierForwardManager.writeCameraWorld(cx, cy, cz);

            // 1a. Render the model into the full-res target + depth textures (for visualization).
            if (!this.viewerState.viewportRenderingFrozen) {
                const spherePassEncoder = commandEncoder.beginRenderPass({
                    label: "mesh render pass (full res)",
                    ...(recordGpu ? { timestampWrites: profWrites(GpuProfilingPair.MeshFullRaster) } : {}),
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
                ...(recordGpu ? { timestampWrites: profWrites(GpuProfilingPair.MeshOptimRaster) } : {}),
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
                this.pathTracePipelineManager.addDispatches(
                    commandEncoder,
                    profWrites(GpuProfilingPair.PathTrace),
                );
            }

            // Use dataset view if available, else PT output, else raster fallback.
            const ptOutputView = this.pathTracePipelineManager.outputTextureView;
            const optimTargetView = datasetView ?? ptOutputView ?? this.optimTextureView!;

            // 1c. Run separable blur on targets if enabled
            if (this.viewerState.compareBlurred) {
                this.blurManager.addDispatches(
                    commandEncoder,
                    optimTargetView,
                    this.optimBlurredTextureView!,
                    this.optimTempTextureView!,
                    this.optimWidth,
                    this.optimHeight,
                    this.viewerState.blurRadius,
                    this.viewerState.blurRadius / 2,
                    true, // isSrgb
                    profWrites(GpuProfilingPair.BlurOptimTarget),
                );
                this.blurManager.addDispatches(
                    commandEncoder,
                    this.optimDepthTextureView!,
                    this.optimBlurredDepthTextureView!,
                    this.optimTempTextureView!,
                    this.optimWidth,
                    this.optimHeight,
                    this.viewerState.blurRadius,
                    this.viewerState.blurRadius / 2,
                    false, // isSrgb
                    profWrites(GpuProfilingPair.BlurOptimDepth),
                );
            }
            
            if (this.viewerState.coarseColorBeziersEnabled) {
                this.depthAwareBlurManager.addDispatches(
                    commandEncoder,
                    optimTargetView,
                    this.optimDepthTextureView!,
                    this.optimNormalTextureView!,
                    this.optimDepthAwareBlurredTextureView!,
                    this.optimWidth,
                    this.optimHeight,
                    15,
                    profWrites(GpuProfilingPair.DepthAwareBlur),
                );
            }
            
            // Update backward targets for all optimizers to point to current frame's target view
            this.splatOptimizerManager.setBackwardTarget(
                this.viewerState.compareBlurred ? this.optimBlurredTextureView! : optimTargetView,
                this.viewerState.compareBlurred ? this.optimBlurredDepthTextureView! : this.optimDepthTextureView!,
                this.optimWidth,
                this.optimHeight
            );

            if (this.viewerState.coarseColorBeziersEnabled) {
                this.coarseColorLayerBezierManager.setBackwardTarget(
                    this.optimDepthAwareBlurredTextureView!,
                    this.optimDepthTextureView!,
                    this.optimSplatTextureView!,
                    this.optimNormalTextureView!,
                    this.optimWidth,
                    this.optimHeight,
                );
            }
            if (this.viewerState.fineColorBeziersEnabled) {
                this.fineColorLayerBezierManager.setBackwardTarget(
                    optimTargetView,
                    this.optimDepthTextureView!,
                    this.optimSplatTextureView!,
                    this.optimNormalTextureView!,
                    this.optimWidth,
                    this.optimHeight,
                );
            }
            if (this.viewerState.edgeBeziersEnabled) {
                this.edgeLayerBezierManager.setBackwardTarget(
                    this.optimEdgeTextureView!,
                    this.optimDepthTextureView!,
                    this.dummyTextureView!,
                    this.optimNormalTextureView!,
                    this.optimWidth,
                    this.optimHeight,
                );
            }

            // 2. Optimization Pass (Compute)
            const defaultPause = this.viewerState.renderMode === RENDER_MODE_MULTIVIEW && (!this.viewerState.turntableTraining || !this.viewerState.multiviewDatasetReady);
            
            // Clear all binning and sorting buffers
            this.splatOptimizerManager.addBinningDispatches(null as any, sortVp, commandEncoder);
            if (this.viewerState.coarseColorBeziersEnabled) this.coarseColorLayerBezierManager.addBinningDispatches(null as any, sortVp, commandEncoder);
            if (this.viewerState.fineColorBeziersEnabled) this.fineColorLayerBezierManager.addBinningDispatches(null as any, sortVp, commandEncoder);
            if (this.viewerState.edgeBeziersEnabled) this.edgeLayerBezierManager.addBinningDispatches(null as any, sortVp, commandEncoder);

            // Edge detection (optim res)
            const edgeOptimPass = commandEncoder.beginComputePass({
                label: "splat edge detection (optim res)",
                ...(recordGpu ? { timestampWrites: profWrites(GpuProfilingPair.SplatEdgeDetectOptim) } : {}),
            });
            this.splatOptimizerManager.addEdgeDispatches(edgeOptimPass, this.optimWidth, this.optimHeight);
            edgeOptimPass.end();

            // Splat optimization
            const splatPause = this.viewerState.splatTrainingPaused || defaultPause;
            if (this.viewerState.splatsEnabled && !splatPause) {
                const splatOptPass = commandEncoder.beginComputePass({
                    label: "splat optimization compute",
                    ...(recordGpu ? { timestampWrites: profWrites(GpuProfilingPair.SplatOptimization) } : {}),
                });
                this.splatOptimizerManager.addBinningDispatches(splatOptPass, sortVp, commandEncoder);
                this.splatOptimizerManager.addOptimizationDispatches(splatOptPass);
                splatOptPass.end();
            }
            if (this.viewerState.splatsEnabled) {
                const splatSortPass = commandEncoder.beginComputePass({ label: "splat depth sort" });
                this.splatOptimizerManager.addDepthSortDispatches(splatSortPass, sortVp);
                splatSortPass.end();
            }

            // Coarse Bezier optimization
            if (this.viewerState.coarseColorBeziersEnabled) {
                if (!(this.viewerState.coarseColorBezierTrainingPaused || defaultPause)) {
                    const coarseOptPass = commandEncoder.beginComputePass({
                        label: "coarse bezier optimization compute",
                        ...(recordGpu ? { timestampWrites: profWrites(GpuProfilingPair.BezierCoarseOptimization) } : {}),
                    });
                    this.coarseColorLayerBezierManager.addBinningDispatches(coarseOptPass, sortVp, commandEncoder);
                    this.coarseColorLayerBezierManager.addOptimizationDispatches(coarseOptPass);
                    coarseOptPass.end();
                }
                const coarseSortPass = commandEncoder.beginComputePass({ label: "coarse bezier sort" });
                this.coarseColorLayerBezierManager.addSortDispatches(coarseSortPass, sortVp);
                coarseSortPass.end();
            }

            // Fine Bezier optimization
            if (this.viewerState.fineColorBeziersEnabled) {
                if (!(this.viewerState.fineColorBezierTrainingPaused || defaultPause)) {
                    const fineOptPass = commandEncoder.beginComputePass({
                        label: "fine bezier optimization compute",
                        ...(recordGpu ? { timestampWrites: profWrites(GpuProfilingPair.BezierFineOptimization) } : {}),
                    });
                    this.fineColorLayerBezierManager.addBinningDispatches(fineOptPass, sortVp, commandEncoder);
                    this.fineColorLayerBezierManager.addOptimizationDispatches(fineOptPass);
                    fineOptPass.end();
                }
                const fineSortPass = commandEncoder.beginComputePass({ label: "fine bezier sort" });
                this.fineColorLayerBezierManager.addSortDispatches(fineSortPass, sortVp);
                fineSortPass.end();
            }

            // Edge Bezier optimization
            if (this.viewerState.edgeBeziersEnabled) {
                if (!(this.viewerState.edgeBezierTrainingPaused || defaultPause)) {
                    const edgeOptPass = commandEncoder.beginComputePass({
                        label: "edge bezier optimization compute",
                        ...(recordGpu ? { timestampWrites: profWrites(GpuProfilingPair.BezierEdgeOptimization) } : {}),
                    });
                    this.edgeLayerBezierManager.addBinningDispatches(edgeOptPass, sortVp, commandEncoder);
                    this.edgeLayerBezierManager.addOptimizationDispatches(edgeOptPass);
                    edgeOptPass.end();
                }
                const edgeSortPass = commandEncoder.beginComputePass({ label: "edge bezier sort" });
                this.edgeLayerBezierManager.addSortDispatches(edgeSortPass, sortVp);
                edgeSortPass.end();
            }

            if (!this.viewerState.viewportRenderingFrozen || needsTurntableExportLayers) {
                this.splatOptimizerManager.setEdgeTarget(this.targetDepthTextureView!, this.fullEdgeTextureView!, this.targetNormalTextureView!);

                const edgeFullPass = commandEncoder.beginComputePass({
                    label: "splat edge detection (full res)",
                    ...(recordGpu ? { timestampWrites: profWrites(GpuProfilingPair.EdgeDetectFull) } : {}),
                });
                this.splatOptimizerManager.addEdgeDispatches(edgeFullPass, fullW, fullH);
                edgeFullPass.end();
                // Reset target for next frame
                this.splatOptimizerManager.setEdgeTarget(this.optimDepthTextureView!, this.optimEdgeTextureView!, this.optimNormalTextureView!);
            }

            // 3. Render Pass (Optim-Res)
            // Group Splat and Coarse Bezier which target the same texture
            this.splatForwardManager.setTarget(this.optimSplatTextureView!, this.optimSplatDepthTextureView!, this.optimWidth, this.optimHeight);
            this.baseColorBezierForwardManager.setTarget(this.optimSplatTextureView!, this.optimWidth, this.optimHeight);

            const optimRenderPass = commandEncoder.beginRenderPass({
                label: "optim-res render pass",
                ...(recordGpu ? { timestampWrites: profWrites(GpuProfilingPair.SplatForwardOptim) } : {}),
                colorAttachments: [
                    {
                        view: this.optimSplatTextureView!,
                        clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1.0 },
                        loadOp: "clear",
                        storeOp: "store",
                    },
                    {
                        view: this.optimSplatDepthTextureView!,
                        clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
                        loadOp: "clear",
                        storeOp: "store",
                    },
                ],
            });

            this.splatForwardManager.render(optimRenderPass, this.viewerState.splatsEnabled);
            if (this.viewerState.coarseColorBeziersEnabled) {
                this.baseColorBezierForwardManager.render(optimRenderPass, true);
            }
            optimRenderPass.end();

            // 4. Full-res Visualization Renders
            if (!this.viewerState.viewportRenderingFrozen || needsTurntableExportLayers) {
                // Splat Full-res
                this.splatForwardManager.setTarget(this.fullSplatTextureView!, this.fullSplatDepthTextureView!, fullW, fullH);
                this.splatForwardManager.addDispatches(
                    commandEncoder, 
                    true, 
                    this.viewerState.splatsEnabled,
                    recordGpu ? profWrites(GpuProfilingPair.SplatForwardFull) : undefined
                );
                
                // Edge Bezier Full-res
                if (this.viewerState.edgeBeziersEnabled) {
                    this.bezierForwardManager.setTarget(this.fullBezierTextureView!, fullW, fullH);
                    this.bezierForwardManager.addDispatches(
                        commandEncoder, 
                        true,
                        recordGpu ? profWrites(GpuProfilingPair.BezierEdgeForwardFull) : undefined
                    );
                }
                
                // Coarse Bezier Full-res
                if (this.viewerState.coarseColorBeziersEnabled) {
                    this.baseColorBezierForwardManager.setTarget(this.fullBaseColorBezierTextureView!, fullW, fullH);
                    this.baseColorBezierForwardManager.addDispatches(
                        commandEncoder, 
                        true,
                        recordGpu ? profWrites(GpuProfilingPair.BezierCoarseForwardFull) : undefined
                    );
                }
                
                // Fine Bezier Full-res
                if (this.viewerState.fineColorBeziersEnabled) {
                    this.colorBezierForwardManager.setTarget(this.fullColorBezierTextureView!, fullW, fullH);
                    this.colorBezierForwardManager.addDispatches(
                        commandEncoder, 
                        true,
                        recordGpu ? profWrites(GpuProfilingPair.BezierFineForwardFull) : undefined
                    );
                }
            }

            // 5. Render Splat Visualization to Screen Views (uses full-res textures)
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

            const aspects = this.getCanvasAspects();
            this.splatOptimizerManager.writeRenderUniforms(
                this.viewerState.edgeBeziersEnabled,
                this.viewerState.coarseColorBeziersEnabled,
                this.viewerState.fineColorBeziersEnabled,
                this.viewerState.meshSplatsEnabled,
                this.viewerState.splatsEnabled,
                aspects
            );

            const PANEL_MODES: Record<string, number> = {
                target: 0,
                splats: 1,
                splatColor: 2,
                targetDepth: 3,
                targetEdges: 4,
                edgeBeziers: 5,
                coarseBezier: 6,
                fineBezier: 7,
            };

            for (const [id, ctx] of Object.entries(this.contexts)) {
                const mode = PANEL_MODES[id];
                if (mode === undefined) continue;

                const screenTex = ctx.getCurrentTexture();
                const screenView = screenTex.createView();
                
                const finalPassEncoder = commandEncoder.beginRenderPass({
                    label: `final render pass for ${id}`,
                    ...(recordGpu && id === "target" ? { timestampWrites: profWrites(GpuProfilingPair.FinalCompositor) } : {}),
                    colorAttachments: [
                        {
                            clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1.0 },
                            loadOp: "clear",
                            storeOp: "store",
                            view: screenView,
                        },
                    ],
                });
                this.splatOptimizerManager.addDraw(finalPassEncoder, mode);
                finalPassEncoder.end();
            }

            if (recordGpu && this.gpuPerfBuffers) {
                this.gpuPerfBuffers.addResolve(commandEncoder, activeProfilerIndices);
            }

            this.device.queue.submit([commandEncoder.finish()]);

            if (recordGpu && this.gpuPerfBuffers) {
                try {
                    await this.device.queue.onSubmittedWorkDone();
                    const deltasNs = await this.gpuPerfBuffers.mapDeltasNanoseconds(activeProfilerIndices);
                    const msArr = deltasNs.map(ns => ns === null ? null : Number(ns) / 1e6);
                    this.viewerState.setGpuProfilingFrameMs(msArr);
                } catch (e) {
                    console.warn("[gpu profiler]", e);
                }
            } else {
                // If profiling is disabled, wipe with nulls to avoid stale charts
                this.viewerState.setGpuProfilingFrameMs(Array(GPU_PROFILER_PAIR_COUNT).fill(null));
            }

            if (canceled) return;

            if (this.capturePromise) {
                const { resolve, reject } = this.capturePromise;
                this.capturePromise = null;

                const mainCtx = this.contexts["target"];
                if (mainCtx) {
                    const texture = mainCtx.getCurrentTexture();
                    const width = texture.width;
                    const height = texture.height;

                    readTextureToImageData(this.device, texture, width, height, this.format)
                        .then(imageDataToBlob)
                        .then(resolve)
                        .catch(reject);
                } else {
                    reject(new Error("Target canvas context not found"));
                }
            }

            this.turntable.resolvePendingCapture(
                this.fullWidth,
                this.fullHeight,
                this.fullSplatTexture,
                this.fullBaseColorBezierTexture,
                this.fullColorBezierTexture,
                this.fullBezierTexture,
            );

            handle = requestAnimationFrame(loop);
        };

        handle = requestAnimationFrame(loop);

        return () => {
            cancelAnimationFrame(handle);
            canceled = true;
        };
    }
    private getCanvasAspects(): Record<number, number> {
        const PANEL_MODES: Record<string, number> = {
            target: 0,
            splats: 1,
            splatColor: 2,
            targetDepth: 3,
            targetEdges: 4,
            edgeBeziers: 5,
            coarseBezier: 6,
            fineBezier: 7,
        };
        const aspects: Record<number, number> = {};
        for (const [id, context] of Object.entries(this.contexts)) {
            const canvas = context.canvas;
            const mode = PANEL_MODES[id];
            if (mode !== undefined && canvas) {
                aspects[mode] = canvas.width / canvas.height;
            }
        }
        return aspects;
    }
}
