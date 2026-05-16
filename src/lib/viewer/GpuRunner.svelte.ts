import type { Camera } from "./Camera.svelte.ts";
import { GpuUniformsBufferManager } from "$/gpu/GpuUniformsBufferManager.ts";
import { GpuMeshRenderPipelineManager } from "$/gpu/GpuMeshRenderPipelineManager.ts";
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
import { vec3 } from "wgpu-matrix";
import { constants } from "$/gpu/constants.ts";
import { readTextureToImageData, imageDataToBlob } from "$/gpu/file-save/readback.ts";
import { GpuTextureManager } from "./GpuTextureManager.ts";


export class GpuRunner {
    private readonly device: GPUDevice;
    private readonly format: GPUTextureFormat;
    private readonly viewportCamera: Camera;
    private readonly backendCamera: Camera;
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
    readonly edgeBezierForwardManager: GpuBezierForwardPipelineManager;
    readonly coarseColorBezierForwardManager: GpuBezierForwardPipelineManager;
    readonly fineColorBezierForwardManager: GpuBezierForwardPipelineManager;
    private readonly blurManager: GpuBlurPipelineManager;
    private readonly depthAwareBlurManager: GpuDepthAwareBlurPipelineManager;
    private readonly matcapTextureView: GPUTextureView;
    private readonly envmapPipelineManager: GpuEnvmapPipelineManager;
    readonly pathTracePipelineManager: GpuPathTracePipelineManager;
    private gpuPerfBuffers: GpuPerformanceMeasurementBufferManager | null = null;

    private readonly textures: GpuTextureManager;


    private capturePromise: { resolve: (blob: Blob) => void, reject: (err: Error) => void } | null = null;
    readonly turntable: TurntableController;

    readonly destroy: () => void;

    constructor({
        device,
        contexts,
        format,
        viewportCamera,
        backendCamera,
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
        viewportCamera: Camera,
        backendCamera: Camera,
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
        this.viewportCamera = viewportCamera;
        this.backendCamera = backendCamera;
        this.viewerState = viewerState;
        this.textures = new GpuTextureManager(device, format);
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
            numSplats: constants.N_GAUSSIAN_SPLATS,
        });

        this.edgeLayerBezierManager = new GpuBezierOptimizerManager({
            device,
            numBeziers: constants.N_EDGE_BEZIERS,
        });

        this.coarseColorLayerBezierManager = new GpuBezierOptimizerManager({
            device,
            numBeziers: constants.N_COARSE_COLOR_BEZIERS,
        });

        this.fineColorLayerBezierManager = new GpuBezierOptimizerManager({
            device,
            numBeziers: constants.N_FINE_COLOR_BEZIERS,
        });

        this.splatForwardManager = new GpuSplatForwardPipelineManager({
            device,
            numSplats: constants.N_GAUSSIAN_SPLATS,
            splatBuffer: this.splatOptimizerManager.splatBuffer,
            sortOrderBuffer: this.splatOptimizerManager.sortIndicesBuffer,
        });

        this.edgeBezierForwardManager = new GpuBezierForwardPipelineManager({
            device,
            numBeziers: constants.N_EDGE_BEZIERS,
            bezierBuffer: this.edgeLayerBezierManager.bezierBuffer,
            sortOrderBuffer: this.edgeLayerBezierManager.sortIndicesBuffer,
            brushTexture,
        });

        this.coarseColorBezierForwardManager = new GpuBezierForwardPipelineManager({
            device,
            numBeziers: constants.N_COARSE_COLOR_BEZIERS,
            bezierBuffer: this.coarseColorLayerBezierManager.bezierBuffer,
            sortOrderBuffer: this.coarseColorLayerBezierManager.sortIndicesBuffer,
            brushTexture,
        });

        this.fineColorBezierForwardManager = new GpuBezierForwardPipelineManager({
            device,
            numBeziers: constants.N_FINE_COLOR_BEZIERS,
            bezierBuffer: this.fineColorLayerBezierManager.bezierBuffer,
            sortOrderBuffer: this.fineColorLayerBezierManager.sortIndicesBuffer,
            brushTexture,
        });
        
        this.blurManager = new GpuBlurPipelineManager(device);
        this.depthAwareBlurManager = new GpuDepthAwareBlurPipelineManager(device);

        this.turntable = new TurntableController({
            device,
            viewportCamera,
            backendCamera,
            viewerState,
            managers: {
                uniformsManager: this.uniformsManager,
                pathTracePipelineManager: this.pathTracePipelineManager,
                splatOptimizerManager: this.splatOptimizerManager,
                edgeLayerBezierManager: this.edgeLayerBezierManager,
                coarseColorLayerBezierManager: this.coarseColorLayerBezierManager,
                fineColorLayerBezierManager: this.fineColorLayerBezierManager,
                bezierForwardManager: this.edgeBezierForwardManager,
                baseColorBezierForwardManager: this.coarseColorBezierForwardManager,
                colorBezierForwardManager: this.fineColorBezierForwardManager,
            },
        });

        this.destroy = $effect.root(() => {
            const activeCamera = $derived(
                this.viewerState.turntableTraining || this.viewerState.isTurntableRendering
                    ? this.backendCamera 
                    : this.viewportCamera
            );

            $effect(() => this.uniformsManager.writeViewProjMat(activeCamera.viewProjMat));
            $effect(() => this.uniformsManager.writeViewMat(activeCamera.viewMat));
            $effect(() => this.uniformsManager.writeInvViewProjMat(activeCamera.viewProjInvMat));
            $effect(() => {
                // Write invViewProjMat to path tracer and reset accumulation on camera change
                this.pathTracePipelineManager.writeInvViewProjMat(activeCamera.viewProjInvMat as Float32Array);
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
                activeCamera.viewProjMat;
                if (this.viewerState.isTurntableRendering) return;
                this.splatOptimizerManager.resetAdam();
                this.edgeLayerBezierManager.resetAdam();
                this.coarseColorLayerBezierManager.resetAdam();
                this.fineColorLayerBezierManager.resetAdam();
            });
            /** Bezier optimizers skip multiview; turntable+dataset assigns their VP inside loop(). */
            $effect(() => {
                if (this.viewerState.renderMode !== RENDER_MODE_MULTIVIEW) {
                    this.edgeLayerBezierManager.writeVPMatrix(activeCamera.viewProjMat);
                    this.coarseColorLayerBezierManager.writeVPMatrix(activeCamera.viewProjMat);
                    this.fineColorLayerBezierManager.writeVPMatrix(activeCamera.viewProjMat);
                    this.edgeLayerBezierManager.writeVPInvMatrix(activeCamera.viewProjInvMat);
                    this.coarseColorLayerBezierManager.writeVPInvMatrix(activeCamera.viewProjInvMat);
                    this.fineColorLayerBezierManager.writeVPInvMatrix(activeCamera.viewProjInvMat);
                }
            });
            $effect(() => this.edgeBezierForwardManager.writeVPMatrix(activeCamera.viewProjMat));
            $effect(() => this.coarseColorBezierForwardManager.writeVPMatrix(activeCamera.viewProjMat));
            $effect(() => this.fineColorBezierForwardManager.writeVPMatrix(activeCamera.viewProjMat));
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
                this.edgeBezierForwardManager.destroy();
                this.coarseColorBezierForwardManager.destroy();
                this.fineColorBezierForwardManager.destroy();
                this.blurManager.destroy();
                this.depthAwareBlurManager.destroy();
                this.pathTracePipelineManager.destroy();
                this.gpuPerfBuffers?.destroy();
                this.textures.destroy();
                this.turntable.destroy();
            };
        });
    }

    takeScreenshot(): Promise<Blob> {
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
        // Wait until optimization textures are ready (loop may not have run yet).
        while (!this.textures.optimizationColorTextureView || this.textures.optimizationWidth === 0) {
            await new Promise<void>(r => requestAnimationFrame(() => r()));
        }
        return this.turntable.prerenderDataset(this.textures.optimizationWidth, this.textures.optimizationHeight);
    }

    private recreateOptimizationTextures(panelAspect: number) {
        if (this.textures.recreateOptimizationTextures(panelAspect)) {
            // Resize path tracer output to match optimization resolution
            this.pathTracePipelineManager.setOutputSize(this.textures.optimizationWidth, this.textures.optimizationHeight);

            // Rebind
            this.splatOptimizerManager.setEdgeTarget(this.textures.optimizationDepthTextureView!, this.textures.optimizationEdgeTextureView!, this.textures.optimizationNormalTextureView!);
            this.splatOptimizerManager.setBackwardTarget(this.textures.optimizationColorTextureView!, this.textures.optimizationDepthTextureView!, this.textures.optimizationWidth, this.textures.optimizationHeight);

            // Edge layer: color target = edge map, depth = real depth, background = black (dummy).
            // Mode=1 color loss drives beziers white on edges, transparent off edges.
            this.edgeLayerBezierManager.setBackwardTarget(
                this.textures.optimizationEdgeTextureView!,
                this.textures.optimizationDepthTextureView!,
                this.textures.dummyTextureView!,
                this.textures.optimizationNormalTextureView!,
                this.textures.optimizationWidth,
                this.textures.optimizationHeight,
            );

            this.coarseColorLayerBezierManager.setBackwardTarget(
                this.textures.optimizationDepthAwareBlurredTextureView!,
                this.textures.optimizationDepthTextureView!,
                this.textures.optimizationSplatColorTextureView!,
                this.textures.optimizationNormalTextureView!,
                this.textures.optimizationWidth, this.textures.optimizationHeight
            );

            this.fineColorLayerBezierManager.setBackwardTarget(
                this.textures.optimizationColorTextureView!,
                this.textures.optimizationDepthTextureView!,
                this.textures.optimizationSplatColorTextureView!,
                this.textures.optimizationNormalTextureView!,
                this.textures.optimizationWidth, this.textures.optimizationHeight
            );
        }
    }

    loop() {
        let handle = 0;
        let canceled = false;
        const loop = () => {
            this.recreateOptimizationTextures(1);

            const displayResWidth = Math.max(1, this.viewerState.renderWidth);
            const displayResHeight = Math.max(1, this.viewerState.renderHeight);
            
            if (this.textures.recreateDisplayResTextures(displayResWidth, displayResHeight)) {
                this.splatForwardManager.setTarget(this.textures.displayResSplatColorTextureView!, this.textures.displayResSplatDepthTextureView!, displayResWidth, displayResHeight);
                this.edgeBezierForwardManager.setTarget(this.textures.displayResEdgeBezierTextureView!, displayResWidth, displayResHeight);
                this.coarseColorBezierForwardManager.setTarget(this.textures.displayResCoarseBezierTextureView!, displayResWidth, displayResHeight);
                this.fineColorBezierForwardManager.setTarget(this.textures.displayResFineBezierTextureView!, displayResWidth, displayResHeight);

                this.splatOptimizerManager.setRenderTarget(
                    this.textures.displayResColorTextureView!,
                    this.textures.displayResSplatColorTextureView!,
                    this.textures.displayResDepthTextureView!,
                    this.textures.displayResEdgeTextureView!,
                    this.textures.displayResEdgeBezierTextureView!,
                    this.textures.displayResCoarseBezierTextureView!,
                    this.textures.displayResFineBezierTextureView!,
                    this.textures.dummyTextureView!, // PT not ready yet at setup time
                );
            }

            if (!this.textures.displayResColorTextureView || !this.textures.displayResDepthTextureView || !this.textures.optimizationColorTextureView) {
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
            const recordGpu = this.gpuPerfBuffers !== null
                && this.viewerState.gpuProfilingEnabled
                && this.gpuPerfBuffers.resultBuffer.mapState === "unmapped";
            const profWrites = (label: string) => {
                if (!recordGpu) return undefined;
                const idx = this.gpuPerfBuffers!.getIndex(label);
                activeProfilerIndices.add(idx);
                return this.gpuPerfBuffers!.writes(idx);
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
            this.edgeBezierForwardManager.writeVPMatrix(sortVp);
            this.coarseColorBezierForwardManager.writeVPMatrix(sortVp);
            this.fineColorBezierForwardManager.writeVPMatrix(sortVp);

            this.edgeLayerBezierManager.writeCamWorld(camWorld[0], camWorld[1], camWorld[2]);
            this.coarseColorLayerBezierManager.writeCamWorld(camWorld[0], camWorld[1], camWorld[2]);
            this.fineColorLayerBezierManager.writeCamWorld(camWorld[0], camWorld[1], camWorld[2]);
            const cx = camWorld[0];
            const cy = camWorld[1];
            const cz = camWorld[2];
            this.edgeBezierForwardManager.writeCameraWorld(cx, cy, cz);
            this.coarseColorBezierForwardManager.writeCameraWorld(cx, cy, cz);
            this.fineColorBezierForwardManager.writeCameraWorld(cx, cy, cz);

            // 1a. Render the model into the full-res target + depth textures (for visualization).
            if (!this.viewerState.viewportRenderingFrozen) {
                const spherePassEncoder = commandEncoder.beginRenderPass({
                    label: "mesh render pass (full res)",
                    ...(recordGpu ? { timestampWrites: profWrites("Mesh: full viewport") } : {}),
                    colorAttachments: [
                        {
                            clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1.0 },
                            loadOp: "clear",
                            storeOp: "store",
                            view: this.textures.displayResColorTextureView!,
                        },
                        {
                            clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
                            loadOp: "clear",
                            storeOp: "store",
                            view: this.textures.displayResDepthTextureView!,
                        },
                        {
                            clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 1.0 },
                            loadOp: "clear",
                            storeOp: "store",
                            view: this.textures.displayResNormalTextureView!,
                        },
                    ],
                    depthStencilAttachment: {
                        view: this.textures.displayResZBufferTextureView!,
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
            const optimizationPassEncoder = commandEncoder.beginRenderPass({
                label: "mesh render pass (optimization res)",
                ...(recordGpu ? { timestampWrites: profWrites("Mesh: optim target") } : {}),
                colorAttachments: [
                    {
                        clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1.0 },
                        loadOp: "clear",
                        storeOp: "store",
                        view: this.textures.optimizationColorTextureView!,
                    },
                    {
                        clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
                        loadOp: "clear",
                        storeOp: "store",
                        view: this.textures.optimizationDepthTextureView!,
                    },
                    {
                        clearValue: { r: 0.5, g: 0.5, b: 0.5, a: 1.0 },
                        loadOp: "clear",
                        storeOp: "store",
                        view: this.textures.optimizationNormalTextureView!,
                    },
                ],
                depthStencilAttachment: {
                    view: this.textures.optimizationZBufferTextureView!,
                    depthClearValue: 1.0,
                    depthLoadOp: "clear",
                    depthStoreOp: "store",
                },
            });
            this.envmapPipelineManager.addDraw(optimizationPassEncoder);
            this.meshRenderPipelineManager.addDraw(optimizationPassEncoder, this.matcapTextureView);
            optimizationPassEncoder.end();

            // 1b.5 Path trace pass — accumulates one sample per pixel into the PT output texture.
            // The PT output is used as the target for the splat/bezier optimizers instead of
            // the rasterized mesh render, giving a more physically-based training signal.
            // Skip during dataset-driven training — the prerendered textures are used directly.
            if (!datasetView) {
                this.pathTracePipelineManager.addDispatches(
                    commandEncoder,
                    profWrites("Path trace (compute)"),
                );
            }

            // Use dataset view if available, else PT output, else raster fallback.
            const ptOutputView = this.pathTracePipelineManager.outputTextureView;
            const optimizationTargetView = datasetView ?? ptOutputView ?? this.textures.optimizationColorTextureView!;

            // 1c. Run separable blur on targets if enabled
            if (this.viewerState.compareBlurred) {
                this.blurManager.addDispatches(
                    commandEncoder,
                    optimizationTargetView,
                    this.textures.optimizationBlurredTextureView!,
                    this.textures.optimizationBlurTempTextureView!,
                    this.textures.optimizationWidth,
                    this.textures.optimizationHeight,
                    this.viewerState.blurRadius,
                    this.viewerState.blurRadius / 2,
                    true, // isSrgb
                    profWrites("Blur: optim color"),
                );
                this.blurManager.addDispatches(
                    commandEncoder,
                    this.textures.optimizationDepthTextureView!,
                    this.textures.optimizationBlurredDepthTextureView!,
                    this.textures.optimizationBlurTempTextureView!,
                    this.textures.optimizationWidth,
                    this.textures.optimizationHeight,
                    this.viewerState.blurRadius,
                    this.viewerState.blurRadius / 2,
                    false, // isSrgb
                    profWrites("Blur: optim depth"),
                );
            }
            
            if (this.viewerState.coarseColorBeziersEnabled) {
                this.depthAwareBlurManager.addDispatches(
                    commandEncoder,
                    optimizationTargetView,
                    this.textures.optimizationDepthTextureView!,
                    this.textures.optimizationNormalTextureView!,
                    this.textures.optimizationDepthAwareBlurredTextureView!,
                    this.textures.optimizationWidth,
                    this.textures.optimizationHeight,
                    15,
                    profWrites("Blur: depth-aware"),
                );
            }
            
            // Update backward targets for all optimizers to point to current frame's target view
            this.splatOptimizerManager.setBackwardTarget(
                this.viewerState.compareBlurred ? this.textures.optimizationBlurredTextureView! : optimizationTargetView,
                this.viewerState.compareBlurred ? this.textures.optimizationBlurredDepthTextureView! : this.textures.optimizationDepthTextureView!,
                this.textures.optimizationWidth,
                this.textures.optimizationHeight
            );

            if (this.viewerState.coarseColorBeziersEnabled) {
                this.coarseColorLayerBezierManager.setBackwardTarget(
                    this.textures.optimizationDepthAwareBlurredTextureView!,
                    this.textures.optimizationDepthTextureView!,
                    this.textures.optimizationSplatColorTextureView!,
                    this.textures.optimizationNormalTextureView!,
                    this.textures.optimizationWidth,
                    this.textures.optimizationHeight,
                );
            }
            if (this.viewerState.fineColorBeziersEnabled) {
                this.fineColorLayerBezierManager.setBackwardTarget(
                    optimizationTargetView,
                    this.textures.optimizationDepthTextureView!,
                    this.textures.optimizationSplatColorTextureView!,
                    this.textures.optimizationNormalTextureView!,
                    this.textures.optimizationWidth,
                    this.textures.optimizationHeight,
                );
            }
            if (this.viewerState.edgeBeziersEnabled) {
                this.edgeLayerBezierManager.setBackwardTarget(
                    this.textures.optimizationEdgeTextureView!,
                    this.textures.optimizationDepthTextureView!,
                    this.textures.dummyTextureView!,
                    this.textures.optimizationNormalTextureView!,
                    this.textures.optimizationWidth,
                    this.textures.optimizationHeight,
                );
            }

            // 2. Optimization Pass (Compute)
            const defaultPause = this.viewerState.renderMode === RENDER_MODE_MULTIVIEW && (!this.viewerState.turntableTraining || !this.viewerState.multiviewDatasetReady);
            
            // Clear all binning and sorting buffers
            if (this.viewerState.splatsEnabled) {
                this.splatOptimizerManager.clearBinningBuffers(commandEncoder);
            }
            if (this.viewerState.coarseColorBeziersEnabled) {
                this.coarseColorLayerBezierManager.clearBinningBuffers(commandEncoder);
            }
            if (this.viewerState.fineColorBeziersEnabled) {
                this.fineColorLayerBezierManager.clearBinningBuffers(commandEncoder);
            }
            if (this.viewerState.edgeBeziersEnabled) {
                this.edgeLayerBezierManager.clearBinningBuffers(commandEncoder);
            }

            // Edge detection (optimization res)
            const edgeOptimizationPass = commandEncoder.beginComputePass({
                label: "splat edge detection (optimization res)",
                ...(recordGpu ? { timestampWrites: profWrites("Splat: edge detect (optim)") } : {}),
            });
            this.splatOptimizerManager.addEdgeDispatches(edgeOptimizationPass, this.textures.optimizationWidth, this.textures.optimizationHeight);
            edgeOptimizationPass.end();

            // Splat optimization
            const splatPause = this.viewerState.splatTrainingPaused || defaultPause;
            if (this.viewerState.splatsEnabled && !splatPause) {
                const splatOptPass = commandEncoder.beginComputePass({
                    label: "splat optimization compute",
                    ...(recordGpu ? { timestampWrites: profWrites("Splat: optimization") } : {}),
                });
                this.splatOptimizerManager.addBinningDispatches(splatOptPass, sortVp);
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
                        ...(recordGpu ? { timestampWrites: profWrites("Bézier (coarse): optimization") } : {}),
                    });
                    this.coarseColorLayerBezierManager.addBinningDispatches(coarseOptPass, sortVp);
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
                        ...(recordGpu ? { timestampWrites: profWrites("Bézier (fine): optimization") } : {}),
                    });
                    this.fineColorLayerBezierManager.addBinningDispatches(fineOptPass, sortVp);
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
                        ...(recordGpu ? { timestampWrites: profWrites("Bézier (edge): optimization") } : {}),
                    });
                    this.edgeLayerBezierManager.addBinningDispatches(edgeOptPass, sortVp);
                    this.edgeLayerBezierManager.addOptimizationDispatches(edgeOptPass);
                    edgeOptPass.end();
                }
                const edgeSortPass = commandEncoder.beginComputePass({ label: "edge bezier sort" });
                this.edgeLayerBezierManager.addSortDispatches(edgeSortPass, sortVp);
                edgeSortPass.end();
            }

            if (!this.viewerState.viewportRenderingFrozen || needsTurntableExportLayers) {
                this.splatOptimizerManager.setEdgeTarget(this.textures.displayResDepthTextureView!, this.textures.displayResEdgeTextureView!, this.textures.displayResNormalTextureView!);

                const edgeFullPass = commandEncoder.beginComputePass({
                    label: "splat edge detection (full res)",
                    ...(recordGpu ? { timestampWrites: profWrites("Edge detect (display res)") } : {}),
                });
                this.splatOptimizerManager.addEdgeDispatches(edgeFullPass, displayResWidth, displayResHeight);
                edgeFullPass.end();
                // Reset target for next frame
                this.splatOptimizerManager.setEdgeTarget(this.textures.optimizationDepthTextureView!, this.textures.optimizationEdgeTextureView!, this.textures.optimizationNormalTextureView!);
            }

            // 3. Render Pass (Optimization-Res)
            // Group Splat and Coarse Bezier which target the same texture
            this.splatForwardManager.setTarget(this.textures.optimizationSplatColorTextureView!, this.textures.optimizationSplatDepthTextureView!, this.textures.optimizationWidth, this.textures.optimizationHeight);
            this.coarseColorBezierForwardManager.setTarget(this.textures.optimizationSplatColorTextureView!, this.textures.optimizationWidth, this.textures.optimizationHeight);

            const optimizationRenderPass = commandEncoder.beginRenderPass({
                label: "optimization-res render pass",
                ...(recordGpu ? { timestampWrites: profWrites("Splat: forward (optim res)") } : {}),
                colorAttachments: [
                    {
                        view: this.textures.optimizationSplatColorTextureView!,
                        clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1.0 },
                        loadOp: "clear",
                        storeOp: "store",
                    },
                    {
                        view: this.textures.optimizationSplatDepthTextureView!,
                        clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
                        loadOp: "clear",
                        storeOp: "store",
                    },
                ],
            });

            this.splatForwardManager.render(optimizationRenderPass, this.viewerState.splatsEnabled);
            if (this.viewerState.coarseColorBeziersEnabled) {
                this.coarseColorBezierForwardManager.render(optimizationRenderPass, true);
            }
            optimizationRenderPass.end();

            // 4. Full-res Visualization Renders
            if (!this.viewerState.viewportRenderingFrozen || needsTurntableExportLayers) {
                // Splat Full-res
                this.splatForwardManager.setTarget(this.textures.displayResSplatColorTextureView!, this.textures.displayResSplatDepthTextureView!, displayResWidth, displayResHeight);
                this.splatForwardManager.addDispatches(
                    commandEncoder, 
                    true, 
                    this.viewerState.splatsEnabled,
                    recordGpu ? profWrites("Splat: forward (display res)") : undefined
                );
                
                // Edge Bezier Full-res
                if (this.viewerState.edgeBeziersEnabled) {
                    this.edgeBezierForwardManager.setTarget(this.textures.displayResEdgeBezierTextureView!, displayResWidth, displayResHeight);
                    this.edgeBezierForwardManager.addDispatches(
                        commandEncoder, 
                        true,
                        recordGpu ? profWrites("Bézier (edge): forward (display res)") : undefined
                    );
                }
                
                // Coarse Bezier Full-res
                if (this.viewerState.coarseColorBeziersEnabled) {
                    this.coarseColorBezierForwardManager.setTarget(this.textures.displayResCoarseBezierTextureView!, displayResWidth, displayResHeight);
                    this.coarseColorBezierForwardManager.addDispatches(
                        commandEncoder, 
                        true,
                        recordGpu ? profWrites("Bézier (coarse): forward (display res)") : undefined
                    );
                }
                
                // Fine Bezier Full-res
                if (this.viewerState.fineColorBeziersEnabled) {
                    this.fineColorBezierForwardManager.setTarget(this.textures.displayResFineBezierTextureView!, displayResWidth, displayResHeight);
                    this.fineColorBezierForwardManager.addDispatches(
                        commandEncoder, 
                        true,
                        recordGpu ? profWrites("Bézier (fine): forward (display res)") : undefined
                    );
                }
            }

            // 5. Render Splat Visualization to Screen Views (uses full-res textures)
            const ptView = this.pathTracePipelineManager.outputTextureView ?? this.textures.dummyTextureView!;
            this.splatOptimizerManager.setRenderTarget(
                this.textures.displayResColorTextureView!,
                this.textures.displayResSplatColorTextureView!,
                this.textures.displayResDepthTextureView!,
                this.textures.displayResEdgeTextureView!,
                this.textures.displayResEdgeBezierTextureView!,
                this.textures.displayResCoarseBezierTextureView!,
                this.textures.displayResFineBezierTextureView!,
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
                    ...(recordGpu && id === "target" ? { timestampWrites: profWrites("Final compositor (screen)") } : {}),
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
                // Fire-and-forget: don't block the render loop waiting for GPU readback.
                const perfBuffers = this.gpuPerfBuffers;
                const vs = this.viewerState;
                const indices = new Set(activeProfilerIndices);
                this.device.queue.onSubmittedWorkDone().then(async () => {
                    try {
                        const deltasNs = await perfBuffers.mapDeltasNanoseconds(indices);
                        // Skip update when nothing was actually read (buffer still
                        // mapped from a previous frame's async readback).
                        if (deltasNs.every(v => v === null)) return;
                        const labels = perfBuffers.getLabels();
                        const entries = labels.map((label, idx) => ({
                            label,
                            ms: deltasNs[idx] === null ? null : Number(deltasNs[idx]) / 1e6
                        }));
                        vs.setGpuProfilingFrameMs(entries);
                    } catch (e) {
                        console.warn("[gpu profiler]", e);
                    }
                });
            } else if (!this.viewerState.gpuProfilingEnabled) {
                // Profiling disabled — wipe stale charts
                this.viewerState.setGpuProfilingFrameMs([]);
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
                displayResWidth,
                displayResHeight,
                this.textures.displayResSplatColorTexture,
                this.textures.displayResCoarseBezierTexture,
                this.textures.displayResFineBezierTexture,
                this.textures.displayResEdgeBezierTexture,
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
