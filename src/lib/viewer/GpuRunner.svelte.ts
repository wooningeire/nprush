import type { Camera } from "./Camera.svelte";
import { GpuUniformsBufferManager } from "$/gpu/GpuUniformsBufferManager";
import { GpuMeshRenderPipelineManager, MESH_DEPTH_FORMAT } from "$/gpu/GpuMeshRenderPipelineManager";
import { GpuSplatOptimizerManager } from "$/gpu/GpuSplatOptimizerManager";
import { GpuBezierOptimizerManager } from "$/gpu/GpuBezierOptimizerManager";
import { GpuSplatForwardPipelineManager } from "$/gpu/GpuSplatForwardPipelineManager";
import { GpuBezierForwardPipelineManager } from "$/gpu/GpuBezierForwardPipelineManager";
import { GpuBlurPipelineManager } from "$/gpu/GpuBlurPipelineManager";
import { GpuDepthAwareBlurPipelineManager } from "$/gpu/GpuDepthAwareBlurPipelineManager";
import { GpuEnvmapPipelineManager } from "$/gpu/GpuEnvmapPipelineManager";
import { GpuPathTracePipelineManager } from "$/gpu/GpuPathTracePipelineManager";
import type { MeshData } from "$/gpu/loadGlb";
import { STRIP_HEIGHT_FRAC } from "$/util";

const OPTIM_SHORT = 256;

// The edge layer is now cubic bezier curves. A handful is enough since each
// curve is a 1D primitive that natively traces a contour.
const NUM_EDGE_LAYER_BEZIERS = 4096;

export class GpuRunner {
    private readonly device: GPUDevice;
    private readonly context: GPUCanvasContext;
    private readonly format: GPUTextureFormat;
    private readonly camera: Camera;
    private readonly viewerState: any; // Using any to avoid circular dependency if needed, but ViewerState should be fine

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
    private optimWidth = 0;
    private optimHeight = 0;

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
        numSplats = 512,
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
        });

        this.bezierForwardManager = new GpuBezierForwardPipelineManager({
            device,
            numBeziers: NUM_EDGE_LAYER_BEZIERS,
            bezierBuffer: this.edgeLayerBezierManager.bezierBuffer,
            brushTexture,
        });

        this.baseColorBezierForwardManager = new GpuBezierForwardPipelineManager({
            device,
            numBeziers: NUM_EDGE_LAYER_BEZIERS,
            bezierBuffer: this.baseColorLayerBezierManager.bezierBuffer,
            brushTexture,
        });

        this.colorBezierForwardManager = new GpuBezierForwardPipelineManager({
            device,
            numBeziers: NUM_EDGE_LAYER_BEZIERS,
            bezierBuffer: this.colorLayerBezierManager.bezierBuffer,
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
            $effect(() => this.uniformsManager.writeShadingMode(this.viewerState.shadingMode));
            $effect(() => this.splatOptimizerManager.writeRenderUniforms(
                this.viewerState.edgeBeziersEnabled,
                this.viewerState.baseColorBeziersEnabled,
                this.viewerState.colorBeziersEnabled,
                this.viewerState.posterizationEnabled
            ));
            $effect(() => this.splatOptimizerManager.writeSplatVPMatrix(this.camera.viewProjMat, this.camera.viewProjInvMat, this.viewerState.compareBlurred));
            $effect(() => this.splatForwardManager.writeVPMatrix(this.camera.viewProjMat));
            $effect(() => this.edgeLayerBezierManager.writeVPMatrix(this.camera.viewProjMat));
            $effect(() => this.baseColorLayerBezierManager.writeVPMatrix(this.camera.viewProjMat));
            $effect(() => this.colorLayerBezierManager.writeVPMatrix(this.camera.viewProjMat));
            $effect(() => {
                this.edgeLayerBezierManager.writeVPInvMatrix(this.camera.viewProjInvMat);
                this.baseColorLayerBezierManager.writeVPInvMatrix(this.camera.viewProjInvMat);
                this.colorLayerBezierManager.writeVPInvMatrix(this.camera.viewProjInvMat);
            });
            $effect(() => this.bezierForwardManager.writeVPMatrix(this.camera.viewProjMat));
            $effect(() => this.baseColorBezierForwardManager.writeVPMatrix(this.camera.viewProjMat));
            $effect(() => this.colorBezierForwardManager.writeVPMatrix(this.camera.viewProjMat));
            $effect(() => {
                this.edgeLayerBezierManager.writeMode(0); // Edge mode
                this.baseColorLayerBezierManager.writeMode(1); // Color+Depth mode
                this.colorLayerBezierManager.writeMode(1); // Color+Depth mode
                this.colorLayerBezierManager.writeMaxWidth(0.012); // finer strokes on second color layer
                // Fine color layer: less aggressive killing so thin strokes survive,
                // but background penalty enabled to kill off-model curves.
                this.colorLayerBezierManager.writeKillThresholds(0.0001, 0.0001);
                this.colorLayerBezierManager.writeNoKill(true);
                this.colorLayerBezierManager.writeBgPenalty(0.0);
                // Base color layer: no background penalty (blurred target bleeds into bg).
                // Enable no_kill so broad strokes aren't pruned before they settle —
                // the ADC stuck+loss kill was the main source of base-layer jitter.
                // Longer ADC period reduces clone/kill churn on broad strokes.
                this.baseColorLayerBezierManager.writeBgPenalty(0.0);
                this.baseColorLayerBezierManager.writeNoKill(true);
                this.baseColorLayerBezierManager.writeKillThresholds(0.0001, 0.0001);
                this.baseColorLayerBezierManager.setAdcPeriod(150);
            });
        });
    }

    private rebuildOptimTextures(panelAspect: number) {
        // Size optim textures to match the visible panel aspect ratio so the model
        // rendered into them has matching pixel proportions for the gradient pass.
        let ow: number, oh: number;
        if (panelAspect >= 1) {
            oh = OPTIM_SHORT;
            ow = Math.round(OPTIM_SHORT * panelAspect);
        } else {
            ow = OPTIM_SHORT;
            oh = Math.round(OPTIM_SHORT / panelAspect);
        }

        if (ow === this.optimWidth && oh === this.optimHeight) return;
        this.optimWidth = ow;
        this.optimHeight = oh;

        // Resize path tracer output to match optim resolution
        this.pathTracePipelineManager.setOutputSize(ow, oh);

        if (this.optimTexture) this.optimTexture.destroy();
        if (this.optimDepthTexture) this.optimDepthTexture.destroy();
        if (this.optimZTexture) this.optimZTexture.destroy();
        if (this.optimEdgeTexture) this.optimEdgeTexture.destroy();

        this.optimTexture = this.device.createTexture({
            label: "optimization target texture",
            size: [ow, oh],
            format: this.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimTextureView = this.optimTexture.createView();

        this.optimDepthTexture = this.device.createTexture({
            label: "optimization depth texture",
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
            label: "optimization edge texture",
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
            label: "optimization temp blur target",
            size: [ow, oh],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimTempTextureView = this.optimTempTexture.createView();

        if (!this.dummyTexture) {
            this.dummyTexture = this.device.createTexture({
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

            // Rebuild optim textures to match the camera's visible-panel aspect
            // (right half of canvas, above the debug strip). Must match Camera.aspect.
            const panelAspect = (width / 2) / (height * (1 - STRIP_HEIGHT_FRAC));
            this.rebuildOptimTextures(panelAspect);

            // Full-res target for visualization, sized to the visible main panel so the
            // texture aspect matches the camera projection aspect.
            const fullW = Math.max(1, Math.floor(width / 2));
            const fullH = Math.max(1, Math.floor(height * (1 - STRIP_HEIGHT_FRAC)));
            if (!this.targetTexture || this.fullWidth !== fullW || this.fullHeight !== fullH) {
                if (this.targetTexture) this.targetTexture.destroy();
                if (this.targetDepthTexture) this.targetDepthTexture.destroy();
                if (this.targetZTexture) this.targetZTexture.destroy();
                if (this.fullEdgeTexture) this.fullEdgeTexture.destroy();
                if (this.fullSplatTexture) this.fullSplatTexture.destroy();
                if (this.fullBezierTexture) this.fullBezierTexture.destroy();
                if (this.fullBaseColorBezierTexture) this.fullBaseColorBezierTexture.destroy();
                if (this.fullColorBezierTexture) this.fullColorBezierTexture.destroy();

                this.fullWidth = fullW;
                this.fullHeight = fullH;

                this.targetTexture = this.device.createTexture({
                    size: [fullW, fullH],
                    format: this.format,
                    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
                });
                this.targetTextureView = this.targetTexture.createView();

                this.targetDepthTexture = this.device.createTexture({
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
                    size: [fullW, fullH],
                    format: "rgba8unorm",
                    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
                });
                this.fullEdgeTextureView = this.fullEdgeTexture.createView();

                this.fullSplatTexture = this.device.createTexture({
                    label: "full splat view",
                    size: [fullW, fullH],
                    format: "rgba8unorm",
                    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
                });
                this.fullSplatTextureView = this.fullSplatTexture.createView();

                this.fullSplatDepthTexture = this.device.createTexture({
                    label: "full splat depth view",
                    size: [fullW, fullH],
                    format: "rgba8unorm",
                    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
                });
                this.fullSplatDepthTextureView = this.fullSplatDepthTexture.createView();

                this.fullBezierTexture = this.device.createTexture({
                    label: "full bezier view",
                    size: [fullW, fullH],
                    format: "rgba8unorm",
                    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
                });
                this.fullBezierTextureView = this.fullBezierTexture.createView();

                this.fullBaseColorBezierTexture = this.device.createTexture({
                    label: "full base color bezier view",
                    size: [fullW, fullH],
                    format: "rgba8unorm",
                    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
                });
                this.fullBaseColorBezierTextureView = this.fullBaseColorBezierTexture.createView();

                this.fullColorBezierTexture = this.device.createTexture({
                    label: "full color bezier view",
                    size: [fullW, fullH],
                    format: "rgba8unorm",
                    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
                });
                this.fullColorBezierTextureView = this.fullColorBezierTexture.createView();

                this.targetBlurredTexture = this.device.createTexture({
                    label: "full blurred target",
                    size: [fullW, fullH],
                    format: "rgba8unorm",
                    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
                });
                this.targetBlurredTextureView = this.targetBlurredTexture.createView();

                this.targetTempTexture = this.device.createTexture({
                    label: "full temp blur target",
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

            const commandEncoder = this.device.createCommandEncoder({
                label: "loop command encoder",
            });

            // 1a. Render the model into the full-res target + depth textures (for visualization).
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
            this.pathTracePipelineManager.dispatch(commandEncoder);

            // Use path trace output as the optimization target if available, else fall back to raster.
            const ptOutputView = this.pathTracePipelineManager.outputTextureView;
            const optimTargetView = ptOutputView ?? this.optimTextureView!;

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
            if (!this.viewerState.splatTrainingPaused) {
                this.splatOptimizerManager.dispatch(commandEncoder);
            }

            // 3.1 Render current splats at optim-res to use as background for color beziers.
            this.splatForwardManager.setTarget(
                this.optimSplatTextureView!,
                this.optimSplatDepthTextureView!,
                this.optimWidth,
                this.optimHeight
            );
            this.splatForwardManager.dispatch(commandEncoder, true);

            // 3.2 Restore full-res target for visualization later
            this.splatForwardManager.setTarget(this.fullSplatTextureView!, this.fullSplatDepthTextureView!, fullW, fullH);

            // 3b. Train the bezier edge layer: its target is the freshly-computed
            // edge texture, so the curves learn to trace the depth silhouette.
            if (this.viewerState.edgeBeziersEnabled) {
                if (!this.viewerState.edgeBezierTrainingPaused) {
                    this.edgeLayerBezierManager.dispatch(commandEncoder);
                }
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
                if (!this.viewerState.baseColorBezierTrainingPaused) {
                    this.baseColorLayerBezierManager.dispatch(commandEncoder);
                }

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
                if (!this.viewerState.colorBezierTrainingPaused) {
                    this.colorLayerBezierManager.dispatch(commandEncoder);
                }
            }

            // 4. Run edge detection on full-res depth (for display)
            this.splatOptimizerManager.setEdgeTarget(this.targetDepthTextureView!, this.fullEdgeTextureView!);
            this.splatOptimizerManager.dispatchEdge(commandEncoder, fullW, fullH);
            // Restore optim-res edge bind group for next frame
            this.splatOptimizerManager.setEdgeTarget(this.optimDepthTextureView!, this.optimEdgeTextureView!);

            // 4.5. Compute views into textures
            this.splatForwardManager.dispatch(commandEncoder, true);
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
            handle = requestAnimationFrame(loop);
        };

        handle = requestAnimationFrame(loop);

        return () => {
            cancelAnimationFrame(handle);
            canceled = true;
        };
    }
}