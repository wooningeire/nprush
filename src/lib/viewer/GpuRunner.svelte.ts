import type { Camera } from "./Camera.svelte";
import { GpuUniformsBufferManager } from "$/gpu/GpuUniformsBufferManager";
import { GpuMeshRenderPipelineManager, MESH_DEPTH_FORMAT } from "$/gpu/GpuMeshRenderPipelineManager";
import { GpuSplatOptimizerManager } from "$/gpu/GpuSplatOptimizerManager";
import { GpuBezierOptimizerManager } from "$/gpu/GpuBezierOptimizerManager";
import { GpuSplatForwardPipelineManager } from "$/gpu/GpuSplatForwardPipelineManager";
import { GpuBezierForwardPipelineManager } from "$/gpu/GpuBezierForwardPipelineManager";
import type { MeshData } from "$/gpu/loadGlb";
import { STRIP_HEIGHT_FRAC } from "$/util";

const OPTIM_SHORT = 128;

// The edge layer is now cubic bezier curves. A handful is enough since each
// curve is a 1D primitive that natively traces a contour.
const NUM_EDGE_LAYER_BEZIERS = 512;

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
    readonly splatForwardManager: GpuSplatForwardPipelineManager;
    readonly bezierForwardManager: GpuBezierForwardPipelineManager;
    private readonly matcapTexture: GPUTexture;
    private readonly matcapTextureView: GPUTextureView;

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
    private fullBezierTexture: GPUTexture | null = null;
    private fullBezierTextureView: GPUTextureView | null = null;
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
        matcapTexture,
        numSplats = 512,
    }: {
        device: GPUDevice,
        context: GPUCanvasContext,
        format: GPUTextureFormat,
        camera: Camera,
        viewerState: any,
        mesh: MeshData,
        matcapTexture: GPUTexture,
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

        this.splatForwardManager = new GpuSplatForwardPipelineManager({
            device,
            numSplats,
            splatBuffer: this.splatOptimizerManager.splatBuffer,
        });

        this.bezierForwardManager = new GpuBezierForwardPipelineManager({
            device,
            numBeziers: NUM_EDGE_LAYER_BEZIERS,
            bezierBuffer: this.edgeLayerBezierManager.bezierBuffer,
        });

        this.destroy = $effect.root(() => {
            $effect(() => this.uniformsManager.writeViewProjMat(this.camera.viewProjMat));
            $effect(() => this.uniformsManager.writeViewMat(this.camera.viewMat));
            $effect(() => this.uniformsManager.writeShadingMode(this.viewerState.shadingMode));
            $effect(() => this.splatOptimizerManager.writeRenderUniforms(this.viewerState.beziersEnabled));
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

        // Rebind
        this.splatOptimizerManager.setEdgeTarget(this.optimDepthTextureView, this.optimEdgeTextureView);
        this.splatOptimizerManager.setBackwardTarget(this.optimTextureView, this.optimEdgeTextureView, ow, oh);

        // Edge layer: target IS the edge texture itself. The bezier backward
        // shader takes (target, edgeWeight); we pass edge for both, so the
        // curves learn to reconstruct the edge image with extra weighting on
        // edge pixels (parity with how the splat manager treats its edge loss).
        this.edgeLayerBezierManager.setBackwardTarget(this.optimEdgeTextureView, this.optimEdgeTextureView, ow, oh);
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

                this.fullBezierTexture = this.device.createTexture({
                    label: "full bezier view",
                    size: [fullW, fullH],
                    format: "rgba8unorm",
                    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
                });
                this.fullBezierTextureView = this.fullBezierTexture.createView();

                this.splatForwardManager.setTarget(this.fullSplatTextureView, fullW, fullH);
                this.bezierForwardManager.setTarget(this.fullBezierTextureView, fullW, fullH);

                this.splatOptimizerManager.setRenderTarget(
                    this.targetTextureView,
                    this.fullSplatTextureView,
                    this.targetDepthTextureView,
                    this.fullEdgeTextureView,
                    this.fullBezierTextureView,
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
                        clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
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
            this.meshRenderPipelineManager.addDraw(spherePassEncoder, this.matcapTextureView);
            spherePassEncoder.end();

            // 1b. Render the model into the optim-res (aspect-matched) textures for gradient computation.
            const optimPassEncoder = commandEncoder.beginRenderPass({
                label: "mesh render pass (optim res)",
                colorAttachments: [
                    {
                        clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
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
            this.meshRenderPipelineManager.addDraw(optimPassEncoder, this.matcapTextureView);
            optimPassEncoder.end();

            // 2. Run edge detection on optim-res depth
            this.splatOptimizerManager.dispatchEdge(commandEncoder, this.optimWidth, this.optimHeight);

            // 3. Dispatch Splat Optimizer Compute Passes (uses optim-res texture + edge map)
            this.splatOptimizerManager.dispatch(commandEncoder);

            // 3b. Train the bezier edge layer: its target is the freshly-computed
            // edge texture, so the curves learn to trace the depth silhouette.
            if (this.viewerState.beziersEnabled) {
                this.edgeLayerBezierManager.dispatch(commandEncoder);
            }

            // 4. Run edge detection on full-res depth (for display)
            this.splatOptimizerManager.setEdgeTarget(this.targetDepthTextureView!, this.fullEdgeTextureView!);
            this.splatOptimizerManager.dispatchEdge(commandEncoder, fullW, fullH);
            // Restore optim-res edge bind group for next frame
            this.splatOptimizerManager.setEdgeTarget(this.optimDepthTextureView!, this.optimEdgeTextureView!);

            // 4.5. Compute views into textures
            this.splatForwardManager.dispatch(commandEncoder);
            if (this.viewerState.beziersEnabled) {
                this.bezierForwardManager.dispatch(commandEncoder);
            }

            // 5. Render Splat Visualization to Screen View (uses full-res textures)
            const screenView = currentTexture.createView();
            const finalPassEncoder = commandEncoder.beginRenderPass({
                label: "final render pass",
                colorAttachments: [
                    {
                        clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
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