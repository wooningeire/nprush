import type { Camera } from "./Camera.svelte";
import { GpuUniformsBufferManager } from "$/gpu/GpuUniformsBufferManager";
import { GpuSphereRenderPipelineManager } from "$/gpu/GpuSphereRenderPipelineManager";
import { GpuSplatOptimizerManager } from "$/gpu/GpuSplatOptimizerManager";
import { STRIP_HEIGHT_FRAC } from "$/util";

const OPTIM_SHORT = 128;

export class GpuRunner {
    private readonly device: GPUDevice;
    private readonly context: GPUCanvasContext;
    private readonly format: GPUTextureFormat;
    private readonly camera: Camera;

    readonly uniformsManager: GpuUniformsBufferManager;
    readonly sphereRenderPipelineManager: GpuSphereRenderPipelineManager;
    readonly splatOptimizerManager: GpuSplatOptimizerManager;

    // Full-res textures (sized to the visible main panel area: half-width x height-minus-strip).
    // These match the camera projection aspect, so the sphere is circular in pixels.
    private targetTexture: GPUTexture | null = null;
    private targetTextureView: GPUTextureView | null = null;
    private targetDepthTexture: GPUTexture | null = null;
    private targetDepthTextureView: GPUTextureView | null = null;
    private fullEdgeTexture: GPUTexture | null = null;
    private fullEdgeTextureView: GPUTextureView | null = null;
    private fullWidth = 0;
    private fullHeight = 0;

    // Optim-res textures (aspect-matched to half-screen)
    private optimTexture: GPUTexture | null = null;
    private optimTextureView: GPUTextureView | null = null;
    private optimDepthTexture: GPUTexture | null = null;
    private optimDepthTextureView: GPUTextureView | null = null;
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
        numSplats = 512,
    }: {
        device: GPUDevice,
        context: GPUCanvasContext,
        format: GPUTextureFormat,
        camera: Camera,
        numSplats?: number,
    }) {
        this.device = device;
        this.context = context;
        this.format = format;
        this.camera = camera;

        this.uniformsManager = new GpuUniformsBufferManager({ device });
        
        this.sphereRenderPipelineManager = new GpuSphereRenderPipelineManager({
            device,
            format,
            uniformsManager: this.uniformsManager,
        });

        this.splatOptimizerManager = new GpuSplatOptimizerManager({
            device,
            format,
            numSplats,
        });

        this.destroy = $effect.root(() => {
            $effect(() => this.uniformsManager.writeViewProjMat(this.camera.viewProjMat));
        });
    }

    private rebuildOptimTextures(panelAspect: number) {
        // Size optim textures to match the visible panel aspect ratio so the sphere
        // rendered into them is circular in pixels.
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
            // texture aspect matches the camera projection aspect (sphere stays circular).
            const fullW = Math.max(1, Math.floor(width / 2));
            const fullH = Math.max(1, Math.floor(height * (1 - STRIP_HEIGHT_FRAC)));
            if (!this.targetTexture || this.fullWidth !== fullW || this.fullHeight !== fullH) {
                if (this.targetTexture) this.targetTexture.destroy();
                if (this.targetDepthTexture) this.targetDepthTexture.destroy();
                if (this.fullEdgeTexture) this.fullEdgeTexture.destroy();

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

                this.fullEdgeTexture = this.device.createTexture({
                    size: [fullW, fullH],
                    format: "rgba8unorm",
                    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
                });
                this.fullEdgeTextureView = this.fullEdgeTexture.createView();

                this.splatOptimizerManager.setRenderTarget(this.targetTextureView, this.targetDepthTextureView, this.fullEdgeTextureView);
            }

            if (!this.targetTextureView || !this.targetDepthTextureView || !this.optimTextureView) {
                if (!canceled) requestAnimationFrame(loop);
                return;
            }

            const commandEncoder = this.device.createCommandEncoder({
                label: "loop command encoder",
            });

            // 1a. Render Sphere to full-res targetTexture + targetDepthTexture (for visualization)
            const spherePassEncoder = commandEncoder.beginRenderPass({
                label: "sphere render pass (full res)",
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
            });
            this.sphereRenderPipelineManager.addDraw(spherePassEncoder);
            spherePassEncoder.end();

            // 1b. Render Sphere to optim-res (aspect-matched) textures for gradient computation
            const optimPassEncoder = commandEncoder.beginRenderPass({
                label: "sphere render pass (optim res)",
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
            });
            this.sphereRenderPipelineManager.addDraw(optimPassEncoder);
            optimPassEncoder.end();

            // 2. Run edge detection on optim-res depth
            this.splatOptimizerManager.dispatchEdge(commandEncoder, this.optimWidth, this.optimHeight);

            // 3. Dispatch Splat Optimizer Compute Passes (uses optim-res texture + edge map)
            this.splatOptimizerManager.dispatch(commandEncoder);

            // 4. Run edge detection on full-res depth (for display)
            this.splatOptimizerManager.setEdgeTarget(this.targetDepthTextureView!, this.fullEdgeTextureView!);
            this.splatOptimizerManager.dispatchEdge(commandEncoder, fullW, fullH);
            // Restore optim-res edge bind group for next frame
            this.splatOptimizerManager.setEdgeTarget(this.optimDepthTextureView!, this.optimEdgeTextureView!);

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