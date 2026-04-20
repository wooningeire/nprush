import type { Camera } from "./Camera.svelte";
import { GpuUniformsBufferManager } from "$/gpu/GpuUniformsBufferManager";
import { GpuSphereRenderPipelineManager } from "$/gpu/GpuSphereRenderPipelineManager";
import { GpuSplatOptimizerManager } from "$/gpu/GpuSplatOptimizerManager";

const OPTIM_RES = 128;

export class GpuRunner {
    private readonly device: GPUDevice;
    private readonly context: GPUCanvasContext;
    private readonly format: GPUTextureFormat;
    private readonly camera: Camera;

    readonly uniformsManager: GpuUniformsBufferManager;
    readonly sphereRenderPipelineManager: GpuSphereRenderPipelineManager;
    readonly splatOptimizerManager: GpuSplatOptimizerManager;

    private targetTexture: GPUTexture | null = null;
    private targetTextureView: GPUTextureView | null = null;
    private targetDepthTexture: GPUTexture | null = null;
    private targetDepthTextureView: GPUTextureView | null = null;

    // Small resolution textures for gradient computation
    private optimTexture: GPUTexture;
    private optimTextureView: GPUTextureView;
    private optimDepthTexture: GPUTexture;
    private optimDepthTextureView: GPUTextureView;

    // Edge map textures (optim-res for loss, full-res for display)
    private optimEdgeTexture: GPUTexture;
    private optimEdgeTextureView: GPUTextureView;
    private fullEdgeTexture: GPUTexture | null = null;
    private fullEdgeTextureView: GPUTextureView | null = null;

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

        // Create fixed small-res color texture for gradient computation
        this.optimTexture = device.createTexture({
            label: "optimization target texture",
            size: [OPTIM_RES, OPTIM_RES],
            format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimTextureView = this.optimTexture.createView();

        // Create fixed small-res depth texture for gradient computation
        this.optimDepthTexture = device.createTexture({
            label: "optimization depth texture",
            size: [OPTIM_RES, OPTIM_RES],
            format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimDepthTextureView = this.optimDepthTexture.createView();

        // Edge map at optim resolution (writable storage texture)
        this.optimEdgeTexture = device.createTexture({
            label: "optimization edge texture",
            size: [OPTIM_RES, OPTIM_RES],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimEdgeTextureView = this.optimEdgeTexture.createView();

        // Wire optim-res edge detection
        this.splatOptimizerManager.setEdgeTarget(this.optimDepthTextureView, this.optimEdgeTextureView);
        this.splatOptimizerManager.setBackwardTarget(this.optimTextureView, this.optimEdgeTextureView, OPTIM_RES, OPTIM_RES);

        this.destroy = $effect.root(() => {
            $effect(() => this.uniformsManager.writeViewProjMat(this.camera.viewProjMat));
        });
    }

    loop() {
        let handle = 0;
        let canceled = false;

        const loop = () => {
            const currentTexture = this.context.getCurrentTexture();
            const width = currentTexture.width;
            const height = currentTexture.height;

            // Full-res target for visualization
            if (!this.targetTexture || this.targetTexture.width !== width || this.targetTexture.height !== height) {
                if (this.targetTexture) this.targetTexture.destroy();
                if (this.targetDepthTexture) this.targetDepthTexture.destroy();
                if (this.fullEdgeTexture) this.fullEdgeTexture.destroy();

                this.targetTexture = this.device.createTexture({
                    size: [width, height],
                    format: this.format,
                    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
                });
                this.targetTextureView = this.targetTexture.createView();

                this.targetDepthTexture = this.device.createTexture({
                    size: [width, height],
                    format: this.format,
                    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
                });
                this.targetDepthTextureView = this.targetDepthTexture.createView();

                this.fullEdgeTexture = this.device.createTexture({
                    size: [width, height],
                    format: "rgba8unorm",
                    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
                });
                this.fullEdgeTextureView = this.fullEdgeTexture.createView();

                this.splatOptimizerManager.setRenderTarget(this.targetTextureView, this.targetDepthTextureView, this.fullEdgeTextureView);
            }

            if (!this.targetTextureView || !this.targetDepthTextureView) {
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

            // 1b. Render Sphere to small-res optimTexture + optimDepthTexture (for gradient computation)
            const optimPassEncoder = commandEncoder.beginRenderPass({
                label: "sphere render pass (optim res)",
                colorAttachments: [
                    {
                        clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
                        loadOp: "clear",
                        storeOp: "store",
                        view: this.optimTextureView,
                    },
                    {
                        clearValue: { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
                        loadOp: "clear",
                        storeOp: "store",
                        view: this.optimDepthTextureView,
                    },
                ],
            });
            this.sphereRenderPipelineManager.addDraw(optimPassEncoder);
            optimPassEncoder.end();

            // 2. Run edge detection on optim-res depth
            this.splatOptimizerManager.dispatchEdge(commandEncoder);

            // 3. Dispatch Splat Optimizer Compute Passes (uses small-res texture + edge map)
            this.splatOptimizerManager.dispatch(commandEncoder);

            // 4. Run edge detection on full-res depth (for display)
            // Temporarily swap the edge bind group for full-res
            this.splatOptimizerManager.setEdgeTarget(this.targetDepthTextureView!, this.fullEdgeTextureView!);
            this.splatOptimizerManager.dispatchEdge(commandEncoder);
            // Restore optim-res edge bind group for next frame
            this.splatOptimizerManager.setEdgeTarget(this.optimDepthTextureView, this.optimEdgeTextureView);

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