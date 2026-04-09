import type { Camera } from "./Camera.svelte";
import { GpuUniformsBufferManager } from "$/gpu/GpuUniformsBufferManager";
import { GpuSphereRenderPipelineManager } from "$/gpu/GpuSphereRenderPipelineManager";

export class GpuRunner {
    private readonly device: GPUDevice;
    private readonly context: GPUCanvasContext;
    private readonly format: GPUTextureFormat;
    private readonly camera: Camera;

    readonly uniformsManager: GpuUniformsBufferManager;
    readonly sphereRenderPipelineManager: GpuSphereRenderPipelineManager;

    constructor({
        device,
        context,
        format,
        camera,
    }: {
        device: GPUDevice,
        context: GPUCanvasContext,
        format: GPUTextureFormat,
        camera: Camera,
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

        $effect.root(() => {
            $effect(() => this.uniformsManager.writeViewProjMat(this.camera.viewProjMat));
        });
    }

    loop() {
        let handle = 0;
        let canceled = false;

        const loop = () => {
            const commandEncoder = this.device.createCommandEncoder({
                label: "loop command encoder",
            });

            const screenView = this.context.getCurrentTexture().createView();

            const renderPassEncoder = commandEncoder.beginRenderPass({
                label: "sphere render pass",
                colorAttachments: [
                    {
                        clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
                        loadOp: "clear",
                        storeOp: "store",
                        view: screenView,
                    },
                ],
            });

            this.sphereRenderPipelineManager.addDraw(renderPassEncoder);

            renderPassEncoder.end();

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