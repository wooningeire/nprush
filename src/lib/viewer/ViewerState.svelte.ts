import { onDestroy, onMount } from "svelte";
import { Camera } from "./Camera.svelte.ts";
import { CameraOrbit } from "./CameraOrbit.svelte.ts";
import { requestGpu } from "$/gpu/requestGpu.ts";
import { GpuRunner } from "./GpuRunner.svelte.ts";

export class ViewerState {
    width = $state(300);
    height = $state(150);

    runner = $state<GpuRunner | null>(null);
    
    readonly orbit = new CameraOrbit();
    readonly camera = new Camera({
        controlScheme: this.orbit,
        screenDims: { width: () => this.width, height: () => this.height },
    });

    static mount({
        canvasPromise,
        numSplats = 512,
    }: {
        canvasPromise: Promise<HTMLCanvasElement>,
        numSplats?: number,
    }) {
        const state = new ViewerState();
        
        onMount(async () => {
            const gpu = await requestGpu({canvas: await canvasPromise});
            if (!gpu) return;

            const gpuRunner = new GpuRunner({
                device: gpu.device,
                context: gpu.context,
                format: gpu.format,
                camera: state.camera,
                numSplats,
            });
            state.runner = gpuRunner;

            const stopLoop = gpuRunner.loop();

            return () => {
                stopLoop();
            };
        });

        onDestroy(() => {
            state.runner?.destroy();
        });

        return state;
    }
}