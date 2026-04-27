import { onDestroy, onMount } from "svelte";
import { Camera } from "./Camera.svelte.ts";
import { CameraOrbit } from "./CameraOrbit.svelte.ts";
import { requestGpu } from "$/gpu/requestGpu.ts";
import { GpuRunner } from "./GpuRunner.svelte.ts";
import { loadGlb } from "$/gpu/loadGlb.ts";
import artelorianUrl from "$/assets/artelorian.glb?url";

export class ViewerState {
    width = $state(300);
    height = $state(150);
    beziersEnabled = $state(false);

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
            // Kick off mesh load and gpu request concurrently; both are awaited
            // before we build the runner since the mesh is a constructor input.
            const [gpu, mesh] = await Promise.all([
                requestGpu({ canvas: await canvasPromise }),
                loadGlb(artelorianUrl),
            ]);
            if (!gpu) return;

            const gpuRunner = new GpuRunner({
                device: gpu.device,
                context: gpu.context,
                format: gpu.format,
                camera: state.camera,
                viewerState: state,
                mesh,
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