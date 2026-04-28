import { onDestroy, onMount } from "svelte";
import { Camera } from "./Camera.svelte.ts";
import { CameraOrbit } from "./CameraOrbit.svelte.ts";
import { requestGpu } from "$/gpu/requestGpu";
import { GpuRunner } from "./GpuRunner.svelte.ts";
import { loadGlb } from "$/gpu/loadGlb";
import artelorianUrl from "$/assets/artelorian.glb?url";
import matcapUrl from "$/assets/overcast_soil_puresky_2k.png?url";
import { loadTexture } from "$/gpu/loadTexture";

export class ViewerState {
    width = $state(300);
    height = $state(150);
    edgeBeziersEnabled = $state(false);
    colorBeziersEnabled = $state(false);
    shadingMode = $state<'normals' | 'shaded'>('normals');
    
    bezierRegEnabled = $state(false);
    bezierRegWidth = $state(0.003);
    bezierRegSoftness = $state(0.001);

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
        
        let stopLoop: (() => void) | null = null;
        onMount(async () => {
            // Kick off mesh load and gpu request concurrently; both are awaited
            // before we build the runner since the mesh is a constructor input.
            const [gpu, mesh] = await Promise.all([
                requestGpu({ canvas: await canvasPromise }),
                loadGlb(artelorianUrl),
            ]);
            if (!gpu) return;

            const matcapTexture = await loadTexture(gpu.device, matcapUrl);

            const gpuRunner = new GpuRunner({
                device: gpu.device,
                context: gpu.context,
                format: gpu.format,
                camera: state.camera,
                viewerState: state,
                mesh,
                matcapTexture,
                numSplats,
            });
            state.runner = gpuRunner;

            stopLoop = gpuRunner.loop();
        });

        onDestroy(() => {
            stopLoop?.();
            state.runner?.destroy();
        });

        return state;
    }
}