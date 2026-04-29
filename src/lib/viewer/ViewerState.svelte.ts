import { onDestroy, onMount } from "svelte";
import { Camera } from "./Camera.svelte.ts";
import { CameraOrbit } from "./CameraOrbit.svelte.ts";
import { requestGpu } from "$/gpu/requestGpu";
import { GpuRunner } from "./GpuRunner.svelte.ts";
import { loadGlb } from "$/gpu/loadGlb";
import artelorianUrl from "$/assets/artelorian.glb?url";
import groundUrl from "$/assets/ground.glb?url";
import hdrUrl from "$/assets/quarry_cloudy_2k.hdr?url";
import brushUrl from "$/assets/brush.png?url";
import { loadHdrTexture } from "$/gpu/loadHdrTexture";
import { loadTexture } from "$/gpu/loadTexture";

export class ViewerState {
    width = $state(300);
    height = $state(150);
    edgeBeziersEnabled = $state(false);
    baseColorBeziersEnabled = $state(false);
    colorBeziersEnabled = $state(false);
    splatTrainingPaused = $state(false);
    edgeBezierTrainingPaused = $state(false);
    baseColorBezierTrainingPaused = $state(false);
    colorBezierTrainingPaused = $state(false);
    posterizationEnabled = $state(false);
    compareBlurred = $state(false);
    shadingMode = $state<'normals' | 'shaded'>('normals');
    blurRadius = $state(16);
    
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
            const [gpu, mesh, groundMesh] = await Promise.all([
                requestGpu({ canvas: await canvasPromise }),
                loadGlb(artelorianUrl),
                loadGlb(groundUrl, false, [1, 1, 1, 0]).catch(() => null), // keep world-space; a=0 flags as specular mirror
            ]);
            if (!gpu) return;

            const [envTexture, brushTexture] = await Promise.all([
                loadHdrTexture(gpu.device, hdrUrl),
                loadTexture(gpu.device, brushUrl),
            ]);

            const gpuRunner = new GpuRunner({
                device: gpu.device,
                context: gpu.context,
                format: gpu.format,
                camera: state.camera,
                viewerState: state,
                mesh,
                groundMesh,
                matcapTexture: envTexture,
                brushTexture,
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