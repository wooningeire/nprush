import { onMount } from "svelte";
import { Camera } from "./Camera.svelte";
import { CameraOrbit } from "./CameraOrbit.svelte";
import { requestGpu } from "./requestGpu.ts";

export class SimulationState {
    width = $state(300);
    height = $state(150);
    
    readonly orbit = new CameraOrbit();
    readonly camera = new Camera({
        controlScheme: this.orbit,
        screenDims: { width: () => this.width, height: () => this.height },
    });

    static mount({
        canvasPromise,
    }: {
        canvasPromise: Promise<HTMLCanvasElement>,
    }) {
        const state = new SimulationState();
        
        onMount(async () => {
            const gpu = await requestGpu({canvas: await canvasPromise});
        });

        return state;
    }
}