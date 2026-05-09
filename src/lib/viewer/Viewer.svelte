<script lang="ts">
import { onMount } from "svelte";

import Canvas from "./Canvas.svelte";
import GpuProfilerHud from "./GpuProfilerHud.svelte";
import Overlays from "./Overlays.svelte";
import { ViewerState } from "./ViewerState.svelte.ts";

let canvas = $state<HTMLCanvasElement | null>(null);
let canvasPromise = Promise.withResolvers<HTMLCanvasElement>();

const viewerState = ViewerState.mount({
    canvasPromise: canvasPromise.promise,
});

onMount(() => {
    canvasPromise.resolve(canvas!);
});
</script>

<main>
    <Canvas
        {viewerState}
        bind:canvas
    />

    <Overlays {viewerState} />
    <GpuProfilerHud {viewerState} />
</main>


<style lang="scss">
main {
    width: 100vw;
    height: 100vh;

    display: grid;

    > :global(*) {
        grid-area: 1/1;
    }
}
</style>