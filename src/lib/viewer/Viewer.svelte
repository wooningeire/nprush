<script lang="ts">
import { onMount } from "svelte";

import Canvas from "./Canvas.svelte";
import { ViewerState } from "./ViewerState.svelte.ts";

let { numSplats = 1024 }: { numSplats?: number } = $props();

let canvas = $state<HTMLCanvasElement | null>(null);
let canvasPromise = Promise.withResolvers<HTMLCanvasElement>();

const viewerState = ViewerState.mount({
    canvasPromise: canvasPromise.promise,
    numSplats,
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