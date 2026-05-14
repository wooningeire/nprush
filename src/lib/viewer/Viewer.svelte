<script lang="ts">
import { onMount } from "svelte";

import Canvas from "./Canvas.svelte";
import ControlPanel from "./ControlPanel.svelte";
import Toasts from "./Toasts.svelte";
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
    <viewer-content>
        <ControlPanel {viewerState} />

        <Canvas
            {viewerState}
            bind:canvas
        />
    </viewer-content>

    <Toasts />
</main>


<style lang="scss">
main {
    width: 100vw;
    height: 100vh;

    display: grid;
    align-items: stretch;

    overflow: hidden;

    > :global(*) {
        grid-area: 1/1;
    }
}

viewer-content {
    display: flex;
}
</style>