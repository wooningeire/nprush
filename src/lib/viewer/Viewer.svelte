<script lang="ts">
import { onMount } from "svelte";

import ViewPanelGrid from "./ViewPanelGrid.svelte";
import ControlPanel from "./ControlPanel.svelte";
import Toasts from "./Toasts.svelte";
import { ViewerState } from "./ViewerState.svelte.ts";

let canvases = $state<Record<string, HTMLCanvasElement>>({});
let canvasesPromise = Promise.withResolvers<Record<string, HTMLCanvasElement>>();

const viewerState = ViewerState.mount({
    canvasesPromise: canvasesPromise.promise,
});

onMount(() => {
    canvasesPromise.resolve(canvases);
});
</script>

<main>
    <viewer-content>
        <ControlPanel {viewerState} />

        <ViewPanelGrid
            {viewerState}
            bind:canvases
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
    align-items: stretch;
    

    overflow: hidden;
}
</style>