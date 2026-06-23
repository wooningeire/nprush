<script lang="ts">
import { onDestroy, onMount, untrack } from "svelte";

import ViewPanelGrid from "./ViewPanelGrid.svelte";
import ControlPanel from "./ControlPanel.svelte";
import { ViewerState } from "./ViewerState.svelte.ts";

let canvases = $state<Record<string, HTMLCanvasElement>>({});
let canvasesPromise = Promise.withResolvers<Record<string, HTMLCanvasElement>>();

const viewerState = ViewerState.mount({
    canvasesPromise: canvasesPromise.promise,
});

onMount(() => {
    canvasesPromise.resolve(canvases);
});

let lastStopLoop: (() => void) | null = null;
let isInitial = true;

$effect(() => {
    if (isInitial) {
        isInitial = false;
        return;
    }

    untrack(() => {
        lastStopLoop = viewerState.runner?.loop() ?? null;
    });
})

onDestroy(() => {
    lastStopLoop?.();
});
</script>

<brushstroke-optimizer-content>
    <ControlPanel {viewerState} />

    <ViewPanelGrid
        {viewerState}
        bind:canvases
    />
</brushstroke-optimizer-content>

<style lang="scss">
brushstroke-optimizer-content {
    flex-grow: 1;

    display: flex;
    align-items: stretch;
    
    overflow: hidden;
}
</style>
