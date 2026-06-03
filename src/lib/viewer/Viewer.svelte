<script lang="ts">
import Toasts from "./Toasts.svelte";
import BrushstrokeOptimizer from "./BrushstrokeOptimizer.svelte";
import MaterialCreator from "./material-creator/MaterialCreator.svelte";
import ContourModeler from "$/contour-modeler/ContourModeler.svelte";
import PaintModeler from "$/paint-modeling/PaintModeler.svelte";

type ViewMode = "brushstroke-optimizer" | "material-creator" | "contour-modeler" | "paint-modeler";
let mode = $state<ViewMode>("brushstroke-optimizer");
</script>

<main>
    <main-content>
        <mode-bar>
            <button
                class:active={mode === "brushstroke-optimizer"}
                onclick={() => mode = "brushstroke-optimizer"}
            >
                Brushstroke optimizer
            </button>

            <button
                class:active={mode === "material-creator"}
                onclick={() => mode = "material-creator"}
            >
                Materials
            </button>

            <button
                class:active={mode === "contour-modeler"}
                onclick={() => mode = "contour-modeler"}
            >
                Contour Modeler
            </button>

            <button
                class:active={mode === "paint-modeler"}
                onclick={() => mode = "paint-modeler"}
            >
                Paint Modeler
            </button>
        </mode-bar>

        {#if mode === "brushstroke-optimizer"}
            <view-mode-container>
                <BrushstrokeOptimizer active />
            </view-mode-container>
        {:else if mode === "material-creator"}
            <view-mode-container>
                <MaterialCreator active />
            </view-mode-container>
        {:else if mode === "contour-modeler"}
            <view-mode-container>
                <ContourModeler active />
            </view-mode-container>
        {:else}
            <view-mode-container>
                <PaintModeler active />
            </view-mode-container>
        {/if}
    </main-content>

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

main-content {
    display: flex;
    flex-direction: column;
    align-items: stretch;

    overflow: hidden;
}

view-mode-container {
    overflow: hidden;
    display: contents;
}
</style>
