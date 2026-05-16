<script lang="ts">
import Toasts from "./Toasts.svelte";
import BrushstrokeOptimizer from "./BrushstrokeOptimizer.svelte";
import MaterialCreator from "./material-creator/MaterialCreator.svelte";

type ViewMode = "brushstroke-optimizer" | "material-creator";
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
        </mode-bar>

        <view-mode-container
            class:visible={mode === "brushstroke-optimizer"}
        >
            <BrushstrokeOptimizer
                active={mode === "brushstroke-optimizer"}
            />
        </view-mode-container>

        <view-mode-container
            class:visible={mode === "material-creator"}
        >
            <MaterialCreator
                active={mode === "material-creator"}
            />
        </view-mode-container>
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
}

view-mode-container {
    &:not(.visible) {
        display: none;
    }

    &.visible {
        display: contents;
    }
}
</style>