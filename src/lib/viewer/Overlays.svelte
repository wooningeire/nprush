<script lang="ts">
import type { ViewerState } from "./ViewerState.svelte";

const {
    viewerState,
}: {
    viewerState: ViewerState;
} = $props();
</script>

<div class="overlays">
    <label>
        <input type="checkbox" bind:checked={viewerState.beziersEnabled} />
        Beziers
    </label>
</div>

<style lang="scss">
    .overlays {
        position: absolute;
        top: 1rem;
        left: 1rem;
        z-index: 100;
        background: rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(8px);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        color: white;
        font-family: sans-serif;
        font-size: 0.9rem;
        border: 1px solid rgba(255, 255, 255, 0.1);

        label {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            cursor: pointer;
            user-select: none;
        }

        input {
            cursor: pointer;
        }
    }
</style>