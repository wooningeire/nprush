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
    <div class="shading-toggle">
        <button 
            class:active={viewerState.shadingMode === 'normals'} 
            onclick={() => viewerState.shadingMode = 'normals'}
        >
            Normals
        </button>
        <button 
            class:active={viewerState.shadingMode === 'shaded'} 
            onclick={() => viewerState.shadingMode = 'shaded'}
        >
            Shaded
        </button>
    </div>
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

        .shading-toggle {
            display: flex;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            padding: 2px;
            margin-top: 0.5rem;

            button {
                flex: 1;
                background: transparent;
                border: none;
                color: rgba(255, 255, 255, 0.5);
                font-size: 0.8rem;
                padding: 4px 12px;
                border-radius: 3px;
                cursor: pointer;
                transition: all 0.2s;

                &.active {
                    background: rgba(255, 255, 255, 0.2);
                    color: white;
                }

                &:hover:not(.active) {
                    color: rgba(255, 255, 255, 0.8);
                }
            }
        }
    }
</style>