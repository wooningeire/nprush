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
        <input type="checkbox" bind:checked={viewerState.edgeBeziersEnabled} />
        Edge Beziers
    </label>
    {#if viewerState.edgeBeziersEnabled}
        <label class="sub">
            <input type="checkbox" bind:checked={viewerState.edgeBezierTrainingPaused} />
            Pause training
        </label>
    {/if}
    <label>
        <input type="checkbox" bind:checked={viewerState.baseColorBeziersEnabled} />
        Base Color Beziers
    </label>
    {#if viewerState.baseColorBeziersEnabled}
        <label class="sub">
            <input type="checkbox" bind:checked={viewerState.baseColorBezierTrainingPaused} />
            Pause training
        </label>
    {/if}
    <label>
        <input type="checkbox" bind:checked={viewerState.colorBeziersEnabled} />
        Color Beziers
    </label>
    {#if viewerState.colorBeziersEnabled}
        <label class="sub">
            <input type="checkbox" bind:checked={viewerState.colorBezierTrainingPaused} />
            Pause training
        </label>
    {/if}
    <label>
        <input type="checkbox" bind:checked={viewerState.splatTrainingPaused} />
        Pause Splat Training
    </label>
    <label>
        <input type="checkbox" bind:checked={viewerState.posterizationEnabled} />
        Posterization
    </label>
    <label>
        <input type="checkbox" bind:checked={viewerState.compareBlurred} />
        Compare Blurred
    </label>
    {#if viewerState.compareBlurred}
        <div class="slider-group" style="margin-left: 1rem; margin-bottom: 0.5rem;">
            <label style="font-size: 0.8rem; color: rgba(255, 255, 255, 0.7);">
                Radius: {viewerState.blurRadius}
                <input type="range" min="1" max="64" step="1" bind:value={viewerState.blurRadius} />
            </label>
        </div>
    {/if}
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

            &.sub {
                margin-left: 1.25rem;
                font-size: 0.8rem;
                color: rgba(255, 255, 255, 0.65);
            }
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