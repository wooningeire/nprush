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
    <label>
        <input type="checkbox" bind:checked={viewerState.colorBeziersEnabled} />
        Color Beziers
    </label>
    <label>
        <input type="checkbox" bind:checked={viewerState.compareBlurred} />
        Compare Blurred
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

    <div class="reg-controls">
        <label>
            <input type="checkbox" bind:checked={viewerState.bezierRegEnabled} />
            Regularize Beziers
        </label>
        
        {#if viewerState.bezierRegEnabled}
            <div class="slider-group">
                <label>
                    Width: {viewerState.bezierRegWidth.toFixed(4)}
                    <input type="range" min="0.0001" max="0.01" step="0.0001" bind:value={viewerState.bezierRegWidth} />
                </label>
            </div>
            <div class="slider-group">
                <label>
                    Softness: {viewerState.bezierRegSoftness.toFixed(4)}
                    <input type="range" min="0.0001" max="0.01" step="0.0001" bind:value={viewerState.bezierRegSoftness} />
                </label>
            </div>
        {/if}
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

        .reg-controls {
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            
            .slider-group {
                display: flex;
                flex-direction: column;
                gap: 0.2rem;
                
                label {
                    display: flex;
                    flex-direction: column;
                    align-items: flex-start;
                    font-size: 0.8rem;
                    gap: 0.2rem;
                    color: rgba(255, 255, 255, 0.8);
                }

                input[type="range"] {
                    width: 100%;
                }
            }
        }
    }
</style>