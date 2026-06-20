<script lang="ts">
import type { PaintModelingState } from "./PaintModelingState.svelte.ts";

let {
    modelerState,
    rendererError,
    showChartWireframe,
    showSurfaceField,
    requestRender,
    setShowChartWireframe,
    setShowSurfaceField,
}: {
    modelerState: PaintModelingState,
    rendererError: string | null,
    showChartWireframe: boolean,
    showSurfaceField: boolean,
    requestRender: () => void,
    setShowChartWireframe: (value: boolean) => void,
    setShowSurfaceField: (value: boolean) => void,
} = $props();

let sortedObjects = $derived([...modelerState.objects].sort((a, b) => a.layerIndex - b.layerIndex));
let sortedViews = $derived([...modelerState.views].sort((a, b) => a.createdAt - b.createdAt));
</script>

<aside class="control-panel">
    <section>
        <div class="section-title">Paint Modeler</div>
        <div class="subtle">Surface paint prototype</div>
    </section>

    <div class="separator"></div>

    <section class="button-row">
        <button onclick={() => {
            modelerState.addObject();
            requestRender();
        }}>Add Object</button>
        <button onclick={() => {
            modelerState.undo();
            requestRender();
        }} disabled={!modelerState.canUndo}>Undo</button>
    </section>

    <div class="separator"></div>

    <section>
        <div class="section-title">Brush</div>
        <div class="mode-row" role="group" aria-label="Brush mode">
            <button
                type="button"
                class:active={modelerState.brushMode === "color"}
                aria-pressed={modelerState.brushMode === "color"}
                onclick={() => {
                    modelerState.setBrushMode("color");
                    requestRender();
                }}
            >Color</button>
            <button
                type="button"
                class:active={modelerState.brushMode === "surface"}
                aria-pressed={modelerState.brushMode === "surface"}
                onclick={() => {
                    modelerState.setBrushMode("surface");
                    requestRender();
                }}
            >Surface</button>
        </div>
        <label class="color-row">
            <span>Color</span>
            <input
                type="color"
                value={modelerState.brush.color}
                disabled={modelerState.brushMode === "surface"}
                oninput={(event) => modelerState.setBrushColor((event.currentTarget as HTMLInputElement).value)}
            />
        </label>
        <label class="range-row">
            <span>Width</span>
            <input
                type="range"
                min="1"
                max="72"
                step="1"
                value={modelerState.brush.width}
                oninput={(event) => modelerState.setBrushWidth(Number((event.currentTarget as HTMLInputElement).value))}
            />
            <small>{Math.round(modelerState.brush.width)}</small>
        </label>
        <label class="toggle-row">
            <input
                type="checkbox"
                checked={showChartWireframe}
                onchange={(event) => setShowChartWireframe((event.currentTarget as HTMLInputElement).checked)}
            />
            <span>Chart wire</span>
        </label>
        <label class="toggle-row">
            <input
                type="checkbox"
                checked={showSurfaceField}
                onchange={(event) => setShowSurfaceField((event.currentTarget as HTMLInputElement).checked)}
            />
            <span>Surface field</span>
        </label>
    </section>

    <div class="separator"></div>

    <section>
        <div class="section-title">Objects</div>
        {#if sortedObjects.length === 0}
            <div class="subtle">No objects</div>
        {:else}
            <div class="list">
                {#each sortedObjects as object (object.id)}
                    <div class="list-row">
                        <button
                            class="select-row"
                            class:active={object.id === modelerState.activeObjectId}
                            onclick={() => {
                                modelerState.selectObject(object.id);
                                requestRender();
                            }}
                        >
                            <span>{object.name}</span>
                            <small>{object.charts.length}c {modelerState.strokes.filter(stroke => stroke.objectId === object.id).length}s</small>
                        </button>
                        <button
                            class="delete-row"
                            title={`Delete ${object.name}`}
                            onclick={() => {
                                modelerState.deleteObject(object.id);
                                requestRender();
                            }}
                        >
                            Delete
                        </button>
                    </div>
                {/each}
            </div>
        {/if}
    </section>

    <div class="separator"></div>

    <section>
        <div class="section-title">Views</div>
        {#if sortedViews.length === 0}
            <div class="subtle">No saved views</div>
        {:else}
            <div class="list">
                {#each sortedViews as view (view.id)}
                    <div class="list-row">
                        <button
                            class="select-row"
                            class:active={view.id === modelerState.activeViewId && modelerState.isCameraAtActiveView}
                            onclick={() => {
                                modelerState.selectView(view.id);
                                requestRender();
                            }}
                        >
                            <span>{view.name}</span>
                            <small>{view.width}x{view.height}</small>
                        </button>
                        <button
                            class="delete-row"
                            title={`Delete ${view.name}`}
                            onclick={() => {
                                modelerState.deleteView(view.id);
                                requestRender();
                            }}
                        >
                            Delete
                        </button>
                    </div>
                {/each}
            </div>
        {/if}
    </section>

    <div class="separator"></div>

    <section class="stats">
        <span>Charts {modelerState.chartCount}</span>
        <span>Claims {modelerState.occlusionClaims.length}</span>
    </section>

    {#if rendererError}
        <div class="error">{rendererError}</div>
    {/if}
</aside>

<style lang="scss">
.control-panel {
    width: 18rem;
    flex: 0 0 18rem;
    overflow-y: auto;
    padding: 1rem;
    border-right: 1px solid rgba(255, 255, 255, 0.12);
    background: rgba(11, 14, 16, 0.76);

    section {
        display: flex;
        flex-direction: column;
        gap: 0.45rem;
    }
}

.section-title {
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: rgba(255, 255, 255, 0.62);
    font-size: 0.78rem;
}

.subtle,
small,
.stats {
    color: rgba(255, 255, 255, 0.48);
    font-size: 0.78rem;
}

.stats {
    gap: 0.25rem !important;
}

.separator {
    height: 1px;
    background: rgba(255, 255, 255, 0.14);
    margin: 0.75rem 0;
}

button {
    min-height: 2rem;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.16);
    background: rgba(255, 255, 255, 0.08);
    color: rgba(255, 255, 255, 0.86);
    cursor: pointer;

    &:hover:not(:disabled) {
        background: rgba(255, 255, 255, 0.14);
    }

    &:disabled {
        opacity: 0.42;
        cursor: not-allowed;
    }
}

.button-row {
    display: grid !important;
    grid-template-columns: 1fr 1fr;
    gap: 0.45rem;
}

.mode-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.35rem;

    button {
        min-width: 0;
        padding: 0.35rem 0.5rem;

        &.active {
            border-color: oklch(78% 0.08 185 / 0.65);
            background: oklch(56% 0.07 190 / 0.28);
            color: oklch(94% 0.03 185);
        }
    }
}

.color-row,
.range-row,
.toggle-row {
    display: grid;
    align-items: center;
    gap: 0.5rem;
    color: rgba(255, 255, 255, 0.66);
    font-size: 0.78rem;
}

.color-row {
    grid-template-columns: 4.8rem 3rem;

    input {
        width: 2.5rem;
        height: 1.7rem;
        padding: 0;
        border: 0;
        background: transparent;

        &:disabled {
            opacity: 0.45;
            cursor: not-allowed;
        }
    }
}

.range-row {
    grid-template-columns: 4.8rem minmax(0, 1fr) 2.8rem;

    input {
        width: 100%;
        accent-color: #7dddbd;
    }
}

.toggle-row {
    grid-template-columns: auto 1fr;

    input {
        accent-color: #7dddbd;
    }
}

.list {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
}

.list-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 4.2rem;
    gap: 0.35rem;

    .select-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        min-width: 0;
        gap: 0.5rem;
        padding: 0.35rem 0.5rem;
        text-align: left;

        span,
        small {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        &.active {
            background: rgba(93, 139, 179, 0.28);
            border-color: rgba(128, 179, 221, 0.48);
        }
    }

    .delete-row {
        min-height: 2rem;
        font-size: 0.72rem;
        color: rgba(255, 205, 196, 0.9);
        border-color: rgba(255, 154, 135, 0.22);
        background: rgba(110, 31, 26, 0.22);
    }
}

.error {
    margin-top: 0.85rem;
    color: #ffb4a8;
    font-size: 0.82rem;
}
</style>
