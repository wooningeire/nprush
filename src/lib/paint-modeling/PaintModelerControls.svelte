<script lang="ts">
import { flip } from "svelte/animate";
import { circOut } from "svelte/easing";
import ConstructionPlaneControls from "./ConstructionPlaneControls.svelte";
import type { PaintModelingState } from "./PaintModelingState.svelte.ts";
import { BrushPlacementMode, type BrushPlacementMode as BrushPlacementModeValue } from "./types.ts";

let {
    modelerState,
    rendererError,
    shadeRibbons,
    requestRender,
    setShadeRibbons,
    planePickArmed,
    setPlanePickArmed,
}: {
    modelerState: PaintModelingState,
    rendererError: string | null,
    shadeRibbons: boolean,
    requestRender: () => void,
    setShadeRibbons: (value: boolean) => void,
    planePickArmed: boolean,
    setPlanePickArmed: (value: boolean) => void,
} = $props();

let sortedPaintLayers = $derived([...modelerState.paintLayers].sort((a, b) => a.order - b.order));
let sortedObjects = $derived([...modelerState.objects].sort((a, b) => a.layerIndex - b.layerIndex));
let sortedViews = $derived([...modelerState.views].sort((a, b) => a.order - b.order));

const placementModeOptions: { mode: BrushPlacementModeValue, label: string }[] = [
    { mode: BrushPlacementMode.View, label: "View" },
    { mode: BrushPlacementMode.StartDepth, label: "Start depth" },
    { mode: BrushPlacementMode.StartPlane, label: "Start plane" },
    { mode: BrushPlacementMode.Surface, label: "Surface" },
    { mode: BrushPlacementMode.ConstructionPlane, label: "Construction plane" },
];

const layerStrokeCount = (layerId: string, layerOrder: number): number =>
    modelerState.strokes.filter(stroke =>
        stroke.layerId === layerId || (!stroke.layerId && layerOrder === 0)
    ).length;

const objectStrokeCount = (objectId: string): number =>
    modelerState.strokes.filter(stroke => stroke.objectId === objectId).length;

type ReorderList = "paint-layer" | "object" | "view";

let draggingList = $state<ReorderList | null>(null);
let draggingId = $state<string | null>(null);

const reorderDraggedItem = (list: ReorderList, sourceId: string, targetId: string): boolean => {
    if (list === "paint-layer") return modelerState.reorderPaintLayer(sourceId, targetId);
    if (list === "object") return modelerState.reorderObject(sourceId, targetId);
    return modelerState.reorderView(sourceId, targetId);
};

const beginDragReorder = (list: ReorderList, id: string, event: DragEvent): void => {
    draggingList = list;
    draggingId = id;
    modelerState.beginUndoGroup();
    event.dataTransfer?.setData("text/plain", id);
    if (event.dataTransfer) event.dataTransfer.effectAllowed = "move";
};

const dragOverReorderTarget = (list: ReorderList, targetId: string, event: DragEvent): void => {
    if (draggingList !== list || !draggingId) return;
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "move";
    if (draggingId === targetId) return;

    if (reorderDraggedItem(list, draggingId, targetId)) requestRender();
};

const finishDragReorder = (): void => {
    modelerState.commitUndoGroup();
    draggingList = null;
    draggingId = null;
};

const dropReorderTarget = (event: DragEvent): void => {
    event.preventDefault();
    finishDragReorder();
};

const isDragging = (list: ReorderList, id: string): boolean => draggingList === list && draggingId === id;
</script>

<aside class="control-panel">
    <section>
        <div class="section-title">Paint Modeler</div>
        <div class="subtle">Stroke-owned ribbon prototype</div>
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
        <label class="toggle-row">
            <input
                type="checkbox"
                checked={shadeRibbons}
                onchange={(event) => {
                    setShadeRibbons((event.currentTarget as HTMLInputElement).checked);
                }}
            />
            <span>Shade ribbons</span>
        </label>
        <label class="control-row">
            <span>Paint on</span>
            <select
                aria-label="Paint on"
                value={modelerState.brushPlacementMode}
                onchange={(event) => {
                    modelerState.setBrushPlacementMode(
                        (event.currentTarget as HTMLSelectElement).value as BrushPlacementModeValue,
                    );
                    setPlanePickArmed(false);
                    requestRender();
                }}
            >
                {#each placementModeOptions as option}
                    <option value={option.mode}>{option.label}</option>
                {/each}
            </select>
        </label>
        {#if modelerState.brushPlacementMode === BrushPlacementMode.ConstructionPlane}
            <ConstructionPlaneControls
                {modelerState}
                {planePickArmed}
                {requestRender}
                {setPlanePickArmed}
            />
        {/if}        <label class="color-row">
            <span>Color</span>
            <input
                type="color"
                value={modelerState.brush.color}
                oninput={(event) => {
                    modelerState.setBrushColor((event.currentTarget as HTMLInputElement).value);
                    requestRender();
                }}
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
                oninput={(event) => {
                    modelerState.setBrushWidth(Number((event.currentTarget as HTMLInputElement).value));
                    requestRender();
                }}
            />
            <small>{Math.round(modelerState.brush.width)}</small>
        </label>
    </section>

    <div class="separator"></div>

    <section>
        <div class="section-header-row">
            <div class="section-title">Paint Layers</div>
            <button
                type="button"
                class="mini-button"
                onclick={() => {
                    modelerState.addPaintLayer();
                    requestRender();
                }}
            >Add</button>
        </div>
        <div class="list">
            {#each sortedPaintLayers as layer (layer.id)}
                <button
                    type="button"
                    class="select-row layer-row sortable-row"
                    class:active={layer.id === modelerState.activePaintLayerId}
                    class:dragging={isDragging("paint-layer", layer.id)}
                    draggable="true"
                    ondragstart={(event) => beginDragReorder("paint-layer", layer.id, event)}
                    ondragover={(event) => dragOverReorderTarget("paint-layer", layer.id, event)}
                    ondrop={dropReorderTarget}
                    ondragend={finishDragReorder}
                    animate:flip={{duration: 200, easing: circOut}}
                    onclick={() => {
                        modelerState.selectPaintLayer(layer.id);
                        requestRender();
                    }}
                >
                    <span>{layer.name}</span>
                    <small>{layerStrokeCount(layer.id, layer.order)}s</small>
                </button>
            {/each}
        </div>
    </section>

    <div class="separator"></div>

    <section>
        <div class="section-title">Objects</div>
        {#if sortedObjects.length === 0}
            <div class="subtle">No objects</div>
        {:else}
            <div class="list">
                {#each sortedObjects as object (object.id)}
                    <div
                        class="list-row sortable-row"
                        class:dragging={isDragging("object", object.id)}
                        role="group"
                        aria-label={`Drag ${object.name}`}
                        draggable="true"
                        ondragstart={(event) => beginDragReorder("object", object.id, event)}
                        ondragover={(event) => dragOverReorderTarget("object", object.id, event)}
                        ondrop={dropReorderTarget}
                        ondragend={finishDragReorder}
                        animate:flip={{duration: 200, easing: circOut}}
                    >
                        <button
                            class="select-row"
                            class:active={object.id === modelerState.activeObjectId}
                            onclick={() => {
                                modelerState.selectObject(object.id);
                                requestRender();
                            }}
                        >
                            <span>{object.name}</span>
                            <small>{objectStrokeCount(object.id)}s</small>
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
                    <div
                        class="list-row sortable-row"
                        class:dragging={isDragging("view", view.id)}
                        role="group"
                        aria-label={`Drag ${view.name}`}
                        draggable="true"
                        ondragstart={(event) => beginDragReorder("view", view.id, event)}
                        ondragover={(event) => dragOverReorderTarget("view", view.id, event)}
                        ondrop={dropReorderTarget}
                        ondragend={finishDragReorder}
                        animate:flip={{duration: 200, easing: circOut}}
                    >
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
        <span>Strokes {modelerState.strokes.length}</span>
    </section>

    {#if rendererError}
        <div class="error">{rendererError}</div>
    {/if}
</aside>

<style lang="scss">
.control-panel {
    width: 18rem;
    flex: 0 0 18rem;
    box-sizing: border-box;
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

.section-header-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: center;
    gap: 0.5rem;
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

.mini-button {
    min-height: 1.7rem;
    padding: 0 0.55rem;
    font-size: 0.72rem;
}

.color-row,
.range-row,
.control-row {
    grid-template-columns: 4.8rem minmax(0, 1fr);
}

.control-row select {
    min-width: 0;
    height: 1.8rem;
    padding: 0 0.4rem;
    border: 1px solid oklch(78% 0.018 210 / 0.22);
    border-radius: 4px;
    background: oklch(18% 0.018 210 / 0.86);
    color: oklch(92% 0.012 210);
    font: inherit;
}
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
    }
}

.range-row {
    grid-template-columns: 4.8rem minmax(0, 1fr) 2.8rem;

    input {
        width: 100%;
        accent-color: #7dddbd;
    }
}

.control-row {
    grid-template-columns: 4.8rem minmax(0, 1fr);
}

.control-row select {
    min-width: 0;
    height: 1.8rem;
    padding: 0 0.4rem;
    border: 1px solid oklch(78% 0.018 210 / 0.22);
    border-radius: 4px;
    background: oklch(18% 0.018 210 / 0.86);
    color: oklch(92% 0.012 210);
    font: inherit;
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

.sortable-row {
    user-select: none;
    cursor: grab;

    &.dragging {
        opacity: 0.58;
    }
}

.layer-row {
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
@media (max-width: 48rem) {
    .control-panel {
        width: 100%;
        max-height: 44vh;
        flex: 0 0 auto;
        border-right: 0;
        border-bottom: 1px solid oklch(78% 0.018 210 / 0.2);
    }
}

</style>