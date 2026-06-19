<script lang="ts">
import { onDestroy } from "svelte";
import { PaintModelingRenderer } from "./PaintModelingRenderer.ts";
import { PaintModelingState } from "./PaintModelingState.svelte.ts";
import { clampNdcPoint, ndcFromClientPoint } from "../contour-modeler/contourGeometry.ts";
import type { Vec2 } from "./types.ts";

let {
    active,
}: {
    active: boolean,
} = $props();

const modelerState = new PaintModelingState();

let canvas = $state<HTMLCanvasElement | null>(null);
let viewportWidth = $state(1);
let viewportHeight = $state(1);
let renderer = $state<PaintModelingRenderer | null>(null);
let renderFrameId: number | null = null;
let uploadedStaticSceneKey: string | null = null;
let uploadedDraftKey: string | null = null;
let pointerMode = $state<"paint" | "orbit" | null>(null);
let rendererError = $state<string | null>(null);
let showChartWireframe = $state(true);
let showSurfaceField = $state(false);

let sortedObjects = $derived([...modelerState.objects].sort((a, b) => a.layerIndex - b.layerIndex));
let sortedViews = $derived([...modelerState.views].sort((a, b) => a.createdAt - b.createdAt));

$effect(() => {
    if (!canvas) return;
    const dpr = typeof window === "undefined" ? 1 : window.devicePixelRatio;
    canvas.width = Math.max(1, Math.floor(viewportWidth * dpr));
    canvas.height = Math.max(1, Math.floor(viewportHeight * dpr));
    modelerState.viewportWidth = viewportWidth;
    modelerState.viewportHeight = viewportHeight;
    requestRender();
});

$effect(() => {
    modelerState.meshVersion;
    modelerState.activeViewId;
    modelerState.activeObjectId;
    modelerState.draftStroke;
    modelerState.brush;
    showChartWireframe;
    showSurfaceField;
    requestRender();
});

$effect(() => {
    if (active) {
        requestRender();
    } else {
        cancelRender();
    }
});

onDestroy(() => {
    cancelRender();
    renderer?.destroy();
    renderer = null;
    uploadedStaticSceneKey = null;
    uploadedDraftKey = null;
});

function ensureRenderer() {
    if (renderer || !canvas) return;
    rendererError = null;
    try {
        renderer = PaintModelingRenderer.create(canvas);
        uploadedStaticSceneKey = null;
        uploadedDraftKey = null;
    } catch (error) {
        rendererError = (error as Error)?.message ?? String(error);
    }
}

function requestRender() {
    if (!active || renderFrameId !== null) return;
    renderFrameId = requestAnimationFrame(() => {
        renderFrameId = null;
        render();
    });
}

function render() {
    if (!active) return;
    ensureRenderer();
    if (!renderer) return;

    const staticSceneKey = [
        modelerState.meshVersion,
        showChartWireframe ? "wire" : "no-wire",
        showSurfaceField ? "field" : "no-field",
    ].join(":");
    if (uploadedStaticSceneKey !== staticSceneKey) {
        renderer.setSegments(modelerState.buildRenderSegments({
            showChartWireframe,
            showSurfaceField,
            showDraftStroke: false,
        }));
        uploadedStaticSceneKey = staticSceneKey;
    }

    const draftKey = draftRenderKey();
    if (uploadedDraftKey !== draftKey) {
        renderer.setDraftSegments(modelerState.buildDraftRenderSegments());
        uploadedDraftKey = draftKey;
    }
    renderer.render(modelerState.camera.viewProjMat, modelerState.camera.viewProjInvMat);
}

function cancelRender() {
    if (renderFrameId !== null) {
        cancelAnimationFrame(renderFrameId);
        renderFrameId = null;
    }
}

function onPointerDown(event: PointerEvent) {
    if (!active) return;
    const target = event.currentTarget as HTMLElement;
    target.setPointerCapture(event.pointerId);
    const point = pointerNdc(event, target);

    if (event.button === 0) {
        modelerState.beginStroke(point, target.clientWidth, target.clientHeight);
        pointerMode = "paint";
    } else if (event.button === 1) {
        pointerMode = "orbit";
    }

    requestRender();
    event.preventDefault();
}

function onPointerMove(event: PointerEvent) {
    if (!active) return;
    const target = event.currentTarget as HTMLElement;
    const point = pointerNdc(event, target);

    if (pointerMode === null) return;

    if (pointerMode === "paint") {
        modelerState.appendStrokePoint(point);
    } else {
        const movement = { x: event.movementX, y: event.movementY };
        if (event.shiftKey) {
            modelerState.orbit.pan(movement);
        } else {
            modelerState.orbit.turn(movement);
        }
    }

    requestRender();
    event.preventDefault();
}

function onPointerUp(event: PointerEvent) {
    const target = event.currentTarget as HTMLElement;
    if (pointerMode === "paint") {
        modelerState.finishStroke();
    }
    pointerMode = null;
    if (target.hasPointerCapture(event.pointerId)) {
        target.releasePointerCapture(event.pointerId);
    }
    requestRender();
    event.preventDefault();
}

function onPointerLeave() {
    if (pointerMode !== null) return;
    requestRender();
}

function pointerNdc(event: PointerEvent, target: HTMLElement): Vec2 {
    return clampNdcPoint(ndcFromClientPoint(event.clientX, event.clientY, target.getBoundingClientRect()));
}

function draftRenderKey(): string {
    const draft = modelerState.draftStroke;
    if (!draft || draft.length === 0) return "draft-none";
    const last = draft[draft.length - 1];
    return [
        "draft",
        draft.length,
        last.x.toFixed(4),
        last.y.toFixed(4),
        modelerState.brush.color,
        modelerState.brush.width.toFixed(1),
    ].join(":");
}

</script>

<paint-modeler-content>
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
            <label class="color-row">
                <span>Color</span>
                <input
                    type="color"
                    value={modelerState.brush.color}
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
                <input type="checkbox" bind:checked={showChartWireframe} />
                <span>Chart wire</span>
            </label>
            <label class="toggle-row">
                <input type="checkbox" bind:checked={showSurfaceField} />
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

    <paint-viewport
        bind:clientWidth={() => viewportWidth, value => viewportWidth = value}
        bind:clientHeight={() => viewportHeight, value => viewportHeight = value}
        onpointerdown={onPointerDown}
        onpointermove={onPointerMove}
        onpointerup={onPointerUp}
        onpointercancel={onPointerUp}
        onpointerleave={onPointerLeave}
        oncontextmenu={(event: PointerEvent) => event.preventDefault()}
        onwheel={(event: WheelEvent) => {
            modelerState.orbit.zoom(event.deltaY);
            requestRender();
            event.preventDefault();
        }}
        role="application"
    >
        <canvas bind:this={canvas}></canvas>

        <div class="viewport-hud">
            <span>{modelerState.activeObject?.name ?? "No object"}</span>
            <span>{modelerState.currentViewName}</span>
        </div>
    </paint-viewport>
</paint-modeler-content>

<style lang="scss">
paint-modeler-content {
    flex-grow: 1;
    display: flex;
    align-items: stretch;
    min-height: 0;
    overflow: hidden;
}

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

paint-viewport {
    flex-grow: 1;
    min-width: 0;
    display: grid;
    position: relative;
    overflow: hidden;
    background: #090b0c;
    cursor: crosshair;

    > :global(*) {
        grid-area: 1 / 1;
    }
}

canvas {
    width: 100%;
    height: 100%;
    position: relative;
}

canvas {
    display: block;
    z-index: 1;
}

.viewport-hud {
    align-self: start;
    justify-self: end;
    display: flex;
    gap: 0.45rem;
    padding: 0.5rem;
    pointer-events: none;
    z-index: 12;

    span {
        padding: 0.22rem 0.4rem;
        border-radius: 4px;
        background: rgba(0, 0, 0, 0.48);
        color: rgba(255, 255, 255, 0.72);
        font-size: 0.76rem;
    }
}
</style>


