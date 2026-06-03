<script lang="ts">
import { onDestroy } from "svelte";
import { PaintModelingRenderer } from "./PaintModelingRenderer.ts";
import { PaintModelingState } from "./PaintModelingState.svelte.ts";
import { clampNdcPoint, ndcFromClientPoint } from "../contour-modeler/contourGeometry.ts";
import type { DepthBrushPreview, DepthTool, PaintTool, PlacementMode, Vec2 } from "./types.ts";

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
let uploadedSceneKey: string | null = null;
let pointerMode = $state<PaintTool | "orbit" | null>(null);
let rendererError = $state<string | null>(null);
let showCharts = $state(true);
let effectLastPoint = $state<Vec2 | null>(null);
let effectCursor = $state<Vec2 | null>(null);
let depthPreviewPath = $state<Vec2[] | null>(null);
let depthPreviewDelta = $state(0);
let depthPullAnchor = $state<Vec2 | null>(null);

const placementModes: PlacementMode[] = ["snap", "new-surface", "occluding-surface", "paint-behind"];
const tools: PaintTool[] = ["paint", "depth-brush", "depth-pull", "seam"];
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
    showCharts;
    modelerState.tool;
    effectCursor;
    depthPreviewPath;
    depthPreviewDelta;
    depthPullAnchor;
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
    uploadedSceneKey = null;
});

function ensureRenderer() {
    if (renderer || !canvas) return;
    rendererError = null;
    try {
        renderer = PaintModelingRenderer.create(canvas);
        uploadedSceneKey = null;
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

    const depthPreview = activeDepthPreview();
    const sceneKey = `${modelerState.meshVersion}:${showCharts ? "charts" : "strokes"}:${depthPreviewKey(depthPreview)}`;
    if (uploadedSceneKey !== sceneKey) {
        renderer.setSegments(modelerState.buildRenderSegments(showCharts, depthPreview ?? undefined));
        uploadedSceneKey = sceneKey;
    }
    renderer.render(modelerState.camera.viewProjMat);
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
        if (modelerState.tool === "paint") {
            modelerState.beginStroke(point, target.clientWidth, target.clientHeight);
            pointerMode = "paint";
        } else if (modelerState.tool === "depth-brush") {
            modelerState.beginUndoGroup();
            modelerState.brushDepthAt(point, event.altKey);
            effectLastPoint = point;
            effectCursor = point;
            depthPreviewPath = [point];
            depthPreviewDelta = modelerState.depthBrushStep(event.altKey);
            depthPullAnchor = null;
            pointerMode = "depth-brush";
        } else if (modelerState.tool === "depth-pull") {
            modelerState.beginUndoGroup();
            effectLastPoint = point;
            effectCursor = point;
            depthPreviewPath = [point];
            depthPreviewDelta = 0;
            depthPullAnchor = point;
            pointerMode = "depth-pull";
        } else {
            modelerState.beginUndoGroup();
            modelerState.markSeamAt(point);
            effectLastPoint = point;
            depthPullAnchor = null;
            pointerMode = "seam";
        }
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

    if (isDepthTool(modelerState.tool)) {
        effectCursor = point;
        if (pointerMode === null) {
            depthPreviewPath = [point];
            depthPreviewDelta = 0;
            depthPullAnchor = null;
            requestRender();
            return;
        }
    } else if (pointerMode === null) {
        return;
    }

    if (pointerMode === "paint") {
        modelerState.appendStrokePoint(point);
    } else if (pointerMode === "depth-brush") {
        const previous = effectLastPoint ?? point;
        const delta = modelerState.depthBrushStep(event.altKey);
        modelerState.brushDepthAlong([previous, point], event.altKey);
        depthPreviewPath = [previous, point];
        depthPreviewDelta = delta;
        effectLastPoint = point;
        depthPullAnchor = null;
    } else if (pointerMode === "depth-pull") {
        const anchor = depthPullAnchor ?? effectLastPoint ?? point;
        const delta = depthPullDeltaFromPointer(event, target);
        if (Math.abs(delta) > 1e-6) {
            modelerState.sculptDepthAlong([anchor], delta);
        }
        depthPreviewPath = [anchor];
        depthPreviewDelta += delta;
        effectLastPoint = point;
    } else if (pointerMode === "seam") {
        modelerState.markSeamAlong(effectLastPoint ? [effectLastPoint, point] : [point]);
        effectLastPoint = point;
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
    const completedMode = pointerMode;
    const target = event.currentTarget as HTMLElement;
    if (pointerMode === "paint") {
        modelerState.finishStroke();
    } else if (isDepthTool(pointerMode) || pointerMode === "seam") {
        modelerState.commitUndoGroup();
    }
    pointerMode = null;
    effectLastPoint = null;
    depthPullAnchor = null;
    if (isDepthTool(completedMode)) {
        const point = pointerNdc(event, target);
        effectCursor = point;
        depthPreviewPath = [point];
        depthPreviewDelta = 0;
    }
    if (target.hasPointerCapture(event.pointerId)) {
        target.releasePointerCapture(event.pointerId);
    }
    requestRender();
    event.preventDefault();
}

function onPointerLeave() {
    if (pointerMode !== null) return;
    effectCursor = null;
    depthPreviewPath = null;
    depthPreviewDelta = 0;
    depthPullAnchor = null;
    requestRender();
}

function pointerNdc(event: PointerEvent, target: HTMLElement): Vec2 {
    return clampNdcPoint(ndcFromClientPoint(event.clientX, event.clientY, target.getBoundingClientRect()));
}

function screenPoint(point: Vec2): Vec2 {
    return {
        x: (point.x * 0.5 + 0.5) * viewportWidth,
        y: (-point.y * 0.5 + 0.5) * viewportHeight,
    };
}

function screenStrokePoints(points: Vec2[]): string {
    return points.map(point => {
        const screen = screenPoint(point);
        return `${screen.x},${screen.y}`;
    }).join(" ");
}

function depthBrushScreenRadius(): number {
    return modelerState.depthBrushRadius * Math.min(viewportWidth, viewportHeight) * 0.5;
}

function activeDepthPreview(): DepthBrushPreview | null {
    if (!isDepthTool(modelerState.tool)) return null;
    const points = depthPreviewPath ?? (effectCursor ? [effectCursor] : null);
    if (!points || points.length === 0) return null;
    return {
        tool: modelerState.tool,
        points,
        delta: depthPreviewDelta,
    };
}

function depthPreviewKey(preview: DepthBrushPreview | null): string {
    if (!preview) return "none";
    return `${preview.tool}:${preview.delta.toFixed(5)}:${preview.points.map(point => `${point.x.toFixed(4)},${point.y.toFixed(4)}`).join("|")}`;
}

function depthPullDeltaFromPointer(event: PointerEvent, target: HTMLElement): number {
    const minDimension = Math.max(1, Math.min(target.clientWidth, target.clientHeight));
    const direction = event.altKey ? -1 : 1;
    return clamp(
        event.movementY / minDimension * modelerState.depthBrushStrength * 12 * direction,
        -modelerState.depthBrushStrength,
        modelerState.depthBrushStrength,
    );
}

function isDepthTool(tool: PaintTool | "orbit" | null): tool is DepthTool {
    return tool === "depth-brush" || tool === "depth-pull";
}

function placementLabel(mode: PlacementMode): string {
    if (mode === "snap") return "Snap";
    if (mode === "new-surface") return "New";
    if (mode === "occluding-surface") return "Occlude";
    return "Behind";
}

function toolLabel(tool: PaintTool): string {
    if (tool === "paint") return "Paint";
    if (tool === "depth-brush") return "Depth Brush";
    if (tool === "depth-pull") return "Depth Pull";
    return "Seam";
}

function clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value));
}
</script>

<paint-modeler-content>
    <aside class="control-panel">
        <section>
            <div class="section-title">Paint Modeler</div>
            <div class="subtle">Ray-depth prototype</div>
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
            <div class="section-title">Tool</div>
            <div class="segmented tools">
                {#each tools as tool}
                    <button
                        class:active={modelerState.tool === tool}
                        onclick={() => modelerState.setTool(tool)}
                    >
                        {toolLabel(tool)}
                    </button>
                {/each}
            </div>
        </section>

        <div class="separator"></div>

        <section>
            <div class="section-title">Placement</div>
            <div class="segmented two">
                {#each placementModes as mode}
                    <button
                        class:active={modelerState.placementMode === mode}
                        onclick={() => modelerState.setPlacementMode(mode)}
                    >
                        {placementLabel(mode)}
                    </button>
                {/each}
            </div>
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
                <span>Paint</span>
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
            <label class="range-row">
                <span>Opacity</span>
                <input
                    type="range"
                    min="0.05"
                    max="1"
                    step="0.01"
                    value={modelerState.brush.opacity}
                    oninput={(event) => modelerState.setBrushOpacity(Number((event.currentTarget as HTMLInputElement).value))}
                />
                <small>{Math.round(modelerState.brush.opacity * 100)}%</small>
            </label>
            <label class="range-row">
                <span>Depth rate</span>
                <input
                    type="range"
                    min="0.01"
                    max="0.2"
                    step="0.005"
                    value={modelerState.depthBrushStrength}
                    oninput={(event) => modelerState.setDepthBrushStrength(Number((event.currentTarget as HTMLInputElement).value))}
                />
                <small>{modelerState.depthBrushStrength.toFixed(2)}</small>
            </label>
            <label class="range-row">
                <span>Depth radius</span>
                <input
                    type="range"
                    min="0.04"
                    max="0.85"
                    step="0.01"
                    value={modelerState.depthBrushRadius}
                    oninput={(event) => modelerState.setDepthBrushRadius(Number((event.currentTarget as HTMLInputElement).value))}
                />
                <small>{modelerState.depthBrushRadius.toFixed(2)}</small>
            </label>
            <label class="range-row">
                <span>Seam size</span>
                <input
                    type="range"
                    min="0.015"
                    max="0.22"
                    step="0.005"
                    value={modelerState.seamBrushRadius}
                    oninput={(event) => modelerState.setSeamBrushRadius(Number((event.currentTarget as HTMLInputElement).value))}
                />
                <small>{modelerState.seamBrushRadius.toFixed(2)}</small>
            </label>
            <label class="toggle-row">
                <input type="checkbox" bind:checked={showCharts} />
                <span>Charts</span>
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
            <span>Seams {modelerState.seamCount}</span>
        </section>

        {#if rendererError}
            <div class="error">{rendererError}</div>
        {/if}
    </aside>

    <paint-viewport
        bind:clientWidth={() => viewportWidth, value => viewportWidth = value}
        bind:clientHeight={() => viewportHeight, value => viewportHeight = value}
        class:depth-brush-active={modelerState.tool === "depth-brush"}
        class:depth-pull-active={modelerState.tool === "depth-pull"}
        class:seam-active={modelerState.tool === "seam"}
        style={`--draft-color: ${modelerState.brush.color}; --draft-width: ${modelerState.brush.width}px; --draft-opacity: ${modelerState.brush.opacity};`}
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
        <svg
            class="drawing-overlay"
            viewBox={`0 0 ${viewportWidth} ${viewportHeight}`}
            preserveAspectRatio="none"
        >
            {#if modelerState.draftStroke && modelerState.draftStroke.length > 1}
                <polyline
                    class="draft"
                    points={screenStrokePoints(modelerState.draftStroke)}
                />
            {/if}
            {#if modelerState.tool === "depth-pull" && depthPullAnchor && effectCursor}
                <line
                    class="depth-pull-drag"
                    x1={screenPoint(depthPullAnchor).x}
                    y1={screenPoint(depthPullAnchor).y}
                    x2={screenPoint(effectCursor).x}
                    y2={screenPoint(effectCursor).y}
                />
                <circle
                    class="depth-pull-anchor"
                    cx={screenPoint(depthPullAnchor).x}
                    cy={screenPoint(depthPullAnchor).y}
                    r="5"
                />
            {/if}
            {#if isDepthTool(modelerState.tool) && effectCursor}
                <circle
                    class="depth-brush-cursor"
                    class:brush={modelerState.tool === "depth-brush"}
                    class:pull={modelerState.tool === "depth-pull"}
                    class:dragging={isDepthTool(pointerMode)}
                    cx={screenPoint(effectCursor).x}
                    cy={screenPoint(effectCursor).y}
                    r={depthBrushScreenRadius()}
                />
                <circle
                    class="depth-brush-dot"
                    cx={screenPoint(effectCursor).x}
                    cy={screenPoint(effectCursor).y}
                    r="3.5"
                />
            {/if}
        </svg>

        <div class="viewport-hud">
            <span>{modelerState.activeObject?.name ?? "No object"}</span>
            <span>{modelerState.currentViewName}</span>
            <span>{toolLabel(modelerState.tool)}</span>
            <span>{placementLabel(modelerState.placementMode)}</span>
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

.segmented {
    display: grid;
    gap: 0.32rem;

    &.two {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    &.tools {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    button {
        font-size: 0.76rem;
        padding: 0 0.25rem;

        &.active {
            background: rgba(76, 154, 131, 0.34);
            border-color: rgba(125, 221, 189, 0.48);
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

    &.depth-brush-active {
        cursor: crosshair;
    }

    &.depth-pull-active {
        cursor: ns-resize;
    }

    &.seam-active {
        cursor: cell;
    }

    > :global(*) {
        grid-area: 1 / 1;
    }
}

canvas,
.drawing-overlay {
    width: 100%;
    height: 100%;
    position: relative;
}

canvas {
    display: block;
    z-index: 1;
}

.drawing-overlay {
    z-index: 10;
    pointer-events: none;
    overflow: visible;
}

.draft {
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
    vector-effect: non-scaling-stroke;
}

.draft {
    stroke: var(--draft-color, #ffd27a);
    stroke-width: var(--draft-width, 4px);
    opacity: var(--draft-opacity, 0.9);
}

.depth-brush-cursor {
    stroke-width: 2px;
    stroke-dasharray: 7 5;
    vector-effect: non-scaling-stroke;

    &.brush {
        fill: rgba(68, 163, 255, 0.1);
        stroke: rgba(128, 210, 255, 0.96);
    }

    &.pull {
        fill: rgba(255, 202, 65, 0.11);
        stroke: rgba(255, 228, 112, 0.98);
    }

    &.dragging {
        stroke-dasharray: none;
        stroke-width: 2.6px;
    }
}

.depth-brush-dot {
    fill: rgba(255, 255, 255, 0.95);
    stroke: rgba(0, 0, 0, 0.7);
    stroke-width: 1.5px;
    vector-effect: non-scaling-stroke;
}

.depth-pull-drag {
    stroke: rgba(255, 218, 92, 0.9);
    stroke-width: 2px;
    stroke-linecap: round;
    vector-effect: non-scaling-stroke;
}

.depth-pull-anchor {
    fill: rgba(255, 224, 118, 0.98);
    stroke: rgba(0, 0, 0, 0.72);
    stroke-width: 1.6px;
    vector-effect: non-scaling-stroke;
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


