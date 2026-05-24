<script lang="ts">
import { onDestroy } from "svelte";
import { ContourModelerState } from "./ContourModelerState.svelte.ts";
import { ndcFromClientPoint, clampNdcPoint } from "./contourGeometry.ts";
import { ContourMeshRenderer } from "./ContourMeshRenderer.ts";
import { buildCrossViewGuides, type CrossViewGuide } from "./viewGuides.ts";
import type { ContourStrokeKind, Vec2 } from "./types.ts";

let {
    active,
}: {
    active: boolean,
} = $props();

const modelerState = new ContourModelerState();

let canvas = $state<HTMLCanvasElement | null>(null);
let viewportWidth = $state(1);
let viewportHeight = $state(1);
let renderer: ContourMeshRenderer | null = null;
let rendererInitializing = false;
let renderFrameId: number | null = null;
let pointerMode = $state<"draw" | "orbit" | "depth-brush" | null>(null);
let rendererError = $state<string | null>(null);
let showCrossViewGuides = $state(true);
let depthBrushEnabled = $state(false);
let depthBrushRadius = $state(0.095);
let depthBrushStrength = $state(1);
let depthBrushCursor = $state<Vec2 | null>(null);
let depthBrushLastPoint = $state<Vec2 | null>(null);
let depthBrushSessionGuides = $state<CrossViewGuide[] | null>(null);

let crossViewGuides = $derived.by(() => {
    modelerState.meshVersion;
    if (!showCrossViewGuides) return [];
    return buildCrossViewGuides({
        strokes: modelerState.guideStrokes,
        views: modelerState.views,
        currentViewProjMat: modelerState.camera.viewProjMat,
        shapeParams: modelerState.activeShape?.mesh ? modelerState.activeShape.params : null,
    });
});

$effect(() => {
    if (!canvas) return;
    const dpr = typeof window === "undefined" ? 1 : window.devicePixelRatio;
    canvas.width = Math.max(1, Math.floor(viewportWidth * dpr));
    canvas.height = Math.max(1, Math.floor(viewportHeight * dpr));
    modelerState.viewportWidth = viewportWidth;
    modelerState.viewportHeight = viewportHeight;
    requestViewportRender();
});

$effect(() => {
    modelerState.meshVersion;
    modelerState.activeShapeId;
    if (!hasRenderableMesh()) {
        destroyRenderer();
        return;
    }
    renderer?.setShapes(modelerState.shapes, modelerState.activeShapeId);
    requestViewportRender();
});

$effect(() => {
    if (active && hasRenderableMesh() && !isDrawing()) {
        requestViewportRender();
    } else {
        cancelScheduledRender();
    }
});

onDestroy(() => {
    cancelScheduledRender();
    destroyRenderer();
});

async function ensureRenderer() {
    if (
        renderer
        || rendererInitializing
        || !hasRenderableMesh()
        || isDrawing()
        || !canvas
        || canvas.width === 0
        || canvas.height === 0
    ) return;
    rendererInitializing = true;
    rendererError = null;
    try {
        renderer = await ContourMeshRenderer.create(canvas);
        renderer.setShapes(modelerState.shapes, modelerState.activeShapeId);
    } catch (e) {
        rendererError = (e as Error)?.message ?? String(e);
    } finally {
        rendererInitializing = false;
    }
}

function requestViewportRender() {
    if (!active || isDrawing() || !hasRenderableMesh() || renderFrameId !== null) return;
    renderFrameId = requestAnimationFrame(() => {
        renderFrameId = null;
        void renderViewport();
    });
}

async function renderViewport() {
    if (!active || isDrawing() || !hasRenderableMesh()) return;
    await ensureRenderer();
    if (!active || isDrawing() || !hasRenderableMesh()) return;
    renderer?.render(modelerState.camera.viewProjMat);
}

function cancelScheduledRender() {
    if (renderFrameId !== null) {
        cancelAnimationFrame(renderFrameId);
        renderFrameId = null;
    }
}

function destroyRenderer() {
    renderer?.destroy();
    renderer = null;
    rendererInitializing = false;
}

function hasRenderableMesh(): boolean {
    return modelerState.shapes.some(shape => shape.mesh !== null);
}

function isDrawing(): boolean {
    return pointerMode === "draw";
}

function onPointerDown(event: PointerEvent) {
    const target = event.currentTarget as HTMLElement;
    target.setPointerCapture(event.pointerId);

    if (event.button === 0) {
        cancelScheduledRender();
        if (canDepthBrush()) {
            pointerMode = "depth-brush";
            depthBrushLastPoint = pointerScreen(event, target);
            depthBrushSessionGuides = crossViewGuides;
            updateDepthBrushCursor(event, target);
        } else {
            const point = pointerNdc(event, target);
            modelerState.beginStroke(point, target.clientWidth, target.clientHeight);
            pointerMode = "draw";
        }
    } else if (event.button === 1) {
        pointerMode = "orbit";
    }

    event.preventDefault();
}

function onPointerMove(event: PointerEvent) {
    updateDepthBrushCursor(event, event.currentTarget as HTMLElement);
    if (pointerMode === "draw") {
        modelerState.appendStrokePoint(pointerNdc(event, event.currentTarget as HTMLElement));
    } else if (pointerMode === "depth-brush") {
        applyDepthBrush(event, event.currentTarget as HTMLElement);
    } else if (pointerMode === "orbit") {
        const movement = { x: event.movementX, y: event.movementY };
        if (event.shiftKey) {
            modelerState.orbit.pan(movement);
        } else {
            modelerState.orbit.turn(movement);
        }
        requestViewportRender();
    }
    event.preventDefault();
}

function onPointerUp(event: PointerEvent) {
    if (pointerMode === "draw") {
        modelerState.finishStroke();
    }
    depthBrushLastPoint = null;
    depthBrushSessionGuides = null;
    pointerMode = null;
    requestViewportRender();
    const target = event.currentTarget as HTMLElement;
    if (target.hasPointerCapture(event.pointerId)) {
        target.releasePointerCapture(event.pointerId);
    }
    event.preventDefault();
}

function onPointerLeave() {
    if (pointerMode !== "depth-brush") {
        depthBrushCursor = null;
    }
}

function canDepthBrush(): boolean {
    return depthBrushEnabled
        && showCrossViewGuides
        && modelerState.depthEditableStrokes.length > 0
        && crossViewGuides.some(guide => guide.style === "proxy" && (guide.vertices?.length ?? 0) > 0);
}

function updateDepthBrushCursor(event: PointerEvent, target: HTMLElement) {
    depthBrushCursor = canDepthBrush() ? pointerNdc(event, target) : null;
}

function applyDepthBrush(event: PointerEvent, target: HTMLElement) {
    const currentScreen = pointerScreen(event, target);
    if (depthBrushLastPoint === null) {
        depthBrushLastPoint = currentScreen;
        return;
    }

    const movement = clampScreenMovement({
        x: currentScreen.x - depthBrushLastPoint.x,
        y: currentScreen.y - depthBrushLastPoint.y,
    });
    depthBrushLastPoint = currentScreen;
    if (Math.hypot(movement.x, movement.y) <= 1e-6) return;

    const center = pointerNdc(event, target);
    const edits = depthBrushEdits(center, screenBrushRadiusPx(), movement, depthBrushSessionGuides ?? crossViewGuides);
    modelerState.brushStrokeDepth(edits);
}

function depthBrushEdits(center: Vec2, radiusPx: number, movement: Vec2, guides: CrossViewGuide[]) {
    const edits: Array<{ strokeId: string; pointIndex: number; influence: number; delta: number }> = [];
    const centerScreen = screenPoint(center);
    const r = Math.max(6, radiusPx);
    for (const guide of guides) {
        if (guide.style !== "proxy" || !guide.vertices) continue;
        for (const vertex of guide.vertices) {
            const vertexScreen = screenPoint(vertex.point);
            const d = Math.hypot(vertexScreen.x - centerScreen.x, vertexScreen.y - centerScreen.y);
            if (d > r) continue;
            const depthDirection = screenDepthDirection(vertex.depthDirection);
            if (!depthDirection) continue;
            const projectedMovement = movement.x * depthDirection.x + movement.y * depthDirection.y;
            if (Math.abs(projectedMovement) <= 1e-5) continue;
            const normalized = d / r;
            edits.push({
                strokeId: guide.strokeId,
                pointIndex: vertex.strokePointIndex,
                influence: (1 - normalized * normalized) ** 2,
                delta: projectedMovement * 0.006 * Number(depthBrushStrength),
            });
        }
    }
    return edits;
}

function pointerNdc(event: PointerEvent, target: HTMLElement): Vec2 {
    return clampNdcPoint(ndcFromClientPoint(event.clientX, event.clientY, target.getBoundingClientRect()));
}

function pointerScreen(event: PointerEvent, target: HTMLElement): Vec2 {
    const rect = target.getBoundingClientRect();
    return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
    };
}

function clampScreenMovement(movement: Vec2): Vec2 {
    const length = Math.hypot(movement.x, movement.y);
    if (!Number.isFinite(length) || length <= 12) return movement;
    const scale = 12 / length;
    return {
        x: movement.x * scale,
        y: movement.y * scale,
    };
}

function setKind(kind: ContourStrokeKind) {
    modelerState.activeKind = kind;
}

function kindLabel(kind: ContourStrokeKind): string {
    return kind === "edge" ? "Edge" : "Contour";
}

function screenPoint(point: Vec2): Vec2 {
    return {
        x: (point.x * 0.5 + 0.5) * viewportWidth,
        y: (-point.y * 0.5 + 0.5) * viewportHeight,
    };
}

function screenBrushRadiusPx(): number {
    return Number(depthBrushRadius) * Math.min(viewportWidth, viewportHeight) * 0.5;
}

function screenDepthDirection(direction: Vec2 | undefined): Vec2 | null {
    if (!direction) return null;
    const x = direction.x * viewportWidth;
    const y = -direction.y * viewportHeight;
    const length = Math.hypot(x, y);
    if (!Number.isFinite(length) || length <= 1e-6) return null;
    return {
        x: x / length,
        y: y / length,
    };
}

function screenStrokePoints(points: Vec2[]): string {
    return points
        .map(point => {
            const screen = screenPoint(point);
            return `${screen.x},${screen.y}`;
        })
        .join(" ");
}
</script>

<contour-modeler-content>
    <aside class="control-panel">
        <section>
            <div class="section-title">Contour Modeler</div>
            <div class="subtle">Implicit body scene</div>
        </section>

        <div class="separator"></div>

        <section class="button-row">
            <button onclick={() => modelerState.addShape()}>Add Shape</button>
            <button onclick={() => modelerState.undoStroke()} disabled={!modelerState.activeShape}>Undo</button>
        </section>

        <section class="button-row">
            <button
                class="primary"
                onclick={() => modelerState.fitActiveShape(null)}
                disabled={!modelerState.activeShape || modelerState.isFitting || modelerState.activeShape.strokeIds.length === 0}
            >
                Fit
            </button>
            {#if modelerState.isFitting}
                <button onclick={() => modelerState.cancelFit()}>Cancel</button>
            {:else}
                <button onclick={() => modelerState.clearActiveShape()} disabled={!modelerState.activeShape}>Clear Shape</button>
            {/if}
        </section>

        {#if modelerState.activeShape?.fitStatus === "fitting"}
            <div class="progress">
                <span style:width="{modelerState.fitProgress * 100}%"></span>
            </div>
        {/if}

        <div class="separator"></div>

        <section>
            <div class="section-title">Line Type</div>
            <div class="segmented">
                {#each ["edge", "contour"] as kind}
                    <button
                        class:active={modelerState.activeKind === kind}
                        onclick={() => setKind(kind as ContourStrokeKind)}
                    >
                        {kindLabel(kind as ContourStrokeKind)}
                    </button>
                {/each}
            </div>
            <label class="toggle-row">
                <input type="checkbox" bind:checked={showCrossViewGuides} />
                <span>Cross-view guides</span>
            </label>
        </section>

        {#if modelerState.depthEditableStrokes.length > 0}
            <div class="separator"></div>

            <section>
                <div class="section-title">Depth Brush</div>
                <label class="toggle-row">
                    <input type="checkbox" bind:checked={depthBrushEnabled} />
                    <span>Brush</span>
                </label>
                <label class="range-row">
                    <span>Size</span>
                    <input type="range" min="0.04" max="0.2" step="0.005" bind:value={depthBrushRadius} />
                    <small>{Math.round(Number(depthBrushRadius) * 100)}</small>
                </label>
                <label class="range-row">
                    <span>Flow</span>
                    <input type="range" min="0.25" max="4" step="0.25" bind:value={depthBrushStrength} />
                    <small>{Number(depthBrushStrength).toFixed(2)}</small>
                </label>
                <button onclick={() => modelerState.resetGuideDepths()}>Reset Depths</button>
            </section>
        {/if}

        <div class="separator"></div>

        <section>
            <div class="section-title">Shapes</div>
            {#if modelerState.shapes.length === 0}
                <div class="subtle">Add a shape, then draw contours on the viewport.</div>
            {:else}
                <div class="list">
                    {#each modelerState.shapes as shape}
                        <button
                            class:active={shape.id === modelerState.activeShapeId}
                            onclick={() => {
                                modelerState.selectShape(shape.id);
                                requestViewportRender();
                            }}
                        >
                            <span>{shape.name}</span>
                            <small>{shape.fitStatus}{shape.fitLoss !== null ? ` ${shape.fitLoss.toFixed(4)}` : ""}</small>
                        </button>
                    {/each}
                </div>
            {/if}
        </section>

        <div class="separator"></div>

        <section>
            <div class="section-title">Saved Views</div>
            {#if modelerState.views.length === 0}
                <div class="subtle">A view is saved when you start drawing.</div>
            {:else}
                <div class="list">
                    {#each modelerState.views as view}
                        <button
                            class:active={view.id === modelerState.activeViewId && modelerState.isCameraAtActiveView}
                            onclick={() => {
                                modelerState.selectView(view.id);
                                requestViewportRender();
                            }}
                        >
                            <span>{view.name}</span>
                            <small>{modelerState.strokes.filter(stroke => stroke.viewId === view.id).length} lines</small>
                        </button>
                    {/each}
                </div>
            {/if}
        </section>

        {#if rendererError}
            <div class="error">{rendererError}</div>
        {/if}
    </aside>

    <contour-viewport
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
            requestViewportRender();
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
            {#each crossViewGuides as guide}
                {#if guide.points.length > 1}
                    <polyline
                        class={`guide guide-${guide.style} kind-${guide.kind} ${guide.strokeId === modelerState.activeDepthStrokeId ? "selected" : ""}`}
                        points={screenStrokePoints(guide.points)}
                    />
                {/if}
            {/each}
            {#each modelerState.visibleStrokes as stroke}
                {#if stroke.points.length > 1}
                    <polyline class="stroke-shadow" points={screenStrokePoints(stroke.points)} />
                    <polyline class={`stroke kind-${stroke.kind}`} points={screenStrokePoints(stroke.points)} />
                {/if}
            {/each}
            {#if modelerState.draftStroke && modelerState.draftStroke.length > 1}
                <polyline class="stroke-shadow draft" points={screenStrokePoints(modelerState.draftStroke)} />
                <polyline
                    class={`stroke draft kind-${modelerState.activeKind}`}
                    points={screenStrokePoints(modelerState.draftStroke)}
                />
            {/if}
            {#if depthBrushEnabled && depthBrushCursor}
                <circle
                    class="brush-cursor"
                    cx={screenPoint(depthBrushCursor).x}
                    cy={screenPoint(depthBrushCursor).y}
                    r={screenBrushRadiusPx()}
                />
            {/if}
        </svg>

        <div class="viewport-hud">
            <span>{modelerState.activeShape?.name ?? "No shape"}</span>
            <span>{modelerState.currentViewName}</span>
            <span>{kindLabel(modelerState.activeKind)}</span>
        </div>
    </contour-viewport>
</contour-modeler-content>

<style lang="scss">
contour-modeler-content {
    flex-grow: 1;
    display: flex;
    align-items: stretch;
    min-height: 0;
    overflow: hidden;
}

.control-panel {
    width: 17rem;
    flex: 0 0 17rem;
    overflow-y: auto;
    padding: 1rem;
    border-right: 1px solid rgba(255, 255, 255, 0.12);
    background: rgba(12, 15, 16, 0.72);

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
small {
    color: rgba(255, 255, 255, 0.48);
    font-size: 0.78rem;
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

    &.primary {
        background: linear-gradient(135deg, #2f7d68, #77a84f);
        border-color: rgba(255, 255, 255, 0.2);
    }
}

.button-row {
    display: grid !important;
    grid-template-columns: 1fr 1fr;
    gap: 0.45rem;
}

.segmented {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.3rem;

    button {
        font-size: 0.78rem;
        padding: 0 0.35rem;

        &.active {
            background: rgba(119, 168, 79, 0.32);
            border-color: rgba(166, 218, 114, 0.5);
        }
    }
}

.toggle-row {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    min-height: 1.6rem;
    color: rgba(255, 255, 255, 0.66);
    font-size: 0.78rem;

    input {
        accent-color: #77a84f;
    }
}

.list {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;

    button {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem;
        padding: 0.35rem 0.5rem;
        text-align: left;

        &.active {
            background: rgba(93, 139, 179, 0.28);
            border-color: rgba(128, 179, 221, 0.48);
        }
    }
}

.range-row {
    display: grid;
    grid-template-columns: 2.7rem minmax(0, 1fr) 2rem;
    align-items: center;
    gap: 0.5rem;
    color: rgba(255, 255, 255, 0.66);
    font-size: 0.78rem;

    input {
        width: 100%;
        accent-color: #77a84f;
    }
}

.progress {
    height: 4px;
    overflow: hidden;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.1);
    margin-top: 0.65rem;

    span {
        display: block;
        height: 100%;
        background: #a6da72;
        transition: width 0.12s ease;
    }
}

.error {
    margin-top: 0.85rem;
    color: #ffb4a8;
    font-size: 0.82rem;
}

contour-viewport {
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
    pointer-events: none;
    z-index: 10;
    overflow: visible;
}

.stroke-shadow,
.stroke,
.guide {
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
}

.stroke-shadow {
    stroke: rgba(0, 0, 0, 0.86);
    stroke-width: 8px;
}

.stroke {
    stroke-width: 4px;
}

.guide {
    stroke-width: 2px;
    stroke-dasharray: 7 7;
    opacity: 0.34;
    pointer-events: none;
}

.guide.selected {
    opacity: 0.85;
    stroke-width: 4px;
}

.brush-cursor {
    fill: rgba(166, 218, 114, 0.08);
    stroke: rgba(166, 218, 114, 0.72);
    stroke-width: 1.5px;
    stroke-dasharray: 5 5;
    pointer-events: none;
}

.guide-ray {
    stroke-width: 1px;
    stroke-dasharray: 2 12;
    opacity: 0.08;
}

.guide-surface {
    stroke-width: 3px;
    stroke-dasharray: 10 5;
    opacity: 0.5;
}

.kind-edge {
    stroke: #fff2b8;
}

.kind-contour {
    stroke: #5ecbff;
}

.draft {
    opacity: 0.82;
}

.viewport-hud {
    align-self: start;
    justify-self: end;
    display: flex;
    gap: 0.45rem;
    padding: 0.5rem;
    pointer-events: none;
    z-index: 4;

    span {
        padding: 0.22rem 0.4rem;
        border-radius: 4px;
        background: rgba(0, 0, 0, 0.48);
        color: rgba(255, 255, 255, 0.72);
        font-size: 0.76rem;
    }
}
</style>
