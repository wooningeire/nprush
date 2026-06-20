<script lang="ts">
import { onDestroy } from "svelte";
import PaintModelerControls from "./PaintModelerControls.svelte";
import { PaintModelingRenderer } from "./PaintModelingRenderer.ts";
import { PaintModelingState } from "./PaintModelingState.svelte.ts";
import { clampNdcPoint, ndcFromClientPoint } from "../contour-modeler/contourGeometry.ts";
import { carryStrokeDepths, snapCarryDepthAtPoint } from "./state/snapDepthCarry.ts";
import type { SnapPlacementPlan } from "./state/strokePlacement.ts";
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
let renderer: PaintModelingRenderer | null = null;
let rendererInitializing = false;
let renderFrameId: number | null = null;
let uploadedStaticSceneKey: string | null = null;
let uploadedDraftKey: string | null = null;
let uploadedChartStateVersion: number | null = null;
let pointerMode = $state<"paint" | "orbit" | null>(null);
let rendererError = $state<string | null>(null);
let finishingStroke = false;
let showChartWireframe = $state(true);
let showSurfaceField = $state(false);


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
    modelerState.brushMode;
    modelerState.placementMode;
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
    uploadedChartStateVersion = null;
});

async function ensureRenderer() {
    if (
        renderer
        || rendererInitializing
        || !canvas
        || canvas.width === 0
        || canvas.height === 0
    ) return;
    rendererInitializing = true;
    rendererError = null;
    try {
        renderer = await PaintModelingRenderer.create(canvas);
        uploadedStaticSceneKey = null;
        uploadedDraftKey = null;
        uploadedChartStateVersion = null;
    } catch (error) {
        rendererError = (error as Error)?.message ?? String(error);
    } finally {
        rendererInitializing = false;
    }
}

function requestRender() {
    if (!active || renderFrameId !== null) return;
    renderFrameId = requestAnimationFrame(() => {
        renderFrameId = null;
        void render();
    });
}

function setShowChartWireframe(value: boolean) {
    showChartWireframe = value;
    requestRender();
}

function setShowSurfaceField(value: boolean) {
    showSurfaceField = value;
    requestRender();
}

async function render() {
    if (!active) return;
    await ensureRenderer();
    if (!renderer) return;

    const staticSceneKey = [
        modelerState.meshVersion,
        showChartWireframe ? "wire" : "no-wire",
        showSurfaceField ? "field" : "no-field",
    ].join(":");
    if (uploadedStaticSceneKey !== staticSceneKey) {
        if (uploadedChartStateVersion !== modelerState.meshVersion) {
            renderer.syncChartState(modelerState.objects);
            uploadedChartStateVersion = modelerState.meshVersion;
        }
        renderer.setChartScene(modelerState.objects, modelerState.views, showChartWireframe, showSurfaceField);
        renderer.setSegments(modelerState.buildRenderSegments({
            showChartWireframe: false,
            showSurfaceField: false,
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
    if (!active || finishingStroke) return;
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

async function onPointerUp(event: PointerEvent) {
    const target = event.currentTarget as HTMLElement;
    const shouldFinishPaint = pointerMode === "paint";
    pointerMode = null;
    if (target.hasPointerCapture(event.pointerId)) {
        target.releasePointerCapture(event.pointerId);
    }
    event.preventDefault();

    if (shouldFinishPaint) {
        await finishPaintStroke();
    }
    requestRender();
}

async function finishPaintStroke() {
    if (finishingStroke) return;
    finishingStroke = true;
    try {
        const snapPlacementPlan = await buildGpuSnapPlacementPlan();
        modelerState.finishStroke(snapPlacementPlan ? { snapPlacementPlan } : undefined);
        applyGpuChartPaintRuns();
    } finally {
        finishingStroke = false;
    }
}

async function buildGpuSnapPlacementPlan(): Promise<SnapPlacementPlan | undefined> {
    if (!renderer || modelerState.brushMode !== "color" || modelerState.placementMode !== "snap") return undefined;

    const object = modelerState.activeObject;
    const view = modelerState.activeView;
    const points = modelerState.draftStrokeSourcePoints();
    if (!object || !view || !points || object.charts.length === 0) return undefined;

    try {
        if (uploadedChartStateVersion !== modelerState.meshVersion) {
            renderer.syncChartState(modelerState.objects);
            uploadedChartStateVersion = modelerState.meshVersion;
        }
        const hits = await renderer.raycastObjectSurfaceBatch(object, modelerState.views, view, points);
        return {
            hits,
            carriedDepths: carryStrokeDepths(
                hits.map(hit => hit ? snapCarryDepthAtPoint(hit.viewDepth) : null),
                points,
            ),
        };
    } catch (error) {
        console.warn("GPU snap placement failed; falling back to CPU raycast", error);
        return undefined;
    }
}

function applyGpuChartPaintRuns() {
    const runs = modelerState.consumeGpuChartPaintRuns();
    if (runs.length === 0) return;
    if (!renderer) {
        uploadedChartStateVersion = null;
        return;
    }
    renderer.applyChartPaintRuns(runs);
    uploadedChartStateVersion = modelerState.meshVersion;
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
        modelerState.brushMode,
        modelerState.placementMode,
        last.x.toFixed(4),
        last.y.toFixed(4),
        modelerState.brush.color,
        modelerState.brush.width.toFixed(1),
    ].join(":");
}

</script>

<paint-modeler-content>
    <PaintModelerControls
        {modelerState}
        {rendererError}
        {showChartWireframe}
        {showSurfaceField}
        {requestRender}
        {setShowChartWireframe}
        {setShowSurfaceField}
    />
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


