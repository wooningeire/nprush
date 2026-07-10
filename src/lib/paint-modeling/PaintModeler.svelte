<script lang="ts">
import { onDestroy } from "svelte";
import PaintModelerControls from "./PaintModelerControls.svelte";
import { PaintModelingRenderer } from "./PaintModelingRenderer.ts";
import { PaintModelingState } from "./PaintModelingState.svelte.ts";
import { clampNdcPoint, ndcFromClientPoint } from "../contour-modeler/contourGeometry.ts";
import { BrushPlacementMode, BrushPlacementProvenance, type BrushPlacementProvenance as BrushPlacementProvenanceValue, type Vec2 } from "./types.ts";

type BrushPlacementReadback = {
    center: [number, number, number],
    normal: [number, number, number],
    tangent: [number, number, number],
    bitangent: [number, number, number],
    depth: number,
    provenance: BrushPlacementProvenanceValue,
    snapped: boolean,
};

type StrokePlacementDebug = {
    positions: [number, number, number][],
    provenance: BrushPlacementProvenanceValue[],
};

type PaintModelerDebugWindow = Window & typeof globalThis & {
    __paintModelerDebug?: {
        readBrushPlacement: () => Promise<BrushPlacementReadback | null>,
        readLastStrokePlacement: () => StrokePlacementDebug | null,
    },
};

const modelerState = new PaintModelingState();

let canvas = $state<HTMLCanvasElement | null>(null);
let viewportWidth = $state(1);
let viewportHeight = $state(1);
let renderer: PaintModelingRenderer | null = null;
let rendererInitializing = false;
let renderFrameId: number | null = null;
let uploadedStaticSceneKey: string | null = null;
let uploadedDraftKey: string | null = null;
let pointerMode = $state<"paint" | "orbit" | null>(null);
let brushPointerPoint = $state<Vec2 | null>(null);
let rendererError = $state<string | null>(null);
let shadeRibbons = $state(true);
let planePickArmed = $state(false);
let lastStrokePlacementProvenance = $state<BrushPlacementProvenanceValue[]>([]);
let placementPointerVisible = $derived(brushPointerPoint !== null && pointerMode !== "orbit");
let brushGuideVisible = $derived(placementPointerVisible && !planePickArmed);

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
    modelerState.ribbonVersion;
    modelerState.activeViewId;
    modelerState.activeObjectId;
    modelerState.draftStroke;
    modelerState.brush;
    modelerState.activePaintLayerId;
    modelerState.paintLayers.length;
    modelerState.brushPlacementMode;
    modelerState.constructionPlane.origin;
    modelerState.constructionPlane.normal;
    brushPointerPoint;
    pointerMode;
    planePickArmed;
    shadeRibbons;
    requestRender();
});

$effect(() => {
    requestRender();
});


$effect(() => {
    if (typeof window === "undefined") return;

    const debugWindow = window as PaintModelerDebugWindow;
    debugWindow.__paintModelerDebug = {
        readBrushPlacement: async () => {
            await render();
            return renderer?.readBrushPlacementForTest(
                modelerState.camera.viewProjMat,
                modelerState.camera.viewProjInvMat,
                modelerState.camera.viewInvMat,
            ) ?? null;
        },
        readLastStrokePlacement: () => {
            const ribbon = modelerState.strokes.at(-1)?.ribbon;
            if (!ribbon || !renderer) return null;
            return {
                positions: ribbon.vertices.map(vertex => [...vertex.position]),
                provenance: renderer.readLastStrokeProvenanceForTest(),
            };
        },
    };

    return () => {
        delete debugWindow.__paintModelerDebug;
    };
});
onDestroy(() => {
    cancelRender();
    renderer?.destroy();
    renderer = null;
    uploadedStaticSceneKey = null;
    uploadedDraftKey = null;
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
    } catch (error) {
        rendererError = (error as Error)?.message ?? String(error);
    } finally {
        rendererInitializing = false;
    }
}

function requestRender() {
    if (renderFrameId !== null) return;
    renderFrameId = requestAnimationFrame(() => {
        renderFrameId = null;
        void render();
    });
}

function setShadeRibbons(value: boolean) {
    shadeRibbons = value;
    requestRender();
}

function setPlanePickArmed(value: boolean) {
    planePickArmed = value;
    requestRender();
}

async function render() {
    await ensureRenderer();
    if (!renderer) return;

    const staticSceneKey = [
        modelerState.ribbonVersion,
        modelerState.renderDepthSortKey,
        shadeRibbons ? "shade-ribbons" : "flat-ribbons",
    ].join(":");
    if (uploadedStaticSceneKey !== staticSceneKey) {
        renderer.setSegments(modelerState.buildRenderSegments({
            showDraftStroke: false,
            shadeRibbons,
        }));
        uploadedStaticSceneKey = staticSceneKey;
    }

    const draftKey = draftRenderKey();
    if (uploadedDraftKey !== draftKey) {
        renderer.setDraftSegments(modelerState.buildDraftRenderSegments());
        uploadedDraftKey = draftKey;
    }
    const showConstructionPlane = (
        modelerState.brushPlacementMode === BrushPlacementMode.ConstructionPlane
    );
    renderer.setBrushPlacementInput(brushPointerPoint || showConstructionPlane ? {
        point: brushPointerPoint ?? { x: 0, y: 0 },
        brushWidth: modelerState.brush.width,
        viewportWidth,
        viewportHeight,
        pointerVisible: placementPointerVisible,
        planeSize: Math.max(0.35, modelerState.orbit.radius * 0.75),
        startPoint: modelerState.draftStroke?.[0] ?? null,
        placementMode: modelerState.brushPlacementMode,
        constructionPlane: modelerState.constructionPlane,
    } : null);
    renderer.render(
        modelerState.camera.viewProjMat,
        modelerState.camera.viewProjInvMat,
        modelerState.camera.viewMat,
        modelerState.camera.viewInvMat,
    );
}

function cancelRender() {
    if (renderFrameId !== null) {
        cancelAnimationFrame(renderFrameId);
        renderFrameId = null;
    }
}

function onPointerDown(event: PointerEvent) {
    const target = event.currentTarget as HTMLElement;
    const point = pointerNdc(event, target);
    brushPointerPoint = point;

    if (event.button === 0 && planePickArmed) {
        void pickConstructionPlane(point).catch(error => {
            console.warn("Could not pick construction plane", error);
            requestRender();
        });
        event.preventDefault();
        return;
    }

    target.setPointerCapture(event.pointerId);
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
    const target = event.currentTarget as HTMLElement;
    const point = pointerNdc(event, target);
    brushPointerPoint = point;

    if (pointerMode === null) {
        requestRender();
        return;
    }

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
    const shouldFinishPaint = pointerMode === "paint";
    pointerMode = null;
    if (target.hasPointerCapture(event.pointerId)) {
        target.releasePointerCapture(event.pointerId);
    }
    event.preventDefault();

    if (shouldFinishPaint) {
        void finishPaintStroke().finally(requestRender);
    } else {
        requestRender();
    }
}

function onPointerLeave() {
    if (pointerMode !== null) return;
    brushPointerPoint = null;
    requestRender();
}

async function pickConstructionPlane(point: Vec2) {
    await ensureRenderer();
    if (!renderer) return;

    renderer.setBrushPlacementInput({
        point,
        brushWidth: modelerState.brush.width,
        viewportWidth,
        viewportHeight,
        pointerVisible: true,
        planeSize: Math.max(0.35, modelerState.orbit.radius * 0.75),
        startPoint: null,
        placementMode: BrushPlacementMode.Surface,
        constructionPlane: modelerState.constructionPlane,
    });
    const placement = await renderer.readBrushPlacementForTest(
        modelerState.camera.viewProjMat,
        modelerState.camera.viewProjInvMat,
        modelerState.camera.viewInvMat,
    );
    if (placement?.provenance !== BrushPlacementProvenance.Surface) {
        requestRender();
        return;
    }

    modelerState.setConstructionPlane(placement.center, placement.normal);
    planePickArmed = false;
    requestRender();
}

async function finishPaintStroke() {
    if (renderer) {
        const sourcePoints = modelerState.draftStrokeSourcePoints();
        const view = modelerState.strokePlacementView();
        if (sourcePoints && view) {
            try {
                const ribbon = await renderer.buildPlacedRibbonFromSourcePoints({
                    sourcePoints,
                    view,
                    brushWidth: modelerState.brush.width,
                    placementMode: modelerState.brushPlacementMode,
                    constructionPlane: modelerState.constructionPlane,
                });
                if (ribbon) {
                    lastStrokePlacementProvenance = renderer.readLastStrokeProvenanceForTest();
                    if (modelerState.finishStrokeWithRibbon(sourcePoints, ribbon)) return;
                }
            } catch (error) {
                console.warn("GPU brush placement failed; using view-plane stroke", error);
            }
        }
    }

    lastStrokePlacementProvenance = [];
    modelerState.finishStroke();
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
        modelerState.activePaintLayerId,
    ].join(":");
}

</script>

<paint-modeler-content>
    <PaintModelerControls
        {modelerState}
        {rendererError}
        {shadeRibbons}
        {requestRender}
        {setShadeRibbons}
        {planePickArmed}
        {setPlanePickArmed}
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
        class:brush-guide-visible={brushGuideVisible}
        class:plane-pick-armed={planePickArmed}
        data-placement-provenance={lastStrokePlacementProvenance.join(",")}
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

    &.brush-guide-visible {
        cursor: none;
    }

    &.plane-pick-armed {
        cursor: crosshair;
    }

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
@media (max-width: 48rem) {
    paint-modeler-content {
        flex-direction: column;
    }

    paint-viewport {
        min-height: 18rem;
    }
}

</style>
