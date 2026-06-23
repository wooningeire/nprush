import type { ChartPaintRun } from "./chartPainting.ts";
import { makeId } from "./sceneData.ts";
import { samplePaintStrokeSpline } from "./strokeSampling.ts";
import {
    placeDepthBrushSculpt,
    placeStrokeSamples,
    placeSurfaceBrushMask,
    type SnapPlacementPlan,
    type StrokePlacementContext,
} from "./strokePlacement.ts";
import type { PaintSceneSnapshot } from "./sceneHistory.ts";
import type {
    BrushMode,
    BrushStyle,
    OcclusionClaim,
    PaintObject,
    PaintStroke,
    PaintView,
    PlacementMode,
    Vec2,
} from "../types.ts";

export type FinishStrokeInput = {
    draftStroke: Vec2[] | null,
    pendingStrokeUndoSnapshot: PaintSceneSnapshot | null,
    undoSnapshot: PaintSceneSnapshot,
    object: PaintObject | null,
    view: PaintView | null,
    brushMode: BrushMode,
    placementMode: PlacementMode,
    brush: BrushStyle,
    placementContext: StrokePlacementContext,
    nextPaintOrder: (objectId: string) => number,
    paintLayerId: string,
    snapPlacementPlan?: SnapPlacementPlan,
};

export type FinishStrokeResult =
    | {
        kind: "discard",
        restoreSnapshot?: PaintSceneSnapshot,
    }
    | {
        kind: "surface",
        undoSnapshot: PaintSceneSnapshot,
        touchedChartIds: Set<string>,
        gpuChartPaintRuns: ChartPaintRun[],
    }
    | {
        kind: "depth",
        undoSnapshot: PaintSceneSnapshot,
        touchedChartIds: Set<string>,
        gpuChartPaintRuns: ChartPaintRun[],
    }
    | {
        kind: "stroke",
        undoSnapshot: PaintSceneSnapshot,
        stroke: PaintStroke,
        occlusionClaim?: OcclusionClaim,
        touchedChartIds: Set<string>,
        gpuChartPaintRuns: ChartPaintRun[],
    };

export const planFinishedStroke = ({
    draftStroke,
    pendingStrokeUndoSnapshot,
    undoSnapshot,
    object,
    view,
    brushMode,
    placementMode,
    brush,
    placementContext,
    nextPaintOrder,
    paintLayerId,
    snapPlacementPlan,
}: FinishStrokeInput): FinishStrokeResult => {
    if (!draftStroke || draftStroke.length < 2 || !object || !view) {
        return {
            kind: "discard",
            restoreSnapshot: pendingStrokeUndoSnapshot ?? undefined,
        };
    }

    const sourcePoints = samplePaintStrokeSpline(draftStroke);
    if (brushMode === "surface") {
        const surfacePlacement = placeSurfaceBrushMask(
            placementContext,
            object,
            view,
            sourcePoints,
        );
        if (surfacePlacement.touchedChartIds.size === 0) {
            return {
                kind: "discard",
                restoreSnapshot: undoSnapshot,
            };
        }

        return {
            kind: "surface",
            undoSnapshot,
            touchedChartIds: surfacePlacement.touchedChartIds,
            gpuChartPaintRuns: surfacePlacement.gpuChartPaintRuns,
        };
    }

    if (brushMode === "depth") {
        const depthPlacement = placeDepthBrushSculpt(
            placementContext,
            object,
            view,
            sourcePoints,
        );
        if (depthPlacement.touchedChartIds.size === 0) {
            return {
                kind: "discard",
                restoreSnapshot: undoSnapshot,
            };
        }

        return {
            kind: "depth",
            undoSnapshot,
            touchedChartIds: depthPlacement.touchedChartIds,
            gpuChartPaintRuns: depthPlacement.gpuChartPaintRuns,
        };
    }

    const strokeSamples = placeStrokeSamples(
        placementContext,
        object,
        view,
        sourcePoints,
        placementMode,
        snapPlacementPlan,
    );
    if (strokeSamples.samples.length < 2) {
        return {
            kind: "discard",
            restoreSnapshot: undoSnapshot,
        };
    }

    return {
        kind: "stroke",
        undoSnapshot,
        stroke: {
            id: makeId("stroke"),
            objectId: object.id,
            layerId: paintLayerId,
            sourceViewId: view.id,
            placement: placementMode,
            samples: strokeSamples.samples,
            style: { ...brush },
            paintOrder: nextPaintOrder(object.id),
        },
        occlusionClaim: strokeSamples.occlusionClaim,
        touchedChartIds: strokeSamples.touchedChartIds,
        gpuChartPaintRuns: strokeSamples.gpuChartPaintRuns,
    };
};
