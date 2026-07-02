import { makeId } from "./sceneData.ts";
import { samplePaintStrokeSpline } from "./strokeSampling.ts";
import {
    buildRibbonStrokeGeometry,
    ribbonSegmentCount,
} from "./strokeMesh.ts";
import type { PaintSceneSnapshot } from "./sceneHistory.ts";
import type {
    BrushStyle,
    PaintObject,
    PaintStroke,
    PaintView,
    Vec2,
} from "../types.ts";

export type FinishStrokeInput = {
    draftStroke: Vec2[] | null,
    pendingStrokeUndoSnapshot: PaintSceneSnapshot | null,
    undoSnapshot: PaintSceneSnapshot,
    object: PaintObject | null,
    view: PaintView | null,
    brush: BrushStyle,
    nextPaintOrder: (objectId: string) => number,
    paintLayerId: string,
};

export type FinishStrokeResult =
    | {
        kind: "discard",
        restoreSnapshot?: PaintSceneSnapshot,
    }
    | {
        kind: "stroke",
        undoSnapshot: PaintSceneSnapshot,
        stroke: PaintStroke,
    };

export const planFinishedStroke = ({
    draftStroke,
    pendingStrokeUndoSnapshot,
    undoSnapshot,
    object,
    view,
    brush,
    nextPaintOrder,
    paintLayerId,
}: FinishStrokeInput): FinishStrokeResult => {
    if (!draftStroke || draftStroke.length < 2 || !object || !view) {
        return {
            kind: "discard",
            restoreSnapshot: pendingStrokeUndoSnapshot ?? undefined,
        };
    }

    const sourcePoints = samplePaintStrokeSpline(draftStroke);
    const geometry = buildRibbonStrokeGeometry(sourcePoints, view, brush.width);
    if (!geometry || ribbonSegmentCount(geometry.ribbon) === 0) {
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
            sourcePoints,
            ribbon: geometry.ribbon,
            style: { ...brush },
            paintOrder: nextPaintOrder(object.id),
        },
    };
};
