import {
    cloneObject,
    cloneOcclusionClaim,
    cloneStroke,
    cloneView,
} from "./sceneData.ts";
import type {
    ChartProjectionMode,
    OcclusionClaim,
    PaintObject,
    PaintStroke,
    PaintView,
} from "../types.ts";

export type PaintSceneSnapshot = {
    viewportWidth: number,
    viewportHeight: number,
    views: PaintView[],
    objects: PaintObject[],
    strokes: PaintStroke[],
    occlusionClaims: OcclusionClaim[],
    activeObjectId: string | null,
    activeViewId: string | null,
    chartProjectionMode: ChartProjectionMode,
};

export type PaintSceneSnapshotSource = {
    viewportWidth: number,
    viewportHeight: number,
    views: PaintView[],
    objects: PaintObject[],
    strokes: PaintStroke[],
    occlusionClaims: OcclusionClaim[],
    activeObjectId: string | null,
    activeViewId: string | null,
    chartProjectionMode: ChartProjectionMode,
};

export type RestoredPaintScene = PaintSceneSnapshotSource;

export const capturePaintSceneSnapshot = (
    source: PaintSceneSnapshotSource,
): PaintSceneSnapshot => ({
    viewportWidth: source.viewportWidth,
    viewportHeight: source.viewportHeight,
    views: source.views.map(cloneView),
    objects: source.objects.map(cloneObject),
    strokes: source.strokes.map(cloneStroke),
    occlusionClaims: source.occlusionClaims.map(cloneOcclusionClaim),
    activeObjectId: source.activeObjectId,
    activeViewId: source.activeViewId,
    chartProjectionMode: source.chartProjectionMode,
});

export const restorePaintSceneSnapshot = (
    snapshot: PaintSceneSnapshot,
): RestoredPaintScene => ({
    viewportWidth: snapshot.viewportWidth,
    viewportHeight: snapshot.viewportHeight,
    views: snapshot.views.map(cloneView),
    objects: snapshot.objects.map(cloneObject),
    strokes: snapshot.strokes.map(cloneStroke),
    occlusionClaims: snapshot.occlusionClaims.map(cloneOcclusionClaim),
    activeObjectId: snapshot.activeObjectId,
    activeViewId: snapshot.activeViewId,
    chartProjectionMode: snapshot.chartProjectionMode,
});
