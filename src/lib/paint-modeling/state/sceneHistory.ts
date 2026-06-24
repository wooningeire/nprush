import {
    clonePaintLayer,
    cloneObject,
    cloneOcclusionClaim,
    cloneStroke,
    cloneView,
} from "./sceneData.ts";
import type {
    OcclusionClaim,
    PaintLayer,
    PaintObject,
    PaintStroke,
    PaintView,
} from "../types.ts";

export type PaintSceneSnapshot = {
    viewportWidth: number,
    viewportHeight: number,
    views: PaintView[],
    objects: PaintObject[],
    paintLayers: PaintLayer[],
    strokes: PaintStroke[],
    occlusionClaims: OcclusionClaim[],
    activeObjectId: string | null,
    activeViewId: string | null,
    activePaintLayerId: string,
};

export type PaintSceneSnapshotSource = {
    viewportWidth: number,
    viewportHeight: number,
    views: PaintView[],
    objects: PaintObject[],
    paintLayers: PaintLayer[],
    strokes: PaintStroke[],
    occlusionClaims: OcclusionClaim[],
    activeObjectId: string | null,
    activeViewId: string | null,
    activePaintLayerId: string,
};

export type RestoredPaintScene = PaintSceneSnapshotSource;

export const capturePaintSceneSnapshot = (
    source: PaintSceneSnapshotSource,
): PaintSceneSnapshot => ({
    viewportWidth: source.viewportWidth,
    viewportHeight: source.viewportHeight,
    views: source.views.map(cloneView),
    objects: source.objects.map(cloneObject),
    paintLayers: source.paintLayers.map(clonePaintLayer),
    strokes: source.strokes.map(cloneStroke),
    occlusionClaims: source.occlusionClaims.map(cloneOcclusionClaim),
    activeObjectId: source.activeObjectId,
    activeViewId: source.activeViewId,
    activePaintLayerId: source.activePaintLayerId,
});

export const restorePaintSceneSnapshot = (
    snapshot: PaintSceneSnapshot,
): RestoredPaintScene => ({
    viewportWidth: snapshot.viewportWidth,
    viewportHeight: snapshot.viewportHeight,
    views: snapshot.views.map(cloneView),
    objects: snapshot.objects.map(cloneObject),
    paintLayers: snapshot.paintLayers.map(clonePaintLayer),
    strokes: snapshot.strokes.map(cloneStroke),
    occlusionClaims: snapshot.occlusionClaims.map(cloneOcclusionClaim),
    activeObjectId: snapshot.activeObjectId,
    activeViewId: snapshot.activeViewId,
    activePaintLayerId: snapshot.activePaintLayerId,
});
