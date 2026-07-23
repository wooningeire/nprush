import {
    clonePaintLayer,
    cloneObject,
    cloneStroke,
} from "./sceneData.ts";
import type {
    PaintLayer,
    PaintObject,
    PaintStroke,
} from "../types.ts";

export type PaintSceneSnapshot = {
    viewportWidth: number,
    viewportHeight: number,
    objects: PaintObject[],
    paintLayers: PaintLayer[],
    strokes: PaintStroke[],
    activeObjectId: string | null,
    activePaintLayerId: string,
};

export type PaintSceneSnapshotSource = PaintSceneSnapshot;

export type RestoredPaintScene = PaintSceneSnapshotSource;

export const capturePaintSceneSnapshot = (
    source: PaintSceneSnapshotSource,
): PaintSceneSnapshot => ({
    viewportWidth: source.viewportWidth,
    viewportHeight: source.viewportHeight,
    objects: source.objects.map(cloneObject),
    paintLayers: source.paintLayers.map(clonePaintLayer),
    strokes: source.strokes.map(cloneStroke),
    activeObjectId: source.activeObjectId,
    activePaintLayerId: source.activePaintLayerId,
});

export const restorePaintSceneSnapshot = (
    snapshot: PaintSceneSnapshot,
): RestoredPaintScene => ({
    viewportWidth: snapshot.viewportWidth,
    viewportHeight: snapshot.viewportHeight,
    objects: snapshot.objects.map(cloneObject),
    paintLayers: snapshot.paintLayers.map(clonePaintLayer),
    strokes: snapshot.strokes.map(cloneStroke),
    activeObjectId: snapshot.activeObjectId,
    activePaintLayerId: snapshot.activePaintLayerId,
});
