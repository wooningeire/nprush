import type {
    PaintObject,
    PaintStroke,
} from "../types.ts";

export type ObjectDeletionResult = {
    objects: PaintObject[],
    strokes: PaintStroke[],
    activeObjectId: string | null,
};

export const deletePaintObject = (
    objectId: string,
    objects: PaintObject[],
    strokes: PaintStroke[],
    activeObjectId: string | null,
): ObjectDeletionResult | null => {
    if (!objects.some(object => object.id === objectId)) return null;

    const nextObjects = objects.filter(object => object.id !== objectId);
    return {
        objects: nextObjects,
        strokes: strokes.filter(stroke => stroke.objectId !== objectId),
        activeObjectId: activeObjectId === objectId
            ? nextObjects[0]?.id ?? null
            : activeObjectId,
    };
};
