import type { PaintLayer, PaintObject } from "../types.ts";

type OrderedItemReorderOptions<T> = {
    items: T[],
    sourceId: string,
    targetId: string,
    idFor: (item: T) => string,
    orderFor: (item: T) => number,
    withOrder: (item: T, order: number) => T,
};

export const reorderOrderedItems = <T>({
    items,
    sourceId,
    targetId,
    idFor,
    orderFor,
    withOrder,
}: OrderedItemReorderOptions<T>): T[] | null => {
    if (sourceId === targetId) return null;

    const orderedItems = items
        .map((item, index) => ({ item, index }))
        .sort((a, b) => orderFor(a.item) - orderFor(b.item) || a.index - b.index)
        .map(({ item }) => item);
    const sourceIndex = orderedItems.findIndex(item => idFor(item) === sourceId);
    const targetIndex = orderedItems.findIndex(item => idFor(item) === targetId);

    if (sourceIndex < 0 || targetIndex < 0) return null;

    const nextItems = [...orderedItems];
    const [sourceItem] = nextItems.splice(sourceIndex, 1);
    nextItems.splice(targetIndex, 0, sourceItem);

    return nextItems.map((item, order) => withOrder(item, order));
};

export const reorderPaintLayers = (
    layers: PaintLayer[],
    sourceId: string,
    targetId: string,
): PaintLayer[] | null => reorderOrderedItems({
    items: layers,
    sourceId,
    targetId,
    idFor: layer => layer.id,
    orderFor: layer => layer.order,
    withOrder: (layer, order) => ({ ...layer, order }),
});

export const reorderPaintObjects = (
    objects: PaintObject[],
    sourceId: string,
    targetId: string,
): PaintObject[] | null => reorderOrderedItems({
    items: objects,
    sourceId,
    targetId,
    idFor: object => object.id,
    orderFor: object => object.layerIndex,
    withOrder: (object, layerIndex) => ({ ...object, layerIndex }),
});
