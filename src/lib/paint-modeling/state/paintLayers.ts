import { makeId } from "./sceneData.ts";
import type { PaintLayer } from "../types.ts";

export const BASE_PAINT_LAYER_ID = "paint-layer-base";

export const createBasePaintLayer = (): PaintLayer => ({
    id: BASE_PAINT_LAYER_ID,
    name: "Layer 1",
    order: 0,
    visible: true,
});

export const createPaintLayer = (order: number): PaintLayer => ({
    id: makeId("paint-layer"),
    name: `Layer ${order + 1}`,
    order,
    visible: true,
});