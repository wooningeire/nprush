import type {
    PaintLayer,
    PaintObject,
    PaintRibbon,
    PaintRibbonVertex,
    PaintStroke,
    PaintView,
    Vec2,
    Vec3,
} from "../types.ts";

export const cloneView = (view: PaintView): PaintView => {
    return {
        ...view,
        offset: [...view.offset] as Vec3,
        viewProjMat: [...view.viewProjMat],
        viewProjInvMat: [...view.viewProjInvMat],
        viewMat: [...view.viewMat],
        viewInvMat: [...view.viewInvMat],
    };
};

export const cloneObject = (object: PaintObject): PaintObject => ({ ...object });

export const clonePaintLayer = (layer: PaintLayer): PaintLayer => ({ ...layer });

export const cloneStroke = (stroke: PaintStroke): PaintStroke => {
    return {
        ...stroke,
        sourcePoints: stroke.sourcePoints.map(cloneVec2),
        ribbon: cloneRibbon(stroke.ribbon),
        style: { ...stroke.style },
    };
};

export const cloneRibbon = (ribbon: PaintRibbon): PaintRibbon => ({
    closed: ribbon.closed,
    vertices: ribbon.vertices.map(cloneRibbonVertex),
});

export const makeId = (prefix: string): string => `${prefix}-${Math.random().toString(36).slice(2, 10)}`;

const cloneRibbonVertex = (vertex: PaintRibbonVertex): PaintRibbonVertex => ({
    ...vertex,
    position: [...vertex.position] as Vec3,
    side: [...vertex.side] as Vec3,
});

const cloneVec2 = (point: Vec2): Vec2 => ({ ...point });
