import type {
    DeformationLine,
    PaintLayer,
    PaintObject,
    PaintRibbonMesh,
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
        centerline: stroke.centerline.map(point => [...point] as Vec3),
        mesh: cloneRibbonMesh(stroke.mesh),
        deformationLines: stroke.deformationLines.map(cloneDeformationLine),
        style: { ...stroke.style },
    };
};

export const cloneRibbonMesh = (mesh: PaintRibbonMesh): PaintRibbonMesh => ({
    rows: mesh.rows,
    columns: [...mesh.columns],
    closed: mesh.closed,
    vertices: mesh.vertices.map(cloneRibbonVertex),
    faces: mesh.faces.map(face => [...face]),
});

export const makeId = (prefix: string): string => `${prefix}-${Math.random().toString(36).slice(2, 10)}`;

const cloneRibbonVertex = (vertex: PaintRibbonVertex): PaintRibbonVertex => ({
    ...vertex,
    position: [...vertex.position] as Vec3,
});

const cloneDeformationLine = (line: DeformationLine): DeformationLine => ({
    ...line,
    points: line.points.map(point => ({ ...point })),
});


const cloneVec2 = (point: Vec2): Vec2 => ({ ...point });
