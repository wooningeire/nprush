export type Vec2 = {
    x: number,
    y: number,
};

export type Vec3 = [number, number, number];
export type Vec4 = [number, number, number, number];

export type PaintLayer = {
    id: string,
    name: string,
    order: number,
    visible: boolean,
};

export type PaintView = {
    id: string,
    name: string,
    order: number,
    long: number,
    lat: number,
    radius: number,
    offset: Vec3,
    width: number,
    height: number,
    viewProjMat: number[],
    viewProjInvMat: number[],
    viewMat: number[],
    viewInvMat: number[],
    createdAt: number,
};

export type BrushStyle = {
    color: string,
    width: number,
    opacity: number,
};

export type RibbonUv = {
    u: number,
    v: number,
};

export type PaintRibbonVertex = {
    position: Vec3,
    u: number,
    v: number,
};

export type PaintRibbonFace = [number, number, number, number];

export type PaintRibbonMesh = {
    rows: number,
    columns: number[],
    closed: boolean,
    vertices: PaintRibbonVertex[],
    faces: PaintRibbonFace[],
};

export type DeformationLine = {
    id: string,
    points: RibbonUv[],
};

export type PaintStroke = {
    id: string,
    objectId: string,
    layerId?: string,
    sourceViewId: string,
    sourcePoints: Vec2[],
    centerline: Vec3[],
    mesh: PaintRibbonMesh,
    deformationLines: DeformationLine[],
    style: BrushStyle,
    paintOrder: number,
};

export type PaintObject = {
    id: string,
    name: string,
    visible: boolean,
    locked: boolean,
    layerIndex: number,
};

export type StrokeSurfaceHit = {
    objectId: string,
    strokeId: string,
    faceIndex: number,
    uv: RibbonUv,
    world: Vec3,
    viewDepth: number,
};

export type RenderSegment = {
    kind?: "segment",
    a: Vec3,
    b: Vec3,
    color: Vec4,
    width?: number,
    capStart?: boolean,
    capEnd?: boolean,
};

export type RenderTriangle = {
    kind: "triangle",
    a: Vec3,
    b: Vec3,
    c: Vec3,
    color: Vec4,
    normal?: Vec3,
    shade?: number,
    depthWrite?: boolean,
};

export type RenderStroke = {
    kind: "stroke",
    points: Vec3[],
    color: Vec4,
    width: number,
};

export type RenderPrimitive = RenderSegment | RenderTriangle | RenderStroke;

export type PaintRenderOptions = {
    showDraftStroke?: boolean,
    shadeRibbons?: boolean,
};