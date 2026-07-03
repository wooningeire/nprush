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

export type PaintRibbonVertex = {
    position: Vec3,
    side: Vec3,
    u: number,
};

export type PaintRibbon = {
    closed: boolean,
    vertices: PaintRibbonVertex[],
};

export type PaintStroke = {
    id: string,
    objectId: string,
    layerId?: string,
    sourceViewId: string,
    sourcePoints: Vec2[],
    ribbon: PaintRibbon,
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

export type RenderRibbon = {
    kind: "ribbon",
    vertices: PaintRibbonVertex[],
    closed: boolean,
    color: Vec4,
    shade?: number,
    depthBias?: number,
};

export type RenderPrimitive = RenderSegment | RenderTriangle | RenderStroke | RenderRibbon;

export type PaintRenderOptions = {
    showDraftStroke?: boolean,
    shadeRibbons?: boolean,
};
