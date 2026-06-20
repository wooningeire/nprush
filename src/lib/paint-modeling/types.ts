export interface Vec2 {
    x: number;
    y: number;
}

export type Vec3 = [number, number, number];
export type Vec4 = [number, number, number, number];

export type PlacementMode = "snap" | "new-surface" | "occluding-surface" | "paint-behind";
export type BrushMode = "color" | "surface";
export type ChartRole = "surface" | "occluder" | "behind";
export type ChartProjectionMode = "view-plane" | "ray-depth";

export interface PaintView {
    id: string;
    name: string;
    long: number;
    lat: number;
    radius: number;
    offset: Vec3;
    width: number;
    height: number;
    viewProjMat: number[];
    viewProjInvMat: number[];
    viewMat: number[];
    viewInvMat: number[];
    createdAt: number;
}

export interface BrushStyle {
    color: string;
    width: number;
    opacity: number;
}

export interface SurfaceRef {
    chartId: string;
    uv: Vec2;
}

export interface PaintSample {
    sourcePoint: Vec2;
    surfaceRef: SurfaceRef;
    placement: PlacementMode;
}

export interface PaintStroke {
    id: string;
    objectId: string;
    sourceViewId: string;
    placement: PlacementMode;
    samples: PaintSample[];
    style: BrushStyle;
    paintOrder: number;
}

export interface PaintChart {
    id: string;
    objectId: string;
    sourceViewId: string;
    role: ChartRole;
    projectionMode: ChartProjectionMode;
    width: number;
    height: number;
    depths: number[];
    coverage: number[];
    paint: number[];
    seams: boolean[];
    createdAt: number;
}

export interface PaintObject {
    id: string;
    name: string;
    visible: boolean;
    locked: boolean;
    layerIndex: number;
    charts: PaintChart[];
}

export interface OcclusionClaim {
    id: string;
    objectId: string;
    viewId: string;
    frontChartId: string;
    backRefs: SurfaceRef[];
    mask: Vec2[];
    createdAt: number;
}

export interface SurfaceHit {
    objectId: string;
    chartId: string;
    surfaceRef: SurfaceRef;
    world: Vec3;
    viewDepth: number;
}

export interface RenderSegment {
    kind?: "segment";
    a: Vec3;
    b: Vec3;
    color: Vec4;
    width?: number;
    capStart?: boolean;
    capEnd?: boolean;
}

export interface RenderTriangle {
    kind: "triangle";
    a: Vec3;
    b: Vec3;
    c: Vec3;
    color: Vec4;
}

export interface RenderStroke {
    kind: "stroke";
    points: Vec3[];
    color: Vec4;
    width: number;
}

export type RenderPrimitive = RenderSegment | RenderTriangle | RenderStroke;

export interface PaintRenderOptions {
    showChartWireframe?: boolean;
    showSurfaceField?: boolean;
    showDraftStroke?: boolean;
}
