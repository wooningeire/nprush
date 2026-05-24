import type { MeshData } from "$/gpu/file-load/loadGlb";

export type LegacyContourRole = "exterior" | "interior" | "occluded";
export type ContourStrokeKind = "edge" | "contour";

export const CONTOUR_STROKE_KIND_WEIGHTS: Record<ContourStrokeKind, number> = {
    edge: 1,
    contour: 0.85,
};

export interface Vec2 {
    x: number;
    y: number;
}

export type Vec3 = [number, number, number];

export interface ContourView {
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

export interface ContourStroke {
    id: string;
    kind: ContourStrokeKind;
    viewId: string;
    shapeId: string;
    points: Vec2[];
    resampledPoints: Vec2[];
    tangents: Vec2[];
    normals: Vec2[];
    weight: number;
    depthNdc?: number;
    depthOffset?: number;
    depthLocked?: boolean;
    depthSamplesNdc?: number[];
    depthSamplesOffset?: number[];
    depthSamplesLocked?: boolean[];
}

export interface ImplicitBodyParams {
    center: Vec3;
    axisX?: Vec3;
    axisY?: Vec3;
    axisZ?: Vec3;
    height: number;
    radiusBottom: number;
    radiusTop: number;
    bulge: number;
    ovalX: number;
    ovalZ: number;
    boxiness: number;
}

export type FitStatus = "idle" | "fitting" | "fitted" | "failed" | "canceled";

export interface ImplicitShape {
    id: string;
    name: string;
    params: ImplicitBodyParams;
    mesh: MeshData | null;
    fitStatus: FitStatus;
    fitLoss: number | null;
    strokeIds: string[];
}

export interface FitSample {
    point: Vec2;
    kind: ContourStrokeKind;
    viewIndex: number;
    weight: number;
}

export interface FitView {
    id: string;
    viewProjMat: number[];
    viewProjInvMat?: number[];
    viewInvMat?: number[];
    width?: number;
    height?: number;
}
