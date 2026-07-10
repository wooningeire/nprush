import {
    BrushPlacementMode,
    BrushPlacementProvenance,
    type BrushPlacementMode as BrushPlacementModeValue,
    type BrushPlacementProvenance as BrushPlacementProvenanceValue,
    type ConstructionPlane,
    type PaintView,
    type Vec2,
    type Vec3,
} from "../types.ts";

export const PLACEMENT_UNIFORM_FLOATS = 72;
export const PLACEMENT_RESULT_FLOATS = 16;
export const TARGET_INFO_UINTS = 4;
export const GUIDE_VERTEX_COUNT = 48 * 2 + 2 + (8 * 2 + 1) * 4 + 2;
export const WORKGROUP_SIZE = 64;

export const placementModeUsesTargets = (mode: BrushPlacementModeValue): boolean => (
    mode === BrushPlacementMode.StartDepth
    || mode === BrushPlacementMode.StartPlane
    || mode === BrushPlacementMode.Surface
);

export const placementModeUniformValue = (mode: BrushPlacementModeValue): number => {
    if (mode === BrushPlacementMode.StartDepth) return 1;
    if (mode === BrushPlacementMode.StartPlane) return 2;
    if (mode === BrushPlacementMode.Surface) return 3;
    if (mode === BrushPlacementMode.ConstructionPlane) return 4;
    return 0;
};

export const provenanceFromUniformValue = (
    value: number,
): BrushPlacementProvenanceValue => {
    if (value === 1) return BrushPlacementProvenance.Surface;
    if (value === 2) return BrushPlacementProvenance.Bridge;
    if (value === 3) return BrushPlacementProvenance.StartDepth;
    if (value === 4) return BrushPlacementProvenance.StartPlane;
    if (value === 5) return BrushPlacementProvenance.ConstructionPlane;
    return BrushPlacementProvenance.View;
};

type PlacementSettings = {
    placementMode: BrushPlacementModeValue,
    constructionPlane: ConstructionPlane,
};

export type BrushPlacementInput = PlacementSettings & {
    point: Vec2,
    brushWidth: number,
    viewportWidth: number,
    viewportHeight: number,
    pointerVisible: boolean,
    planeSize: number,
    startPoint: Vec2 | null,
};

export type BrushPlacementReadback = {
    center: Vec3,
    normal: Vec3,
    tangent: Vec3,
    bitangent: Vec3,
    depth: number,
    provenance: BrushPlacementProvenanceValue,
    snapped: boolean,
};

export type StrokePlacementInput = PlacementSettings & {
    sourcePoints: Vec2[],
    view: PaintView,
    brushWidth: number,
};