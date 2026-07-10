import { describe, expect, it } from "vitest";
import guideShader from "./paint_brush_guide.wgsl?raw";
import placementShader from "./paint_brush_placement.wgsl?raw";
import placementPrelude from "./paint_brush_placement_prelude.wgsl?raw";

describe("paint brush GPU placement shaders", () => {
    it("keeps raycast and plane projection in the shared GPU prelude", () => {
        expect(placementPrelude).toContain("fn surface_hit");
        expect(placementPrelude).toContain("fn project_plane");
        expect(placementPrelude).not.toContain("forward_arc_distance(from");
    });

    it("resolves direct hits before parallel surface-gap placement", () => {
        expect(placementShader).toContain("fn compute_direct_stroke");
        expect(placementShader).toContain("fn compute_stroke");
        expect(placementShader).toContain("fn find_hit_neighbors");
        expect(placementShader).toContain("fn forward_arc_distance");
        expect(placementShader).toContain("distance_from_a / gap_distance");
    });

    it("uses endpoint derivatives for C1 plane interpolation", () => {
        expect(placementShader).toContain("fn plane_derivative_before");
        expect(placementShader).toContain("fn plane_derivative_after");
        expect(placementShader).toContain("fn hermite_vec3");
        expect(placementShader).toContain("derivative_a.origin * gap_distance");
        expect(placementShader).toContain("derivative_b.normal * gap_distance");
    });

    it("renders a finite construction grid and an occluded xray pass", () => {
        expect(guideShader).toContain("GUIDE_GRID_HALF_LINES");
        expect(guideShader).toContain("placement_uniforms.construction_origin");
        expect(guideShader).toContain("placement_uniforms.construction_normal");
        expect(guideShader).toContain("fn guide_fragment_xray");
    });
});