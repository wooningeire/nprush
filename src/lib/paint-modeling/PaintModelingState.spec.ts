import { describe, expect, it } from "vitest";
import { PaintModelingState } from "./PaintModelingState.svelte.ts";
import { BrushPlacementMode } from "./types.ts";
import { RIBBON_RENDER_COLUMNS } from "./state/constants.ts";
import type {
    PaintRibbon,
    RenderPrimitive,
    RenderRibbon,
    RenderStroke,
    RenderTriangle,
    Vec2,
    Vec3,
} from "./types.ts";

const drawStroke = (state: PaintModelingState, a: Vec2, b: Vec2) => {
    state.beginStroke(a, 800, 600);
    state.appendStrokePoint(b);
    state.finishStroke();
};

describe("PaintModelingState stroke-owned ribbons", () => {
    it("captures viewport aspect in transient stroke projections", () => {
        const wideState = new PaintModelingState();
        const tallState = new PaintModelingState();

        wideState.beginStroke({ x: 0, y: 0 }, 1200, 600);
        tallState.beginStroke({ x: 0, y: 0 }, 600, 1200);
        const wideProjection = wideState.strokePlacementProjection();
        const tallProjection = tallState.strokePlacementProjection();

        expect(wideProjection).not.toBeNull();
        expect(tallProjection).not.toBeNull();
        expect(projectedSpanAspect(wideProjection!.viewProjInvMat)).toBeCloseTo(2, 5);
        expect(projectedSpanAspect(tallProjection!.viewProjInvMat)).toBeCloseTo(0.5, 5);
    });

    it("holds the source projection stable for the active stroke", () => {
        const state = new PaintModelingState();

        state.beginStroke({ x: -0.2, y: 0 }, 1200, 600);
        const projection = state.strokePlacementProjection();
        const viewProjInvMat = [...projection!.viewProjInvMat];

        state.viewportWidth = 600;
        state.viewportHeight = 1200;
        state.orbit.turn({ x: 24, y: 12 });

        expect(state.strokePlacementProjection()).toBe(projection);
        expect(state.strokePlacementProjection()?.width).toBe(1200);
        expect(state.strokePlacementProjection()?.height).toBe(600);
        expect(state.strokePlacementProjection()?.viewProjInvMat).toEqual(viewProjInvMat);
    });

    it("does not persist projection snapshots as scene objects", () => {
        const state = new PaintModelingState();
        drawStroke(state, { x: -0.14, y: 0 }, { x: 0.14, y: 0 });

        expect(state.strokePlacementProjection()).toBeNull();
        expect(state.strokes[0]).not.toHaveProperty("sourceViewId");
        expect(state).not.toHaveProperty("views");
        expect(state).not.toHaveProperty("activeViewId");
    });

    it("creates an oriented-vertex ribbon for a brushstroke", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setBrushWidth(36);
        drawStroke(state, { x: -0.28, y: -0.08 }, { x: 0.28, y: 0.12 });

        const stroke = state.strokes[0];
        const ribbon = stroke.ribbon;

        expect(state.objects).toHaveLength(1);
        expect(state.strokes).toHaveLength(1);
        expect(ribbon.closed).toBe(false);
        expect(ribbon.vertices.length).toBeGreaterThan(1);
        expect(ribbon.vertices[0].u).toBe(0);
        expect(ribbon.vertices.at(-1)?.u).toBeCloseTo(1, 5);
        expect(ribbon.vertices.every(vertex => distance3(vertex.side, [0, 0, 0]) > 0)).toBe(true);
    });

    it("keeps a ring-like stroke as a closed oriented-vertex ribbon", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setBrushWidth(48);
        state.beginStroke({ x: 0.22, y: 0 }, 800, 600);
        for (let i = 1; i <= 96; i++) {
            const angle = i / 96 * Math.PI * 2;
            state.appendStrokePoint({
                x: Math.cos(angle) * 0.22,
                y: Math.sin(angle) * 0.22,
            });
        }
        state.finishStroke();

        const ribbon = state.strokes[0].ribbon;

        expect(ribbon.closed).toBe(true);
        expect(ribbon.vertices.length).toBeGreaterThan(8);
        expect(ribbon.vertices.at(-1)?.u).toBeLessThan(1);
    });

    it("selects GPU brush placement modes without changing brush style", () => {
        const state = new PaintModelingState();

        expect(state.brushPlacementMode).toBe(BrushPlacementMode.View);

        for (const mode of [
            BrushPlacementMode.StartDepth,
            BrushPlacementMode.StartPlane,
            BrushPlacementMode.Surface,
            BrushPlacementMode.ConstructionPlane,
        ]) {
            state.setBrushPlacementMode(mode);
            expect(state.brushPlacementMode).toBe(mode);
        }

        expect(state.brush.width).toBe(18);
    });

    it("edits a normalized construction plane at camera-forward view depth", () => {
        const state = new PaintModelingState();

        state.setConstructionPlaneViewDepth(2.4);
        expect(state.constructionPlaneViewDepth).toBeCloseTo(2.4, 5);

        state.setConstructionPlaneNormal([0, 3, 4]);
        expect(state.constructionPlane.normal).toEqual([0, 0.6, 0.8]);

        state.flipConstructionPlaneNormal();
        expect(state.constructionPlane.normal).toEqual([-0, -0.6, -0.8]);

        const depthBeforeAlign = state.constructionPlaneViewDepth;
        state.alignConstructionPlaneToView();

        expect(Math.hypot(...state.constructionPlane.normal)).toBeCloseTo(1, 6);
        expect(state.constructionPlaneViewDepth).toBeCloseTo(depthBeforeAlign, 6);
    });

    it("commits a renderer-produced ribbon for snapped strokes", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.beginStroke({ x: -0.2, y: 0 }, 800, 600);
        state.appendStrokePoint({ x: 0.2, y: 0.05 });
        const sourcePoints = state.draftStrokeSourcePoints();
        const ribbon: PaintRibbon = {
            closed: false,
            vertices: [
                { position: [-0.1, 0, 0], side: [0, 0.02, 0], u: 0 },
                { position: [0.1, 0, 0], side: [0, 0.02, 0], u: 1 },
            ],
        };

        expect(sourcePoints).not.toBeNull();
        expect(state.finishStrokeWithRibbon(sourcePoints!, ribbon)).toBe(true);

        expect(state.strokes).toHaveLength(1);
        expect(state.strokes[0].sourcePoints).toBe(sourcePoints);
        expect(state.strokes[0].ribbon).toBe(ribbon);
        expect(state.draftStroke).toBeNull();
    });
    it("does not expose CPU ribbon raycast or deformation hooks", () => {
        const state = new PaintModelingState() as unknown as Record<string, unknown>;

        expect(state.raycastStrokeAt).toBeUndefined();
        expect(state.sculptStrokeAt).toBeUndefined();
        expect(state.addDeformationLine).toBeUndefined();
    });

    it("renders committed ribbons as GPU-expanded ribbon primitives", () => {
        const state = new PaintModelingState();
        state.addObject();
        drawStroke(state, { x: -0.24, y: 0 }, { x: 0.24, y: 0 });

        const primitives = state.buildRenderSegments({ showDraftStroke: false });
        const ribbons = primitives.filter(isRenderRibbon);

        expect(ribbons).toHaveLength(1);
        expect(ribbons[0].vertices).toBe(state.strokes[0].ribbon.vertices);
        expect(ribbons[0].shade).toBe(1);
        expect(ribbons[0].depthBias ?? 0).toBe(0);
        expect(RIBBON_RENDER_COLUMNS).toBe(2);
        expect(primitives.filter(isRenderTriangle)).toHaveLength(0);
        expect(primitives.filter(isRenderStroke)).toHaveLength(0);
    });

    it("renders draft ribbons as opaque depth-biased GPU primitives", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.beginStroke({ x: -0.24, y: 0 }, 800, 600);
        state.appendStrokePoint({ x: 0.24, y: 0 });

        const primitives = state.buildDraftRenderSegments();
        const ribbons = primitives.filter(isRenderRibbon);

        expect(ribbons).toHaveLength(1);
        expect(ribbons[0].color[3]).toBe(1);
        expect(ribbons[0].depthBias).toBeGreaterThan(0);
        expect(primitives.filter(isRenderTriangle)).toHaveLength(0);
    });

    it("undoes a paint stroke together with its auto-created object", () => {
        const state = new PaintModelingState();
        drawStroke(state, { x: -0.14, y: 0 }, { x: 0.14, y: 0 });

        expect(state.objects).toHaveLength(1);
        expect(state.strokes).toHaveLength(1);

        expect(state.undo()).toBe(true);

        expect(state.objects).toHaveLength(0);
        expect(state.strokes).toHaveLength(0);
        expect(state.activeObjectId).toBeNull();
    });

    it("deletes objects with their ribbons", () => {
        const state = new PaintModelingState();
        state.addObject("First");
        const objectId = state.activeObjectId!;
        drawStroke(state, { x: -0.14, y: 0 }, { x: 0.14, y: 0 });

        state.addObject("Second");
        drawStroke(state, { x: -0.1, y: 0 }, { x: 0.1, y: 0 });
        expect(state.strokes).toHaveLength(2);
        expect(state.deleteObject(objectId)).toBe(true);
        expect(state.objects.some(object => object.name === "First")).toBe(false);
        expect(state.strokes).toHaveLength(1);
        expect(state.deleteObject(state.activeObjectId!)).toBe(true);
        expect(state.objects.some(object => object.name === "Second")).toBe(false);
    });
});

const isRenderRibbon = (primitive: RenderPrimitive): primitive is RenderRibbon => primitive.kind === "ribbon";

const isRenderTriangle = (primitive: RenderPrimitive): primitive is RenderTriangle => primitive.kind === "triangle";

const isRenderStroke = (primitive: RenderPrimitive): primitive is RenderStroke => primitive.kind === "stroke";

const projectedSpanAspect = (viewProjInvMat: number[]): number => {
    const left = unprojectNdc(viewProjInvMat, -1, 0, 0.5);
    const right = unprojectNdc(viewProjInvMat, 1, 0, 0.5);
    const bottom = unprojectNdc(viewProjInvMat, 0, -1, 0.5);
    const top = unprojectNdc(viewProjInvMat, 0, 1, 0.5);

    return distance3(left, right) / distance3(bottom, top);
};

const unprojectNdc = (
    viewProjInvMat: number[],
    x: number,
    y: number,
    z: number,
): Vec3 => {
    const worldX = viewProjInvMat[0] * x + viewProjInvMat[4] * y + viewProjInvMat[8] * z + viewProjInvMat[12];
    const worldY = viewProjInvMat[1] * x + viewProjInvMat[5] * y + viewProjInvMat[9] * z + viewProjInvMat[13];
    const worldZ = viewProjInvMat[2] * x + viewProjInvMat[6] * y + viewProjInvMat[10] * z + viewProjInvMat[14];
    const worldW = viewProjInvMat[3] * x + viewProjInvMat[7] * y + viewProjInvMat[11] * z + viewProjInvMat[15];
    if (!Number.isFinite(worldW) || Math.abs(worldW) <= 1e-6) {
        throw new Error("Cannot unproject NDC point");
    }
    return [worldX / worldW, worldY / worldW, worldZ / worldW];
};

const distance3 = (a: Vec3, b: Vec3): number => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
