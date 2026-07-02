import { describe, expect, it } from "vitest";
import { PaintModelingState } from "./PaintModelingState.svelte.ts";
import {
    evaluatedStrokeMesh,
    meshConnectedComponentCount,
} from "./state/strokeMesh.ts";
import type {
    PaintView,
    RenderPrimitive,
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
    it("uses viewport aspect when calibrating paint-view rays", () => {
        const state = new PaintModelingState();

        const wideView = state.saveCurrentView(1200, 600, false);
        const tallView = state.saveCurrentView(600, 1200, false);

        expect(projectedSpanAspect(wideView.viewProjInvMat)).toBeCloseTo(2, 5);
        expect(projectedSpanAspect(tallView.viewProjInvMat)).toBeCloseTo(0.5, 5);
    });

    it("treats resized paint views as a different calibrated projection", () => {
        const state = new PaintModelingState();

        state.saveCurrentView(1200, 600, false);
        expect(state.isCameraAtActiveView).toBe(true);

        state.viewportWidth = 600;
        state.viewportHeight = 600;

        expect(state.isCameraAtActiveView).toBe(false);
    });

    it("creates one connected ribbon mesh for a brushstroke", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setBrushWidth(36);
        drawStroke(state, { x: -0.28, y: -0.08 }, { x: 0.28, y: 0.12 });

        const stroke = state.strokes[0];
        const mesh = stroke.mesh;

        expect(state.objects).toHaveLength(1);
        expect(state.strokes).toHaveLength(1);
        expect(stroke.centerline).toHaveLength(mesh.rows);
        expect(mesh.columns).toEqual([-1, 0, 1]);
        expect(mesh.vertices).toHaveLength(mesh.rows * mesh.columns.length);
        expect(mesh.faces).toHaveLength((mesh.rows - 1) * 2);
        expect(meshConnectedComponentCount(mesh)).toBe(1);
        expect(mesh.faces.every(face => new Set(face).size === 4)).toBe(true);
    });

    it("keeps a ring-like stroke as one connected surface", () => {
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

        const mesh = state.strokes[0].mesh;

        expect(mesh.closed).toBe(true);
        expect(meshConnectedComponentCount(mesh)).toBe(1);
        expect(mesh.faces).toHaveLength(mesh.rows * (mesh.columns.length - 1));
    });

    it("sculpts a ribbon without changing mesh connectivity", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setBrushWidth(36);
        drawStroke(state, { x: -0.22, y: 0 }, { x: 0.22, y: 0 });

        const stroke = state.strokes[0];
        const verticesBefore = stroke.mesh.vertices.map(vertex => [...vertex.position] as Vec3);
        const facesBefore = stroke.mesh.faces.map(face => [...face]);
        const moved = state.sculptStrokeAt(stroke.id, { u: 0.5, v: 0 }, [0, 0.08, 0], 0.24);
        const sculpted = state.strokes[0];
        const movement = sculpted.mesh.vertices.map((vertex, index) => distance3(vertex.position, verticesBefore[index]));

        expect(moved).toBe(true);
        expect(sculpted.mesh.vertices).toHaveLength(verticesBefore.length);
        expect(sculpted.mesh.faces).toEqual(facesBefore);
        expect(meshConnectedComponentCount(sculpted.mesh)).toBe(1);
        expect(Math.max(...movement)).toBeGreaterThan(0.01);
    });

    it("adds a deformation line as local support columns", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setBrushWidth(42);
        drawStroke(state, { x: -0.3, y: -0.04 }, { x: 0.3, y: 0.04 });

        const stroke = state.strokes[0];
        const rowsBefore = stroke.mesh.rows;
        const columnsBefore = stroke.mesh.columns.length;
        const facesBefore = stroke.mesh.faces.length;
        const added = state.addDeformationLine(stroke.id, [
            { u: 0.12, v: 0.45 },
            { u: 0.88, v: 0.5 },
        ]);
        const refined = state.strokes[0];

        expect(added).toBe(true);
        expect(refined.deformationLines).toHaveLength(1);
        expect(refined.mesh.rows).toBe(rowsBefore);
        expect(refined.mesh.columns.length).toBeGreaterThan(columnsBefore);
        expect(refined.mesh.faces.length).toBeGreaterThan(facesBefore);
        expect(refined.mesh.vertices.length).toBe(rowsBefore * refined.mesh.columns.length);
        expect(meshConnectedComponentCount(refined.mesh)).toBe(1);
    });

    it("raycasts the evaluated ribbon surface", () => {
        const state = new PaintModelingState();
        state.addObject();
        drawStroke(state, { x: -0.2, y: 0 }, { x: 0.2, y: 0 });

        const hit = state.raycastStrokeAt({ x: 0, y: 0 }, state.activeView);

        expect(hit?.strokeId).toBe(state.strokes[0].id);
        expect(hit?.uv.u).toBeGreaterThan(0.3);
        expect(hit?.uv.u).toBeLessThan(0.7);
        expect(Math.abs(hit?.uv.v ?? 1)).toBeLessThan(0.2);
    });

    it("renders committed ribbons from the evaluated surface", () => {
        const state = new PaintModelingState();
        state.addObject();
        drawStroke(state, { x: -0.24, y: 0 }, { x: 0.24, y: 0 });

        const primitives = state.buildRenderSegments({ showDraftStroke: false });
        const triangles = primitives.filter(isRenderTriangle);
        const evaluated = evaluatedStrokeMesh(state.strokes[0]);

        expect(triangles.length).toBe(evaluated.faces.length * 2);
        expect(primitives.filter(isRenderStroke)).toHaveLength(0);
        expect(triangles.every(triangle => triangle.normal !== undefined)).toBe(true);
        expect(triangles.every(triangle => triangle.shade === 1)).toBe(true);
    });

    it("undoes a paint stroke together with auto-created object and view", () => {
        const state = new PaintModelingState();
        drawStroke(state, { x: -0.14, y: 0 }, { x: 0.14, y: 0 });

        expect(state.objects).toHaveLength(1);
        expect(state.views).toHaveLength(1);
        expect(state.strokes).toHaveLength(1);

        expect(state.undo()).toBe(true);

        expect(state.objects).toHaveLength(0);
        expect(state.views).toHaveLength(0);
        expect(state.strokes).toHaveLength(0);
        expect(state.activeObjectId).toBeNull();
        expect(state.activeViewId).toBeNull();
    });

    it("deletes objects and views with their stroke meshes", () => {
        const state = new PaintModelingState();
        state.addObject("First");
        const objectId = state.activeObjectId!;
        drawStroke(state, { x: -0.14, y: 0 }, { x: 0.14, y: 0 });
        const viewId = state.activeViewId!;

        expect(state.deleteView(viewId)).toBe(true);
        expect(state.views).toHaveLength(0);
        expect(state.strokes).toHaveLength(0);

        state.addObject("Second");
        drawStroke(state, { x: -0.1, y: 0 }, { x: 0.1, y: 0 });
        expect(state.deleteObject(objectId)).toBe(true);
        expect(state.objects.some(object => object.name === "First")).toBe(false);
        expect(state.strokes).toHaveLength(1);
        expect(state.deleteObject(state.activeObjectId!)).toBe(true);
        expect(state.objects.some(object => object.name === "Second")).toBe(false);
    });
});

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