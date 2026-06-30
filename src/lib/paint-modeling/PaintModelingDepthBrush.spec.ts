import { describe, expect, it } from "vitest";
import { PaintModelingState } from "./PaintModelingState.svelte.ts";
import { meshConnectedComponentCount } from "./state/strokeMesh.ts";
import type { Vec2, Vec3 } from "./types.ts";

const drawStroke = (state: PaintModelingState, a: Vec2, b: Vec2) => {
    state.beginStroke(a, 800, 600);
    state.appendStrokePoint(b);
    state.finishStroke();
};

describe("PaintModelingState ribbon deformation", () => {
    it("supports direct sculpt edits without authored chart views", () => {
        const state = new PaintModelingState();
        state.addObject();
        drawStroke(state, { x: -0.18, y: 0 }, { x: 0.18, y: 0 });

        const stroke = state.strokes[0];
        const viewCount = state.views.length;
        const positionsBefore = stroke.mesh.vertices.map(vertex => [...vertex.position] as Vec3);

        state.orbit.turn({ x: 42, y: 0 });
        expect(state.isCameraAtActiveView).toBe(false);

        expect(state.sculptStrokeAt(stroke.id, { u: 0.5, v: 0 }, [0, 0, 0.08], 0.22)).toBe(true);

        const sculpted = state.strokes[0];
        const movement = sculpted.mesh.vertices.map((vertex, index) => distance3(vertex.position, positionsBefore[index]));

        expect(state.views).toHaveLength(viewCount);
        expect(state.strokes).toHaveLength(1);
        expect(meshConnectedComponentCount(sculpted.mesh)).toBe(1);
        expect(Math.max(...movement)).toBeGreaterThan(0.01);
    });

    it("keeps deformation lines through undo and redo-style restore", () => {
        const state = new PaintModelingState();
        state.addObject();
        drawStroke(state, { x: -0.18, y: 0 }, { x: 0.18, y: 0 });
        const strokeId = state.strokes[0].id;

        state.addDeformationLine(strokeId, [
            { u: 0.2, v: -0.35 },
            { u: 0.8, v: -0.32 },
        ]);

        expect(state.strokes[0].deformationLines).toHaveLength(1);
        expect(state.undo()).toBe(true);
        expect(state.strokes[0].deformationLines).toHaveLength(0);
    });
});

const distance3 = (a: Vec3, b: Vec3): number => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);