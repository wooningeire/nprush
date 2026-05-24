import { describe, expect, it } from "vitest";
import { ContourModelerState } from "./ContourModelerState.svelte.ts";

describe("ContourModelerState", () => {
    it("captures a view, records a stroke, and supports undo", () => {
        const state = new ContourModelerState();
        state.addShape();
        expect(state.activeShape?.mesh).toBeNull();
        state.beginStroke({ x: -0.3, y: 0.2 }, 800, 600);
        state.appendStrokePoint({ x: 0.3, y: -0.2 });
        state.finishStroke();

        expect(state.shapes.length).toBe(1);
        expect(state.views.length).toBe(1);
        expect(state.strokes.length).toBe(1);
        expect(state.strokes[0].kind).toBe("edge");

        state.undoStroke();

        expect(state.strokes.length).toBe(0);
        expect(state.activeShape?.strokeIds.length).toBe(0);
    });

    it("clears active shape contours and leaves the shape empty until fit", () => {
        const state = new ContourModelerState();
        state.addShape();
        state.beginStroke({ x: -0.3, y: 0.2 }, 800, 600);
        state.appendStrokePoint({ x: 0.3, y: -0.2 });
        state.finishStroke();

        state.clearActiveShape();

        expect(state.strokes.length).toBe(0);
        expect(state.activeShape?.mesh).toBeNull();
        expect(state.activeShape?.fitStatus).toBe("idle");
    });

    it("turns saved-view strokes into guides after the camera moves", () => {
        const state = new ContourModelerState();
        state.addShape();
        state.beginStroke({ x: -0.3, y: 0.2 }, 800, 600);
        state.appendStrokePoint({ x: 0.3, y: -0.2 });
        state.finishStroke();

        const viewId = state.activeViewId;
        expect(state.visibleStrokes.length).toBe(1);
        expect(state.guideStrokes.length).toBe(0);

        state.orbit.turn({ x: 40, y: 0 });

        expect(state.currentViewName).toBe("New view");
        expect(state.visibleStrokes.length).toBe(0);
        expect(state.guideStrokes.length).toBe(1);

        state.selectView(viewId!);

        expect(state.visibleStrokes.length).toBe(1);
        expect(state.guideStrokes.length).toBe(0);
    });

    it("stores edited guide depth as a fit constraint", () => {
        const state = new ContourModelerState();
        state.addShape();
        state.beginStroke({ x: -0.3, y: 0.2 }, 800, 600);
        state.appendStrokePoint({ x: 0.3, y: -0.2 });
        state.finishStroke();
        const strokeId = state.strokes[0].id;

        state.orbit.turn({ x: 40, y: 0 });
        state.selectDepthStroke(strokeId);
        state.setStrokeDepth(strokeId, 0.72);

        expect(state.activeDepthStrokeId).toBe(strokeId);
        expect(state.strokes[0].depthLocked).toBe(true);
        expect(state.strokes[0].depthNdc).toBeCloseTo(0.72);

        state.resetActiveDepth();

        expect(state.strokes[0].depthLocked).toBe(false);

        state.beginStroke({ x: 0.1, y: 0.2 }, 800, 600);
        state.appendStrokePoint({ x: 0.2, y: 0.1 });
        state.finishStroke();

        expect(state.views.length).toBe(2);
        expect(state.depthEditableStrokes.some(stroke => stroke.id === strokeId)).toBe(true);
    });

    it("brushes depth samples on drawn stroke vertices", () => {
        const state = new ContourModelerState();
        state.addShape();
        state.beginStroke({ x: -0.3, y: 0.2 }, 800, 600);
        state.appendStrokePoint({ x: 0, y: 0 });
        state.appendStrokePoint({ x: 0.3, y: -0.2 });
        state.finishStroke();
        const strokeId = state.strokes[0].id;

        state.brushStrokeDepth([
            { strokeId, pointIndex: 0, influence: 1 },
            { strokeId, pointIndex: 1, influence: 0.5 },
        ], -0.12);

        expect(state.strokes[0].depthLocked).toBe(true);
        expect(state.strokes[0].depthSamplesLocked?.[0]).toBe(true);
        expect(state.strokes[0].depthSamplesLocked?.[1]).toBe(true);
        expect(state.strokes[0].depthSamplesOffset?.[0]).toBeLessThan(state.strokes[0].depthSamplesOffset?.[1] ?? 0);

        state.orbit.turn({ x: 40, y: 0 });
        state.resetGuideDepths();
        expect(state.strokes[0].depthSamplesOffset).toBeUndefined();
    });
});
