import { describe, expect, it } from "vitest";
import { PaintModelingState } from "./PaintModelingState.svelte.ts";
import { sampleChartDepth } from "./state/chartPainting.ts";
import type { SurfaceRef, Vec2, Vec3 } from "./types.ts";

const drawStroke = (state: PaintModelingState, a: Vec2, b: Vec2) => {
    state.beginStroke(a, 800, 600);
    state.appendStrokePoint(b);
    state.finishStroke();
};

describe("PaintModelingState depth brush", () => {
    it("defaults depth mode to a sculpt-width brush and remembers per-mode widths", () => {
        const state = new PaintModelingState();

        expect(state.brush.width).toBe(18);

        state.setBrushMode("depth");
        expect(state.brush.width).toBe(36);

        state.setBrushWidth(28);
        state.setBrushMode("surface");
        expect(state.brush.width).toBe(72);

        state.setBrushWidth(64);
        state.setBrushMode("color");
        expect(state.brush.width).toBe(18);

        state.setBrushMode("depth");
        expect(state.brush.width).toBe(28);

        state.setBrushMode("surface");
        expect(state.brush.width).toBe(64);
    });

    it("sculpts covered chart depth without creating paint strokes or coverage", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.18, y: 0 }, { x: 0.18, y: 0 });

        const chart = state.activeObject!.charts[0];
        const coverageBefore = [...chart.coverage];
        const depthsBefore = [...chart.depths];
        const chartCount = state.chartCount;
        const strokeCount = state.strokes.length;

        state.setBrushMode("depth");
        drawStroke(state, { x: -0.06, y: 0 }, { x: 0.06, y: 0 });

        const sculptedChart = state.activeObject!.charts[0];
        const coveredDepthDeltas = sculptedChart.depths
            .map((depth, index) => coverageBefore[index] > 0.015 ? depth - depthsBefore[index] : 0)
            .filter(delta => Math.abs(delta) > 1e-5);
        const uncoveredDepthDeltas = sculptedChart.depths
            .map((depth, index) => coverageBefore[index] <= 0.015 ? depth - depthsBefore[index] : 0)
            .filter(delta => Math.abs(delta) > 1e-5);

        expect(state.chartCount).toBe(chartCount);
        expect(state.strokes).toHaveLength(strokeCount);
        expect(sculptedChart.coverage).toEqual(coverageBefore);
        expect(coveredDepthDeltas.length).toBeGreaterThan(0);
        expect(Math.min(...coveredDepthDeltas)).toBeLessThan(-0.001);
        expect(uncoveredDepthDeltas).toHaveLength(0);
    });

    it("keeps source-view stroke projection stable while depth changes", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.18, y: 0 }, { x: 0.18, y: 0 });

        const sourceView = state.activeView!;
        const sample = middleSample(state.strokes[0].samples);
        const sourceProjectionBefore = state.projectSurfaceRef(sample.surfaceRef, sourceView)!;
        const worldBefore = state.surfaceRefWorldPoint(sample.surfaceRef)!;

        state.setBrushMode("depth");
        drawStroke(state, { x: sample.sourcePoint.x - 0.03, y: sample.sourcePoint.y }, {
            x: sample.sourcePoint.x + 0.03,
            y: sample.sourcePoint.y,
        });

        const sourceProjectionAfter = state.projectSurfaceRef(sample.surfaceRef, sourceView)!;
        const worldAfter = state.surfaceRefWorldPoint(sample.surfaceRef)!;

        expect(distance2d(sourceProjectionBefore, sample.sourcePoint)).toBeLessThan(1e-5);
        expect(distance2d(sourceProjectionAfter, sample.sourcePoint)).toBeLessThan(1e-5);
        expect(distance3(worldBefore, worldAfter)).toBeGreaterThan(0.001);
        expect(state.strokes).toHaveLength(1);
    });

    it("sculpts an existing source chart from a later view without creating a chart", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.18, y: 0 }, { x: 0.18, y: 0 });

        const sourceChart = state.activeObject!.charts[0];
        const surfaceRef: SurfaceRef = { chartId: sourceChart.id, uv: { x: 0, y: 0 } };
        const depthBefore = sampleChartDepth(sourceChart, surfaceRef.uv);

        state.orbit.turn({ x: 42, y: 0 });
        state.ensureActiveView(800, 600);
        const laterPoint = state.projectSurfaceRef(surfaceRef, state.activeView)!;

        state.setBrushMode("depth");
        drawStroke(state, { x: laterPoint.x - 0.025, y: laterPoint.y }, {
            x: laterPoint.x + 0.025,
            y: laterPoint.y,
        });

        const sculptedChart = state.activeObject!.charts.find(chart => chart.id === sourceChart.id)!;
        const depthAfter = sampleChartDepth(sculptedChart, surfaceRef.uv);

        expect(state.chartCount).toBe(1);
        expect(state.strokes).toHaveLength(1);
        expect(Math.abs(depthAfter - depthBefore)).toBeGreaterThan(0.001);
    });
});

const middleSample = <T>(samples: T[]): T => samples[Math.floor(samples.length / 2)];

const distance2d = (a: Vec2, b: Vec2): number => Math.hypot(a.x - b.x, a.y - b.y);

const distance3 = (a: Vec3, b: Vec3): number => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
