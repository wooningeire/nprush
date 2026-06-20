import { describe, expect, it } from "vitest";
import { PaintModelingState } from "./PaintModelingState.svelte.ts";
import type {
    PaintSample,
    PaintView,
    RenderPrimitive,
    RenderSegment,
    RenderStroke,
    RenderTriangle,
    Vec2,
    Vec3,
} from "./types.ts";

function drawStroke(state: PaintModelingState, a: Vec2, b: Vec2) {
    state.beginStroke(a, 800, 600);
    state.appendStrokePoint(b);
    state.finishStroke();
}

describe("PaintModelingState prototype", () => {
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

    it("defaults surface mode to a wider brush and remembers per-mode widths", () => {
        const state = new PaintModelingState();

        expect(state.brush.width).toBe(18);

        state.setBrushMode("surface");
        expect(state.brush.width).toBe(72);

        state.setBrushWidth(48);
        state.setBrushMode("color");
        expect(state.brush.width).toBe(18);

        state.setBrushWidth(12);
        state.setBrushMode("surface");
        expect(state.brush.width).toBe(48);

        state.setBrushMode("color");
        expect(state.brush.width).toBe(12);
    });

    it("can snap a later-view stroke to an existing surface chart", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.18, y: 0 }, { x: 0.18, y: 0 });

        const originalRef = state.strokes[0].samples[Math.floor(state.strokes[0].samples.length / 2)].surfaceRef;

        state.orbit.turn({ x: 42, y: 0 });
        state.ensureActiveView(800, 600);
        const laterPoint = state.projectSurfaceRef(originalRef, state.activeView)!;

        state.setPlacementMode("snap");
        drawStroke(state, { x: laterPoint.x - 0.025, y: laterPoint.y }, { x: laterPoint.x + 0.025, y: laterPoint.y });

        const snappedStroke = state.strokes.at(-1)!;
        expect(snappedStroke.placement).toBe("snap");
        expect(snappedStroke.samples.some(sample => sample.surfaceRef.chartId === originalRef.chartId)).toBe(true);
        expect(state.activeObject?.charts.length).toBe(1);
    });

    it("carries last surface depth when later-view snap strokes leave coverage", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.56, y: 0.18 }, { x: -0.28, y: 0.18 });

        const sourceStroke = state.strokes[0];
        const sourceChart = state.activeObject!.charts[0];
        const sourceViewId = state.activeViewId!;
        const sourceRef = sourceStroke.samples[Math.floor(sourceStroke.samples.length / 2)].surfaceRef;

        state.orbit.turn({ x: 58, y: 10 });
        state.ensureActiveView(800, 600);
        const laterView = state.activeView!;
        const laterPoint = state.projectSurfaceRef(sourceRef, laterView)!;
        const strokeEnd = {
            x: Math.min(0.92, laterPoint.x + 0.56),
            y: Math.max(-0.92, Math.min(0.92, laterPoint.y + 0.04)),
        };

        state.setPlacementMode("snap");
        drawStroke(state, laterPoint, strokeEnd);

        const laterStroke = state.strokes.at(-1)!;
        const fallbackSampleIndex = laterStroke.samples.findIndex(sample => sample.surfaceRef.chartId !== sourceChart.id);
        const lastHitBeforeFallback = laterStroke.samples
            .slice(0, fallbackSampleIndex)
            .findLast(sample => sample.surfaceRef.chartId === sourceChart.id)!;
        expect(laterView.id).not.toBe(sourceViewId);
        expect(fallbackSampleIndex).toBeGreaterThan(0);

        const fallbackSample = laterStroke.samples[fallbackSampleIndex];
        const fallbackChart = state.activeObject!.charts.find(chart => chart.id === fallbackSample.surfaceRef.chartId);
        const carriedDepth = strokeSampleRayDepthInView(state, laterView, lastHitBeforeFallback);
        const fallbackDepths = laterStroke.samples
            .slice(fallbackSampleIndex)
            .filter(sample => sample.surfaceRef.chartId !== sourceChart.id)
            .map(sample => strokeSampleRayDepthInView(state, laterView, sample));
        const depthErrors = fallbackDepths.map(depth => Math.abs(depth - carriedDepth));
        const depthSteps = fallbackDepths.slice(1).map((depth, index) => Math.abs(depth - fallbackDepths[index]));

        expect(fallbackChart?.sourceViewId).toBe(laterView.id);
        expect(fallbackDepths.length).toBeGreaterThan(6);
        expect(Math.max(...depthErrors)).toBeLessThan(0.025);
        expect(Math.max(...depthSteps)).toBeLessThan(0.025);
    });

    it("mixes snap fallback depth between exit and reentry hits", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setBrushWidth(42);
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.74, y: 0.16 }, { x: -0.48, y: 0.16 });
        drawStroke(state, { x: 0.22, y: -0.12 }, { x: 0.5, y: -0.12 });

        const sourceChart = state.activeObject!.charts[0];
        const exitRef = state.strokes[0].samples[Math.floor(state.strokes[0].samples.length / 2)].surfaceRef;
        const entryRef = state.strokes[1].samples[Math.floor(state.strokes[1].samples.length / 2)].surfaceRef;

        state.orbit.turn({ x: 62, y: 16 });
        state.ensureActiveView(800, 600);
        const laterView = state.activeView!;
        const exitPoint = state.projectSurfaceRef(exitRef, laterView)!;
        const entryPoint = state.projectSurfaceRef(entryRef, laterView)!;

        state.setBrushWidth(18);
        state.setPlacementMode("snap");
        drawStroke(state, exitPoint, entryPoint);

        const laterStroke = state.strokes.at(-1)!;
        const gapStart = laterStroke.samples.findIndex((sample, index) =>
            index > 0
            && sample.surfaceRef.chartId !== sourceChart.id
            && laterStroke.samples[index - 1].surfaceRef.chartId === sourceChart.id
        );
        const gapEnd = laterStroke.samples.findIndex((sample, index) =>
            gapStart >= 0
            && index > gapStart
            && sample.surfaceRef.chartId === sourceChart.id
        );

        expect(gapStart).toBeGreaterThan(0);
        expect(gapEnd).toBeGreaterThan(gapStart + 6);

        const exitSample = laterStroke.samples[gapStart - 1];
        const entrySample = laterStroke.samples[gapEnd];
        const exitDepth = strokeSampleRayDepthInView(state, laterView, exitSample);
        const entryDepth = strokeSampleRayDepthInView(state, laterView, entrySample);
        const gapSamples = laterStroke.samples.slice(gapStart, gapEnd);
        const depthErrors = gapSamples.map(sample => {
            const expectedDepth = lerpNumber(
                exitDepth,
                entryDepth,
                depthMixFactor(sample.sourcePoint, exitSample.sourcePoint, entrySample.sourcePoint),
            );
            return Math.abs(strokeSampleRayDepthInView(state, laterView, sample) - expectedDepth);
        });

        expect(Math.abs(entryDepth - exitDepth)).toBeGreaterThan(0.08);
        expect(Math.max(...depthErrors)).toBeLessThan(0.04);
    });

    it("backfills leading snap misses from the first surface depth", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.56, y: 0.18 }, { x: -0.28, y: 0.18 });

        const sourceStroke = state.strokes[0];
        const sourceChart = state.activeObject!.charts[0];
        const sourceRef = sourceStroke.samples[Math.floor(sourceStroke.samples.length / 2)].surfaceRef;

        state.orbit.turn({ x: 58, y: 10 });
        state.ensureActiveView(800, 600);
        const laterView = state.activeView!;
        const laterPoint = state.projectSurfaceRef(sourceRef, laterView)!;
        const strokeStart = {
            x: Math.max(-0.92, laterPoint.x - 0.56),
            y: Math.max(-0.92, Math.min(0.92, laterPoint.y + 0.04)),
        };

        state.setPlacementMode("snap");
        drawStroke(state, strokeStart, laterPoint);

        const laterStroke = state.strokes.at(-1)!;
        const firstHitIndex = laterStroke.samples.findIndex(sample => sample.surfaceRef.chartId === sourceChart.id);
        expect(firstHitIndex).toBeGreaterThan(0);

        const firstHitDepth = strokeSampleRayDepthInView(state, laterView, laterStroke.samples[firstHitIndex]);
        const leadingDepths = laterStroke.samples
            .slice(0, firstHitIndex)
            .filter(sample => sample.surfaceRef.chartId !== sourceChart.id)
            .map(sample => strokeSampleRayDepthInView(state, laterView, sample));

        const leadingDepthErrors = leadingDepths.map(depth => Math.abs(depth - firstHitDepth));

        expect(leadingDepths.length).toBeGreaterThan(3);
        expect(Math.max(...leadingDepthErrors)).toBeLessThan(0.025);
    });

    it("creates surface-only brush masks without visible paint strokes", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setBrushMode("surface");
        state.setBrushWidth(36);
        drawStroke(state, { x: -0.2, y: 0 }, { x: 0.2, y: 0 });

        const chart = state.activeObject!.charts[0];
        const noWirePrimitives = state.buildRenderSegments({ showChartWireframe: false });
        const wirePrimitives = state.buildRenderSegments({ showChartWireframe: true });

        expect(state.strokes).toHaveLength(0);
        expect(state.occlusionClaims).toHaveLength(0);
        expect(chart.role).toBe("surface");
        expect(chart.projectionMode).toBe("view-plane");
        const chartFill = wirePrimitives.filter(isRenderTriangle);

        expect(coveredTexelCount(state)).toBeGreaterThan(0);
        expect(noWirePrimitives.filter(isRenderStroke)).toHaveLength(0);
        expect(noWirePrimitives.filter(isRenderTriangle)).toHaveLength(0);
        expect(chartFill.length).toBeGreaterThan(0);
        expect(chartFill.every(triangle => triangle.color[3] > 0 && triangle.color[3] <= 0.1)).toBe(true);
        expect(wirePrimitives.filter(isRenderSegment).length).toBeGreaterThan(0);
    });

    it("undoes a surface brush mask together with auto-created object and view", () => {
        const state = new PaintModelingState();
        state.setBrushMode("surface");
        drawStroke(state, { x: -0.18, y: 0 }, { x: 0.18, y: 0 });

        expect(state.objects).toHaveLength(1);
        expect(state.views).toHaveLength(1);
        expect(state.chartCount).toBe(1);
        expect(state.strokes).toHaveLength(0);

        expect(state.undo()).toBe(true);

        expect(state.objects).toHaveLength(0);
        expect(state.views).toHaveLength(0);
        expect(state.chartCount).toBe(0);
        expect(state.activeObjectId).toBeNull();
        expect(state.activeViewId).toBeNull();
    });

    it("snaps later-view color strokes to surface-only masks", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setBrushMode("surface");
        drawStroke(state, { x: -0.22, y: 0 }, { x: 0.22, y: 0 });

        const surfaceChart = state.activeObject!.charts[0];
        const surfaceRef = { chartId: surfaceChart.id, uv: { x: 0, y: 0 } };

        state.orbit.turn({ x: 42, y: 0 });
        state.ensureActiveView(800, 600);
        const laterPoint = state.projectSurfaceRef(surfaceRef, state.activeView)!;

        state.setBrushMode("color");
        drawStroke(state, { x: laterPoint.x - 0.025, y: laterPoint.y }, { x: laterPoint.x + 0.025, y: laterPoint.y });

        const snappedStroke = state.strokes.at(-1)!;
        expect(snappedStroke.placement).toBe("snap");
        expect(snappedStroke.samples.some(sample => sample.surfaceRef.chartId === surfaceChart.id)).toBe(true);
        expect(state.activeObject?.charts.length).toBe(1);
    });

    it("keeps color brush fallback geometry when no surface exists", () => {
        const state = new PaintModelingState();
        state.addObject();
        drawStroke(state, { x: -0.16, y: 0 }, { x: 0.16, y: 0 });

        const primitives = state.buildRenderSegments({ showChartWireframe: false });
        const brushStrokes = strokeRunsForWidth(primitives, 18);

        expect(state.placementMode).toBe("snap");
        expect(state.strokes).toHaveLength(1);
        expect(state.activeObject?.charts.length).toBe(1);
        expect(coveredTexelCount(state)).toBeGreaterThan(0);
        expect(strokeSegmentCount(brushStrokes)).toBeGreaterThan(0);
    });

    it("forces surface brush masks onto a flat view-plane chart", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setChartProjectionMode("ray-depth");
        state.setBrushMode("surface");
        drawStroke(state, { x: -0.18, y: 0 }, { x: 0.18, y: 0 });

        const chart = state.activeObject!.charts[0];
        const coveredDepths = chart.depths.filter((_, index) => chart.coverage[index] > 0.015);
        const firstDepth = coveredDepths[0];

        expect(chart.projectionMode).toBe("view-plane");
        expect(coveredDepths.length).toBeGreaterThan(0);
        for (const depth of coveredDepths) {
            expect(depth).toBeCloseTo(firstDepth, 10);
        }
    });

    it("creates a foreground chart and view-local ordering claim in occlude mode", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.12, y: 0 }, { x: 0.12, y: 0 });

        state.setPlacementMode("occluding-surface");
        drawStroke(state, { x: -0.08, y: 0 }, { x: 0.08, y: 0 });

        const claim = state.occlusionClaims[0];
        const frontRef = state.strokes.at(-1)!.samples[Math.floor(state.strokes.at(-1)!.samples.length / 2)].surfaceRef;
        const backRef = claim.backRefs[Math.floor(claim.backRefs.length / 2)];
        const view = state.views.find(item => item.id === claim.viewId)!;
        const camera = cameraCenter(view.viewInvMat);
        const front = state.surfaceRefWorldPoint(frontRef)!;
        const back = state.surfaceRefWorldPoint(backRef)!;

        expect(claim.frontChartId).toBe(frontRef.chartId);
        expect(claim.backRefs.length).toBeGreaterThan(0);
        expect(distance3(camera, front)).toBeLessThan(distance3(camera, back));
        expect(state.activeObject?.charts.length).toBe(2);
    });

    it("marks seams on hit chart samples", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.1, y: 0 }, { x: 0.1, y: 0 });

        expect(state.markSeamAt({ x: 0, y: 0 })).toBe(true);
        expect(state.seamCount).toBeGreaterThan(0);
    });

    it("does not dirty scene geometry when only the camera orbits", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.14, y: 0 }, { x: 0.14, y: 0 });
        const meshVersion = state.meshVersion;
        const sceneSegmentCount = state.buildRenderSegments(true).filter(isRenderSegment).length;

        state.orbit.turn({ x: 80, y: 16 });

        expect(state.meshVersion).toBe(meshVersion);
        expect(state.buildRenderSegments(true).filter(isRenderSegment).length).toBe(sceneSegmentCount);
    });

    it("undoes a paint stroke together with auto-created object and view", () => {
        const state = new PaintModelingState();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.14, y: 0 }, { x: 0.14, y: 0 });

        expect(state.objects.length).toBe(1);
        expect(state.views.length).toBe(1);
        expect(state.strokes.length).toBe(1);

        expect(state.undo()).toBe(true);

        expect(state.objects).toHaveLength(0);
        expect(state.views).toHaveLength(0);
        expect(state.strokes).toHaveLength(0);
        expect(state.activeObjectId).toBeNull();
        expect(state.activeViewId).toBeNull();
        expect(state.canUndo).toBe(false);
    });

    it("undoes seam edits, not only paint strokes", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.14, y: 0 }, { x: 0.14, y: 0 });
        const sample = state.strokes[0].samples[Math.floor(state.strokes[0].samples.length / 2)];

        expect(state.markSeamAt(sample.sourcePoint)).toBe(true);
        expect(state.seamCount).toBeGreaterThan(0);
        expect(state.undo()).toBe(true);
        expect(state.seamCount).toBe(0);
        expect(state.strokes).toHaveLength(1);
    });

    it("undoes object and view deletion", () => {
        const state = new PaintModelingState();
        state.addObject("First");
        const objectId = state.activeObjectId!;
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.12, y: 0 }, { x: 0.12, y: 0 });
        const viewId = state.activeViewId!;

        expect(state.deleteObject(objectId)).toBe(true);
        expect(state.objects).toHaveLength(0);
        expect(state.undo()).toBe(true);
        expect(state.objects.map(object => object.id)).toEqual([objectId]);
        expect(state.strokes).toHaveLength(1);

        expect(state.deleteView(viewId)).toBe(true);
        expect(state.views).toHaveLength(0);
        expect(state.undo()).toBe(true);
        expect(state.views.map(view => view.id)).toEqual([viewId]);
        expect(state.strokes).toHaveLength(1);
    });

    it("does not raycast while committing a new-surface paint stroke", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.18, y: -0.1 }, { x: 0.18, y: -0.1 });

        state.resetDiagnostics();
        drawStroke(state, { x: -0.18, y: 0.1 }, { x: 0.18, y: 0.1 });

        expect(state.raycastCountForDiagnostics).toBe(0);
        expect(state.activeObject?.charts.length).toBe(1);
    });

    it("still raycasts snap strokes against existing surfaces", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.16, y: 0 }, { x: 0.16, y: 0 });

        state.resetDiagnostics();
        state.setPlacementMode("snap");
        drawStroke(state, { x: -0.12, y: 0 }, { x: 0.12, y: 0 });

        expect(state.raycastCountForDiagnostics).toBeGreaterThan(0);
        expect(state.strokes.at(-1)?.placement).toBe("snap");
    });

    it("does not treat unpainted chart area as a seam surface", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.72, y: -0.52 }, { x: -0.48, y: -0.52 });

        expect(state.markSeamAt({ x: 0.62, y: 0.54 })).toBe(false);
    });

    it("renders committed paint as continuous brush segments plus bounded chart overlays", () => {
        const emptyOverlayCount = new PaintModelingState().buildRenderSegments(true).filter(isRenderSegment).length;
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        state.setBrushWidth(18);
        drawStroke(state, { x: -0.12, y: 0 }, { x: 0.12, y: 0 });

        const primitives = state.buildRenderSegments(true);
        const renderSegments = primitives.filter(isRenderSegment);
        const chartOverlaySegments = renderSegments.filter(segment => segment.width === 1.15).length;
        const brushStrokes = strokeRunsForWidth(primitives, 18);

        expect(strokeSegmentCount(brushStrokes)).toBeGreaterThanOrEqual(state.strokes[0].samples.length - 1);
        expect(chartOverlaySegments).toBeGreaterThan(0);
        expect(chartOverlaySegments).toBeLessThan(520);
        expect(renderSegments.length).toBeGreaterThan(emptyOverlayCount);
    });

    it("renders committed paint as vector strokes by default", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        state.setBrushWidth(48);
        drawStroke(state, { x: -0.18, y: 0 }, { x: 0.18, y: 0 });

        const primitives = state.buildRenderSegments({ showChartWireframe: false });
        const brushStrokes = strokeRunsForWidth(primitives, 48);

        expect(primitives.filter(isRenderTriangle)).toHaveLength(0);
        expect(strokeSegmentCount(brushStrokes)).toBeGreaterThanOrEqual(state.strokes[0].samples.length - 1);
    });

    it("renders an in-progress stroke through preview segments in surface mode", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        state.setBrushWidth(36);

        state.beginStroke({ x: -0.16, y: -0.04 }, 800, 600);
        state.appendStrokePoint({ x: 0.16, y: 0.04 });

        const draftPrimitives = state.buildRenderSegments({ showChartWireframe: false });
        const draftBrushStrokes = strokeRunsForWidth(draftPrimitives, 36);

        expect(strokeSegmentCount(draftBrushStrokes)).toBeGreaterThan(0);

        state.finishStroke();
        const committedPrimitives = state.buildRenderSegments({ showChartWireframe: false });
        const committedBrushStrokes = strokeRunsForWidth(committedPrimitives, 36);

        expect(strokeSegmentCount(committedBrushStrokes)).toBeGreaterThan(0);
        expect(committedPrimitives.filter(isRenderTriangle)).toHaveLength(0);
    });

    it("renders draft strokes with the same spline resolution used by committed strokes", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");

        state.beginStroke({ x: -0.46, y: -0.08 }, 800, 600);
        for (let i = 1; i <= 160; i++) {
            const t = i / 160;
            state.appendStrokePoint({
                x: -0.46 + t * 0.92,
                y: -0.08 + Math.sin(t * Math.PI * 8) * 0.11,
            });
        }

        const draftPrimitives = state.buildRenderSegments({ showChartWireframe: false });
        const draftBrushStrokes = strokeRunsForWidth(draftPrimitives, 18);
        const rawPointCount = state.draftStroke?.length ?? 0;

        state.finishStroke();
        const stroke = state.strokes.at(-1);
        const committedPrimitives = state.buildRenderSegments({ showChartWireframe: false });
        const committedBrushStrokes = strokeRunsForWidth(committedPrimitives, 18);

        expect(rawPointCount).toBeGreaterThan(100);
        expect(stroke).toBeDefined();
        expect(draftBrushStrokes).toHaveLength(1);
        expect(committedBrushStrokes).toHaveLength(1);
        expect(draftBrushStrokes[0].points).toHaveLength(stroke!.samples.length);
        expect(committedBrushStrokes[0].points).toHaveLength(draftBrushStrokes[0].points.length);
        expect(stroke!.samples.length).toBeGreaterThan(112);

        for (let i = 0; i < draftBrushStrokes[0].points.length; i++) {
            expect(distance3(draftBrushStrokes[0].points[i], committedBrushStrokes[0].points[i])).toBeLessThan(1e-6);
        }
    });

    it("keeps draft stroke sample density stable as strokes get longer", () => {
        const short = new PaintModelingState();
        short.addObject();
        short.setPlacementMode("new-surface");
        short.beginStroke({ x: -0.12, y: 0 }, 800, 600);
        short.appendStrokePoint({ x: 0.12, y: 0 });
        const shortSegments = strokeSegmentCount(strokeRunsForWidth(
            short.buildRenderSegments({ showChartWireframe: false }),
            18,
        ));

        const long = new PaintModelingState();
        long.addObject();
        long.setPlacementMode("new-surface");
        long.beginStroke({ x: -0.72, y: 0 }, 800, 600);
        long.appendStrokePoint({ x: 0.72, y: 0 });
        const longSegments = strokeSegmentCount(strokeRunsForWidth(
            long.buildRenderSegments({ showChartWireframe: false }),
            18,
        ));

        expect(shortSegments).toBeGreaterThan(8);
        expect(longSegments).toBeGreaterThan(shortSegments * 4);
        expect(longSegments / shortSegments).toBeLessThan(8);
    });

    it("keeps the already-drawn draft stroke prefix stable when the stroke grows", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");

        state.beginStroke({ x: -0.62, y: -0.08 }, 800, 600);
        for (let i = 1; i <= 80; i++) {
            const t = i / 160;
            state.appendStrokePoint({
                x: -0.62 + t * 1.24,
                y: -0.08 + Math.sin(t * Math.PI * 8) * 0.12,
            });
        }

        const prefixStroke = strokeRunsForWidth(
            state.buildRenderSegments({ showChartWireframe: false }),
            18,
        )[0];

        for (let i = 81; i <= 160; i++) {
            const t = i / 160;
            state.appendStrokePoint({
                x: -0.62 + t * 1.24,
                y: -0.08 + Math.sin(t * Math.PI * 8) * 0.12,
            });
        }

        const extendedStroke = strokeRunsForWidth(
            state.buildRenderSegments({ showChartWireframe: false }),
            18,
        )[0];
        const stablePrefixCount = Math.max(8, prefixStroke.points.length - 16);

        expect(prefixStroke.points.length).toBeGreaterThan(32);
        expect(extendedStroke.points.length).toBeGreaterThan(prefixStroke.points.length);
        for (let i = 0; i < stablePrefixCount; i++) {
            expect(distance3(prefixStroke.points[i], extendedStroke.points[i])).toBeLessThan(1e-6);
        }
    });

    it("keeps snap draft preview raycasts bounded after the first surface stroke", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.4, y: -0.1 }, { x: 0.4, y: 0.1 });
        state.setPlacementMode("snap");
        state.raycastCountForDiagnostics = 0;

        state.beginStroke({ x: -0.58, y: 0.18 }, 800, 600);
        for (let i = 1; i <= 180; i++) {
            const t = i / 180;
            state.appendStrokePoint({
                x: -0.58 + t * 1.16,
                y: 0.18 + Math.sin(t * Math.PI * 10) * 0.16,
            });
        }

        const draftStrokes = state.buildDraftRenderSegments().filter(isRenderStroke);

        expect(strokeSegmentCount(draftStrokes)).toBeGreaterThan(48);
        expect(state.raycastCountForDiagnostics).toBeLessThanOrEqual(1);
    });

    it("keeps committed scene primitives stable while drawing a later draft stroke", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.18, y: 0 }, { x: 0.18, y: 0 });

        const staticBefore = state.buildRenderSegments({
            showChartWireframe: false,
            showDraftStroke: false,
        }).filter(isRenderStroke);

        state.beginStroke({ x: -0.32, y: 0.08 }, 800, 600);
        for (let i = 1; i <= 96; i++) {
            const t = i / 96;
            state.appendStrokePoint({
                x: -0.32 + t * 0.64,
                y: 0.08 + Math.sin(t * Math.PI * 4) * 0.08,
            });
        }

        const staticDuring = state.buildRenderSegments({
            showChartWireframe: false,
            showDraftStroke: false,
        }).filter(isRenderStroke);
        const draftStrokes = state.buildDraftRenderSegments().filter(isRenderStroke);

        expect(staticBefore.length).toBeGreaterThan(0);
        expect(strokeSegmentCount(draftStrokes)).toBeGreaterThan(0);
        expectRenderStrokesToMatch(staticDuring, staticBefore);
    });

    it("can render chart wire independently from committed brush strokes", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        state.setBrushWidth(48);
        drawStroke(state, { x: -0.2, y: 0 }, { x: 0.2, y: 0 });

        const wireOffPrimitives = state.buildRenderSegments({ showChartWireframe: false });
        const wireOnPrimitives = state.buildRenderSegments({ showChartWireframe: true });
        const wireOff = wireOffPrimitives.filter(isRenderSegment);
        const wireOn = wireOnPrimitives.filter(isRenderSegment);
        const chartFill = wireOnPrimitives.filter(isRenderTriangle);

        expect(wireOff.some(segment => segment.width === 1.15)).toBe(false);
        expect(wireOffPrimitives.filter(isRenderTriangle)).toHaveLength(0);
        expect(wireOn.some(segment => segment.width === 1.15)).toBe(true);
        expect(chartFill.length).toBeGreaterThan(0);
        expect(chartFill.every(triangle => triangle.color[3] > 0 && triangle.color[3] <= 0.1)).toBe(true);
    });

    it("renders a surface normal field without requiring depth preview", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.2, y: 0 }, { x: 0.2, y: 0 });

        const fieldOff = state.buildRenderSegments({
            showChartWireframe: false,
            showSurfaceField: false,
        }).filter(isRenderSegment);
        const fieldOn = state.buildRenderSegments({
            showChartWireframe: false,
            showSurfaceField: true,
        }).filter(isRenderSegment);
        const fieldSegments = fieldOn.filter(segment => segment.width === 1.6);

        expect(fieldSegments.length).toBeGreaterThan(0);
        expect(fieldOn.length).toBeGreaterThan(fieldOff.length);
        expect(fieldOn.some(segment => segment.width === 1.15)).toBe(false);
    });

    it("creates covered surface charts for each painted view", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");

        for (let i = 0; i < 4; i++) {
            drawStroke(state, { x: -0.16, y: 0 }, { x: 0.16, y: 0 });
            state.orbit.turn({ x: 58, y: 6 });
        }

        const surfaceCharts = state.activeObject!.charts.filter(chart => chart.role === "surface");

        expect(state.views.length).toBe(4);
        expect(surfaceCharts.length).toBe(4);
        expect(new Set(surfaceCharts.map(chart => chart.sourceViewId)).size).toBe(4);
        expect(surfaceCharts.every(chart => chart.coverage.some(value => value > 0))).toBe(true);
    });

    it("can create view-plane or ray-depth charts for new paint surfaces", () => {
        const plane = new PaintModelingState();
        plane.addObject();
        plane.setPlacementMode("new-surface");
        plane.setChartProjectionMode("view-plane");
        drawStroke(plane, { x: -0.34, y: 0 }, { x: 0.34, y: 0 });

        const planeChart = plane.activeObject!.charts[0];
        const planeView = plane.views.find(view => view.id === planeChart.sourceViewId)!;
        const planeCamera = cameraCenter(planeView.viewInvMat);
        const planeCenter = plane.surfaceRefWorldPoint({ chartId: planeChart.id, uv: { x: 0, y: 0 } })!;
        const planeNormal = normalize3(sub3(planeCenter, planeCamera));
        const planeLeft = plane.surfaceRefWorldPoint({ chartId: planeChart.id, uv: { x: -0.34, y: 0 } })!;
        const planeRight = plane.surfaceRefWorldPoint({ chartId: planeChart.id, uv: { x: 0.34, y: 0 } })!;
        const planeDepth = dot3(sub3(planeCenter, planeCamera), planeNormal);

        const ray = new PaintModelingState();
        ray.addObject();
        ray.setPlacementMode("new-surface");
        ray.setChartProjectionMode("ray-depth");
        drawStroke(ray, { x: -0.34, y: 0 }, { x: 0.34, y: 0 });

        const rayChart = ray.activeObject!.charts[0];
        const rayView = ray.views.find(view => view.id === rayChart.sourceViewId)!;
        const rayCamera = cameraCenter(rayView.viewInvMat);
        const rayLeft = ray.surfaceRefWorldPoint({ chartId: rayChart.id, uv: { x: -0.34, y: 0 } })!;
        const rayRight = ray.surfaceRefWorldPoint({ chartId: rayChart.id, uv: { x: 0.34, y: 0 } })!;

        expect(planeChart.projectionMode).toBe("view-plane");
        expect(dot3(sub3(planeLeft, planeCamera), planeNormal)).toBeCloseTo(planeDepth, 5);
        expect(dot3(sub3(planeRight, planeCamera), planeNormal)).toBeCloseTo(planeDepth, 5);
        expect(rayChart.projectionMode).toBe("ray-depth");
        expect(distance3(rayLeft, rayCamera)).toBeCloseTo(distance3(rayRight, rayCamera), 5);
    });

    it("keeps painterly brush width and fixed opacity in vector strokes", () => {
        const narrow = new PaintModelingState();
        narrow.addObject();
        narrow.setPlacementMode("new-surface");
        narrow.setBrushWidth(8);
        drawStroke(narrow, { x: -0.12, y: 0 }, { x: 0.12, y: 0 });

        const wide = new PaintModelingState();
        wide.addObject();
        wide.setPlacementMode("new-surface");
        wide.setBrushWidth(48);
        wide.setBrushOpacity(0.42);
        drawStroke(wide, { x: -0.12, y: 0 }, { x: 0.12, y: 0 });

        const narrowCovered = coveredTexelCount(narrow);
        const wideCovered = coveredTexelCount(wide);
        const primitives = wide.buildRenderSegments(false);
        const wideBrushStrokes = strokeRunsForWidth(primitives, 48);

        expect(wideCovered).toBeGreaterThan(narrowCovered * 2);
        expect(strokeSegmentCount(wideBrushStrokes)).toBeGreaterThan(0);
        expect(wideBrushStrokes[0].color[3]).toBe(1);
        expect(primitives.filter(isRenderTriangle)).toHaveLength(0);
    });

    it("emits brush paths as single stroke runs with fixed opacity", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        state.setBrushWidth(48);
        state.setBrushOpacity(0.42);

        state.beginStroke({ x: -0.44, y: -0.08 }, 800, 600);
        for (let i = 1; i <= 80; i++) {
            const t = i / 80;
            state.appendStrokePoint({
                x: -0.44 + t * 0.88,
                y: -0.08 + Math.sin(t * Math.PI * 3) * 0.12,
            });
        }
        state.finishStroke();

        const brushStrokes = strokeRunsForWidth(
            state.buildRenderSegments({ showChartWireframe: false }),
            48,
        );

        expect(brushStrokes).toHaveLength(1);
        expect(brushStrokes[0].points.length).toBeGreaterThan(8);
        expect(brushStrokes[0].points.length).toBe(state.strokes[0].samples.length);
        expect(brushStrokes[0].color[3]).toBe(1);
    });

    it("keeps thin committed strokes visible even when chart fill triangles are sparse", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        state.setBrushWidth(3);
        drawStroke(state, { x: -0.32, y: -0.2 }, { x: 0.32, y: 0.2 });

        const primitives = state.buildRenderSegments(false);
        const brushStrokes = strokeRunsForWidth(primitives, 3);

        expect(state.strokes).toHaveLength(1);
        expect(strokeSegmentCount(brushStrokes)).toBeGreaterThan(12);
        expect(primitives.filter(isRenderTriangle)).toHaveLength(0);
    });

    it("keeps viewport guides out of scene geometry", () => {
        const state = new PaintModelingState();
        const segments = state.buildRenderSegments(false);
        const renderSegments = segments.filter(isRenderSegment);

        expect(renderSegments).toHaveLength(0);
    });

    it("deletes an object and removes its strokes, charts, and occlusion claims", () => {
        const state = new PaintModelingState();
        state.addObject("First");
        const firstId = state.activeObjectId!;
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.12, y: 0 }, { x: 0.12, y: 0 });
        state.setPlacementMode("occluding-surface");
        drawStroke(state, { x: -0.08, y: 0 }, { x: 0.08, y: 0 });

        state.addObject("Second");
        const secondId = state.activeObjectId!;
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.2, y: 0.18 }, { x: 0.2, y: 0.18 });

        expect(state.deleteObject(firstId)).toBe(true);

        expect(state.objects.map(object => object.id)).toEqual([secondId]);
        expect(state.activeObjectId).toBe(secondId);
        expect(state.strokes.every(stroke => stroke.objectId === secondId)).toBe(true);
        expect(state.occlusionClaims.every(claim => claim.objectId !== firstId)).toBe(true);
    });

    it("deletes a view and removes charts plus dependent strokes and claims", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.18, y: 0 }, { x: 0.18, y: 0 });
        const firstViewId = state.activeViewId!;
        const firstChartId = state.activeObject!.charts[0].id;
        const firstRef = state.strokes[0].samples[Math.floor(state.strokes[0].samples.length / 2)].surfaceRef;

        state.orbit.turn({ x: 42, y: 0 });
        state.ensureActiveView(800, 600);
        const secondViewId = state.activeViewId!;
        const snapPoint = state.projectSurfaceRef(firstRef, state.activeView)!;
        state.setPlacementMode("snap");
        drawStroke(state, { x: snapPoint.x - 0.025, y: snapPoint.y }, { x: snapPoint.x + 0.025, y: snapPoint.y });
        state.setPlacementMode("occluding-surface");
        drawStroke(state, { x: snapPoint.x - 0.02, y: snapPoint.y }, { x: snapPoint.x + 0.02, y: snapPoint.y });

        expect(state.deleteView(firstViewId)).toBe(true);

        expect(state.views.map(view => view.id)).toEqual([secondViewId]);
        expect(state.activeViewId).toBe(secondViewId);
        expect(state.activeObject!.charts.some(chart => chart.id === firstChartId)).toBe(false);
        expect(state.strokes.every(stroke =>
            stroke.sourceViewId !== firstViewId
            && !stroke.samples.some(sample => sample.surfaceRef.chartId === firstChartId)
        )).toBe(true);
        expect(state.occlusionClaims.every(claim =>
            claim.viewId !== firstViewId
            && claim.frontChartId !== firstChartId
            && !claim.backRefs.some(ref => ref.chartId === firstChartId)
        )).toBe(true);
    });
});

function projectedSpanAspect(viewProjInvMat: number[]): number {
    const left = unprojectNdc(viewProjInvMat, -1, 0, 0.5);
    const right = unprojectNdc(viewProjInvMat, 1, 0, 0.5);
    const bottom = unprojectNdc(viewProjInvMat, 0, -1, 0.5);
    const top = unprojectNdc(viewProjInvMat, 0, 1, 0.5);

    return distance3(left, right) / distance3(bottom, top);
}

function unprojectNdc(viewProjInvMat: number[], x: number, y: number, z: number): Vec3 {
    const worldX = viewProjInvMat[0] * x + viewProjInvMat[4] * y + viewProjInvMat[8] * z + viewProjInvMat[12];
    const worldY = viewProjInvMat[1] * x + viewProjInvMat[5] * y + viewProjInvMat[9] * z + viewProjInvMat[13];
    const worldZ = viewProjInvMat[2] * x + viewProjInvMat[6] * y + viewProjInvMat[10] * z + viewProjInvMat[14];
    const worldW = viewProjInvMat[3] * x + viewProjInvMat[7] * y + viewProjInvMat[11] * z + viewProjInvMat[15];
    if (!Number.isFinite(worldW) || Math.abs(worldW) <= 1e-6) {
        throw new Error("Cannot unproject NDC point");
    }
    return [worldX / worldW, worldY / worldW, worldZ / worldW];
}

function cameraCenter(viewInvMat: number[]): Vec3 {
    return [viewInvMat[12], viewInvMat[13], viewInvMat[14]];
}

function rayDepthAtPoint(view: PaintView, world: Vec3, point: Vec2): number {
    const ray = viewRay(view, point);
    return dot3(sub3(world, ray.origin), ray.direction);
}

function strokeSampleRayDepthInView(
    state: PaintModelingState,
    view: PaintView,
    sample: PaintSample,
): number {
    return rayDepthAtPoint(
        view,
        state.surfaceRefWorldPoint(sample.surfaceRef)!,
        sample.sourcePoint,
    );
}

function viewRay(view: PaintView, point: Vec2): { origin: Vec3; direction: Vec3 } {
    const near = unprojectNdc(view.viewProjInvMat, point.x, point.y, 0.02);
    const far = unprojectNdc(view.viewProjInvMat, point.x, point.y, 0.98);
    return {
        origin: cameraCenter(view.viewInvMat),
        direction: normalize3(sub3(far, near)),
    };
}

function lerpNumber(a: number, b: number, t: number): number {
    return a + (b - a) * t;
}

function depthMixFactor(point: Vec2, exitPoint: Vec2, entryPoint: Vec2): number {
    const exitDistance = distance2d(point, exitPoint);
    const entryDistance = distance2d(point, entryPoint);
    const denominator = exitDistance + entryDistance;
    if (denominator <= 1e-8) return 0.5;
    return exitDistance / denominator;
}

function distance2d(a: Vec2, b: Vec2): number {
    return Math.hypot(a.x - b.x, a.y - b.y);
}

function sub3(a: Vec3, b: Vec3): Vec3 {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dot3(a: Vec3, b: Vec3): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function normalize3(v: Vec3): Vec3 {
    const length = Math.hypot(v[0], v[1], v[2]);
    return [v[0] / length, v[1] / length, v[2] / length];
}

function isRenderSegment(primitive: RenderPrimitive): primitive is RenderSegment {
    return primitive.kind !== "triangle" && primitive.kind !== "stroke";
}

function isRenderTriangle(primitive: RenderPrimitive): primitive is RenderTriangle {
    return primitive.kind === "triangle";
}

function isRenderStroke(primitive: RenderPrimitive): primitive is RenderStroke {
    return primitive.kind === "stroke";
}

function strokeRunsForWidth(primitives: RenderPrimitive[], width: number): RenderStroke[] {
    return primitives.filter(isRenderStroke).filter(stroke => stroke.width === width);
}

function strokeSegmentCount(strokes: RenderStroke[]): number {
    return strokes.reduce((count, stroke) => count + Math.max(0, stroke.points.length - 1), 0);
}

function coveredTexelCount(state: PaintModelingState): number {
    return state.activeObject!.charts[0].coverage.filter(value => value > 0.015).length;
}

function distance3(a: Vec3, b: Vec3): number {
    return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

const expectRenderStrokesToMatch = (actual: RenderStroke[], expected: RenderStroke[]) => {
    expect(actual).toHaveLength(expected.length);
    for (let i = 0; i < expected.length; i++) {
        expect(actual[i].width).toBe(expected[i].width);
        expect(actual[i].points).toHaveLength(expected[i].points.length);
        for (let point = 0; point < expected[i].points.length; point++) {
            expect(distance3(actual[i].points[point], expected[i].points[point])).toBeLessThan(1e-6);
        }
        for (let channel = 0; channel < expected[i].color.length; channel++) {
            expect(actual[i].color[channel]).toBeCloseTo(expected[i].color[channel], 6);
        }
    }
};
