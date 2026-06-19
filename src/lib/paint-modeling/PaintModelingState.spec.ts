import { describe, expect, it } from "vitest";
import { PaintModelingState } from "./PaintModelingState.svelte.ts";
import type { RenderPrimitive, RenderSegment, RenderStroke, RenderTriangle, Vec2, Vec3 } from "./types.ts";

function drawStroke(state: PaintModelingState, a: Vec2, b: Vec2) {
    state.beginStroke(a, 800, 600);
    state.appendStrokePoint(b);
    state.finishStroke();
}

describe("PaintModelingState prototype", () => {
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

        const wireOff = state.buildRenderSegments({ showChartWireframe: false }).filter(isRenderSegment);
        const wireOn = state.buildRenderSegments({ showChartWireframe: true }).filter(isRenderSegment);

        expect(wireOff.some(segment => segment.width === 1.15)).toBe(false);
        expect(wireOn.some(segment => segment.width === 1.15)).toBe(true);
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

function cameraCenter(viewInvMat: number[]): Vec3 {
    return [viewInvMat[12], viewInvMat[13], viewInvMat[14]];
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
