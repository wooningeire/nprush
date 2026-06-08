import { describe, expect, it } from "vitest";
import { PaintModelingState } from "./PaintModelingState.svelte.ts";
import type { RenderPrimitive, RenderSegment, RenderTriangle, Vec2, Vec3 } from "./types.ts";

function drawStroke(state: PaintModelingState, a: Vec2, b: Vec2) {
    state.beginStroke(a, 800, 600);
    state.appendStrokePoint(b);
    state.finishStroke();
}

describe("PaintModelingState prototype", () => {
    it("preserves the source-view projection when source depth is sculpted", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.15, y: 0 }, { x: 0.15, y: 0 });

        const stroke = state.strokes[0];
        const sample = stroke.samples[Math.floor(stroke.samples.length / 2)];
        const sourceView = state.views.find(view => view.id === stroke.sourceViewId)!;
        const beforeProjection = state.projectSurfaceRef(sample.surfaceRef, sourceView)!;
        const beforeWorld = state.surfaceRefWorldPoint(sample.surfaceRef)!;

        expect(state.sculptDepthAt(sample.sourcePoint)).toBe(true);

        const afterProjection = state.projectSurfaceRef(sample.surfaceRef, sourceView)!;
        const afterWorld = state.surfaceRefWorldPoint(sample.surfaceRef)!;

        expect(afterProjection.x).toBeCloseTo(beforeProjection.x, 5);
        expect(afterProjection.y).toBeCloseTo(beforeProjection.y, 5);
        expect(distance3(afterWorld, beforeWorld)).toBeGreaterThan(0.005);
    });

    it("snaps a later-view stroke to an existing surface chart by default", () => {
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
        const segmentCount = state.buildRenderSegments(true).length;

        state.orbit.turn({ x: 80, y: 16 });

        expect(state.meshVersion).toBe(meshVersion);
        expect(state.buildRenderSegments(true).length).toBe(segmentCount);
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

    it("undoes depth and seam edits, not only paint strokes", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.14, y: 0 }, { x: 0.14, y: 0 });
        const sample = state.strokes[0].samples[Math.floor(state.strokes[0].samples.length / 2)];
        const beforeDepthWorld = state.surfaceRefWorldPoint(sample.surfaceRef)!;

        expect(state.sculptDepthAt(sample.sourcePoint)).toBe(true);
        expect(distance3(state.surfaceRefWorldPoint(sample.surfaceRef)!, beforeDepthWorld)).toBeGreaterThan(0.005);
        expect(state.undo()).toBe(true);
        expect(distance3(state.surfaceRefWorldPoint(sample.surfaceRef)!, beforeDepthWorld)).toBeLessThan(0.0001);

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

    it("does not treat unpainted chart area as snap or edit surface", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.72, y: -0.52 }, { x: -0.48, y: -0.52 });

        expect(state.sculptDepthAt({ x: 0.62, y: 0.54 })).toBe(false);
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
        const brushSegments = renderSegments.filter(segment => segment.width === 18);

        expect(brushSegments.length).toBeGreaterThanOrEqual(state.strokes[0].samples.length - 1);
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

        const surfacePrimitives = state.buildRenderSegments({ showChartWireframe: false });
        const surfaceBrushSegments = surfacePrimitives.filter(isRenderSegment).filter(segment => segment.width === 48);

        expect(surfacePrimitives.filter(isRenderTriangle)).toHaveLength(0);
        expect(surfaceBrushSegments.length).toBeGreaterThanOrEqual(state.strokes[0].samples.length - 1);

        const overlayPrimitives = state.buildRenderSegments({
            showChartWireframe: false,
            strokeRenderMode: "paint-order",
        });
        const overlayBrushSegments = overlayPrimitives.filter(isRenderSegment).filter(segment => segment.width === 48);

        expect(overlayBrushSegments.length).toBe(surfaceBrushSegments.length);
    });

    it("renders an in-progress stroke through WebGL preview segments in surface mode", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        state.setBrushWidth(36);

        state.beginStroke({ x: -0.16, y: -0.04 }, 800, 600);
        state.appendStrokePoint({ x: 0.16, y: 0.04 });

        const draftPrimitives = state.buildRenderSegments({ showChartWireframe: false });
        const draftBrushSegments = draftPrimitives.filter(isRenderSegment).filter(segment => segment.width === 36);

        expect(draftBrushSegments.length).toBeGreaterThan(0);

        state.finishStroke();
        const committedPrimitives = state.buildRenderSegments({ showChartWireframe: false });
        const committedBrushSegments = committedPrimitives.filter(isRenderSegment).filter(segment => segment.width === 36);

        expect(committedBrushSegments.length).toBeGreaterThan(0);
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
        const draftBrushSegments = draftPrimitives.filter(isRenderSegment).filter(segment => segment.width === 18);
        const rawPointCount = state.draftStroke?.length ?? 0;

        state.finishStroke();
        const stroke = state.strokes.at(-1);
        const committedPrimitives = state.buildRenderSegments({ showChartWireframe: false });
        const committedBrushSegments = committedPrimitives.filter(isRenderSegment).filter(segment => segment.width === 18);

        expect(rawPointCount).toBeGreaterThan(100);
        expect(stroke).toBeDefined();
        expect(draftBrushSegments.length).toBe(stroke!.samples.length - 1);
        expect(committedBrushSegments.length).toBe(draftBrushSegments.length);
        expect(stroke!.samples.length).toBeGreaterThan(112);

        for (let i = 0; i < draftBrushSegments.length; i++) {
            expect(distance3(draftBrushSegments[i].a, committedBrushSegments[i].a)).toBeLessThan(1e-6);
            expect(distance3(draftBrushSegments[i].b, committedBrushSegments[i].b)).toBeLessThan(1e-6);
        }
    });

    it("keeps draft stroke sample density stable as strokes get longer", () => {
        const short = new PaintModelingState();
        short.addObject();
        short.setPlacementMode("new-surface");
        short.beginStroke({ x: -0.12, y: 0 }, 800, 600);
        short.appendStrokePoint({ x: 0.12, y: 0 });
        const shortSegments = short.buildRenderSegments({ showChartWireframe: false })
            .filter(isRenderSegment)
            .filter(segment => segment.width === 18).length;

        const long = new PaintModelingState();
        long.addObject();
        long.setPlacementMode("new-surface");
        long.beginStroke({ x: -0.72, y: 0 }, 800, 600);
        long.appendStrokePoint({ x: 0.72, y: 0 });
        const longSegments = long.buildRenderSegments({ showChartWireframe: false })
            .filter(isRenderSegment)
            .filter(segment => segment.width === 18).length;

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

        const prefixSegments = state.buildRenderSegments({ showChartWireframe: false })
            .filter(isRenderSegment)
            .filter(segment => segment.width === 18);

        for (let i = 81; i <= 160; i++) {
            const t = i / 160;
            state.appendStrokePoint({
                x: -0.62 + t * 1.24,
                y: -0.08 + Math.sin(t * Math.PI * 8) * 0.12,
            });
        }

        const extendedSegments = state.buildRenderSegments({ showChartWireframe: false })
            .filter(isRenderSegment)
            .filter(segment => segment.width === 18);
        const stablePrefixCount = Math.max(8, prefixSegments.length - 16);

        expect(prefixSegments.length).toBeGreaterThan(32);
        expect(extendedSegments.length).toBeGreaterThan(prefixSegments.length);
        for (let i = 0; i < stablePrefixCount; i++) {
            expect(distance3(prefixSegments[i].a, extendedSegments[i].a)).toBeLessThan(1e-6);
            expect(distance3(prefixSegments[i].b, extendedSegments[i].b)).toBeLessThan(1e-6);
        }
    });

    it("can render chart wire independently from brush lattice previews", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        state.setBrushWidth(48);
        state.setDepthBrushRadius(0.08);
        drawStroke(state, { x: -0.2, y: 0 }, { x: 0.2, y: 0 });
        const preview = {
            tool: "depth-brush" as const,
            points: [{ x: 0, y: 0 }],
            delta: 0.04,
        };

        const wireOff = state.buildRenderSegments({ showChartWireframe: false }).filter(isRenderSegment);
        const wireOn = state.buildRenderSegments({ showChartWireframe: true }).filter(isRenderSegment);
        const latticeOff = state.buildRenderSegments({
            showChartWireframe: false,
            showBrushLattice: false,
            depthPreview: preview,
        }).filter(isRenderSegment);
        const latticeOn = state.buildRenderSegments({
            showChartWireframe: false,
            showBrushLattice: true,
            depthPreview: preview,
        }).filter(isRenderSegment);

        expect(wireOff.some(segment => segment.width === 1.15)).toBe(false);
        expect(wireOn.some(segment => segment.width === 1.15)).toBe(true);
        expect(latticeOff.some(segment => segment.width === 3.2 || segment.width === 2.6)).toBe(false);
        expect(latticeOn.some(segment => segment.width === 3.2)).toBe(true);
        expect(latticeOn.some(segment => segment.width === 2.6)).toBe(true);
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

    it("keeps painterly brush width and opacity in vector strokes", () => {
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
        const wideBrushSegments = primitives.filter(isRenderSegment).filter(segment => segment.width === 48);

        expect(wideCovered).toBeGreaterThan(narrowCovered * 2);
        expect(wideBrushSegments.length).toBeGreaterThan(0);
        expect(wideBrushSegments[0].color[3]).toBeCloseTo(0.42, 5);
        expect(primitives.filter(isRenderTriangle)).toHaveLength(0);
    });

    it("keeps thin committed strokes visible even when chart fill triangles are sparse", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        state.setBrushWidth(3);
        drawStroke(state, { x: -0.32, y: -0.2 }, { x: 0.32, y: 0.2 });

        const primitives = state.buildRenderSegments(false);
        const brushSegments = primitives.filter(isRenderSegment).filter(segment => segment.width === 3);

        expect(state.strokes).toHaveLength(1);
        expect(brushSegments.length).toBeGreaterThan(12);
        expect(primitives.filter(isRenderTriangle)).toHaveLength(0);
    });

    it("renders orientation grid and axis segments even without charts", () => {
        const state = new PaintModelingState();
        const segments = state.buildRenderSegments(false);
        const renderSegments = segments.filter(isRenderSegment);

        expect(renderSegments.length).toBeGreaterThan(40);
        expect(renderSegments.filter(segment => (segment.width ?? 0) > 2).length).toBe(3);
    });

    it("sculpts covered chart lattice points, not only stroke samples", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.18, y: 0 }, { x: 0.18, y: 0 });
        const chart = state.activeObject!.charts[0];
        const beforeDepths = [...chart.depths];

        expect(state.sculptDepthAt({ x: 0, y: 0 }, 0.04)).toBe(true);

        const editedCoveredTexels = chart.depths.filter((depth, index) =>
            chart.coverage[index] > 0
            && Math.abs(depth - beforeDepths[index]) > 1e-6
        ).length;

        expect(editedCoveredTexels).toBeGreaterThan(6);
    });

    it("separates steady depth brushing from anchored depth pulling", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        drawStroke(state, { x: -0.18, y: 0 }, { x: 0.18, y: 0 });
        const chart = state.activeObject!.charts[0];
        const beforeBrushDepths = [...chart.depths];

        state.setTool("depth-brush");
        expect(state.brushDepthAt({ x: 0, y: 0 })).toBe(true);
        const brushChart = state.activeObject!.charts[0];
        const brushedTexels = brushChart.depths.filter((depth, index) =>
            brushChart.coverage[index] > 0
            && Math.abs(depth - beforeBrushDepths[index]) > 1e-6
        ).length;

        const beforePullDepths = [...brushChart.depths];
        state.setTool("depth-pull");
        expect(state.sculptDepthAt({ x: 0, y: 0 }, -0.035)).toBe(true);
        const pullChart = state.activeObject!.charts[0];
        const pulledTexels = pullChart.depths.filter((depth, index) =>
            pullChart.coverage[index] > 0
            && Math.abs(depth - beforePullDepths[index]) > 1e-6
        ).length;

        expect(state.tool).toBe("depth-pull");
        expect(brushedTexels).toBeGreaterThan(6);
        expect(pulledTexels).toBeGreaterThan(6);
    });

    it("applies depth brush edits under the current cursor footprint after orbiting", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        state.setBrushWidth(56);
        drawStroke(state, { x: -0.22, y: 0 }, { x: 0.22, y: 0 });
        const sample = state.strokes[0].samples[Math.floor(state.strokes[0].samples.length / 2)];
        const beforeWorld = state.surfaceRefWorldPoint(sample.surfaceRef)!;

        state.orbit.turn({ x: 28, y: 4 });
        const currentScreenPoint = projectVisiblePoint(state.camera.viewProjMat, beforeWorld)!;

        expect(state.brushDepthAt(currentScreenPoint)).toBe(true);
        expect(distance3(state.surfaceRefWorldPoint(sample.surfaceRef)!, beforeWorld)).toBeGreaterThan(0.001);
    });

    it("adds visible depth brush preview ring and movement trails over hit surfaces", () => {
        const state = new PaintModelingState();
        state.addObject();
        state.setPlacementMode("new-surface");
        state.setBrushWidth(48);
        state.setDepthBrushRadius(0.08);
        drawStroke(state, { x: -0.2, y: 0 }, { x: 0.2, y: 0 });
        const meshVersion = state.meshVersion;

        const brushPrimitives = state.buildRenderSegments(false, {
            tool: "depth-brush",
            points: [{ x: 0, y: 0 }],
            delta: 0.04,
        });
        const pullPrimitives = state.buildRenderSegments(false, {
            tool: "depth-pull",
            points: [{ x: 0, y: 0 }],
            delta: 0.04,
        });
        const brushSegments = brushPrimitives.filter(isRenderSegment);
        const pullSegments = pullPrimitives.filter(isRenderSegment);
        const brushRing = brushSegments.find(segment => segment.width === 3.2);
        const pullRing = pullSegments.find(segment => segment.width === 3.2);

        expect(brushSegments.filter(segment => segment.width === 3.2).length).toBeGreaterThan(8);
        expect(brushSegments.filter(segment => segment.width === 2.6).length).toBeGreaterThan(0);
        expect(brushRing).toBeDefined();
        expect(pullRing).toBeDefined();
        expect(brushRing!.color[2]).toBeGreaterThan(brushRing!.color[0]);
        expect(pullRing!.color[0]).toBeGreaterThan(pullRing!.color[2]);
        expect(state.meshVersion).toBe(meshVersion);
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
    return primitive.kind !== "triangle";
}

function isRenderTriangle(primitive: RenderPrimitive): primitive is RenderTriangle {
    return primitive.kind === "triangle";
}

function coveredTexelCount(state: PaintModelingState): number {
    return state.activeObject!.charts[0].coverage.filter(value => value > 0.015).length;
}

function projectVisiblePoint(viewProjMat: number[] | Float32Array, p: Vec3): Vec2 | null {
    const clipX = viewProjMat[0] * p[0] + viewProjMat[4] * p[1] + viewProjMat[8] * p[2] + viewProjMat[12];
    const clipY = viewProjMat[1] * p[0] + viewProjMat[5] * p[1] + viewProjMat[9] * p[2] + viewProjMat[13];
    const clipZ = viewProjMat[2] * p[0] + viewProjMat[6] * p[1] + viewProjMat[10] * p[2] + viewProjMat[14];
    const clipW = viewProjMat[3] * p[0] + viewProjMat[7] * p[1] + viewProjMat[11] * p[2] + viewProjMat[15];
    if (!Number.isFinite(clipW) || clipW <= 1e-5) return null;
    const x = clipX / clipW;
    const y = clipY / clipW;
    const z = clipZ / clipW;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return null;
    if (z < -0.02 || z > 1.02) return null;
    return { x, y };
}

function distance3(a: Vec3, b: Vec3): number {
    return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
