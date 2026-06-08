import { vec3 } from "wgpu-matrix";
import { Camera } from "../viewer/Camera.svelte.ts";
import { CameraOrbit } from "../viewer/CameraOrbit.svelte.ts";
import type {
    BrushStyle,
    ChartRole,
    ChartProjectionMode,
    DepthBrushPreview,
    DepthTool,
    OcclusionClaim,
    PaintChart,
    PaintObject,
    PaintSample,
    PaintStroke,
    PaintTool,
    PaintView,
    PaintRenderOptions,
    PlacementMode,
    RenderPrimitive,
    RenderSegment,
    PaintStrokeRenderMode,
    SurfaceHit,
    SurfaceRef,
    Vec2,
    Vec3,
    Vec4,
} from "./types.ts";

const DEFAULT_BRUSH: BrushStyle = {
    color: "#ffd27a",
    width: 18,
    opacity: 0.88,
};

const CHART_RESOLUTION = 65;
const OCCLUSION_GAP = 0.075;
const MIN_DEPTH = 0.06;
const COVERAGE_EPSILON = 0.015;
const PAINT_EPSILON = 0.015;
const GRID_EXTENT = 2.5;
const GRID_STEP = 0.25;
const GRID_PLANE_Z = -0.02;
const DEPTH_PREVIEW_RING_WIDTH = 3.2;
const DEPTH_PREVIEW_TRAIL_WIDTH = 2.6;
const DEPTH_PREVIEW_IDLE_LENGTH = 0.035;
const DEPTH_BRUSH_STAMP_SCALE = 0.15;
const MIN_STROKE_SPLINE_CONTROL_POINTS = 4;
const STROKE_SPLINE_SAMPLE_SPACING = 0.018;
const DEPTH_BRUSH_FOOTPRINT_OFFSETS: Vec2[] = [
    { x: 0, y: 0 },
    { x: 0.62, y: 0 },
    { x: -0.62, y: 0 },
    { x: 0, y: 0.62 },
    { x: 0, y: -0.62 },
];

interface PaintSceneSnapshot {
    viewportWidth: number;
    viewportHeight: number;
    views: PaintView[];
    objects: PaintObject[];
    strokes: PaintStroke[];
    occlusionClaims: OcclusionClaim[];
    activeObjectId: string | null;
    activeViewId: string | null;
    chartProjectionMode: ChartProjectionMode;
}

interface ChartPaintSample {
    point: Vec2;
    depth: number;
}

interface ChartPaintRun {
    chart: PaintChart;
    samples: ChartPaintSample[];
    updateDepth: boolean;
    requireCoverage: boolean;
}

interface ChartUvRun {
    chart: PaintChart;
    points: Vec2[];
}

export class PaintModelingState {
    viewportWidth = $state(1);
    viewportHeight = $state(1);

    views = $state<PaintView[]>([]);
    objects = $state<PaintObject[]>([]);
    strokes = $state<PaintStroke[]>([]);
    occlusionClaims = $state<OcclusionClaim[]>([]);
    activeObjectId = $state<string | null>(null);
    activeViewId = $state<string | null>(null);
    placementMode = $state<PlacementMode>("snap");
    chartProjectionMode = $state<ChartProjectionMode>("view-plane");
    tool = $state<PaintTool>("paint");
    brush = $state<BrushStyle>({ ...DEFAULT_BRUSH });
    depthBrushRadius = $state(0.16);
    depthBrushStrength = $state(0.06);
    seamBrushRadius = $state(0.055);
    draftStroke = $state<Vec2[] | null>(null);
    undoStack = $state<PaintSceneSnapshot[]>([]);
    meshVersion = $state(0);
    raycastCountForDiagnostics = 0;
    private pendingStrokeUndoSnapshot: PaintSceneSnapshot | null = null;
    private undoGroup: { snapshot: PaintSceneSnapshot; dirty: boolean } | null = null;

    readonly orbit = new CameraOrbit();
    readonly camera = new Camera({
        controlScheme: this.orbit,
        screenDims: {
            width: () => this.viewportWidth,
            height: () => this.viewportHeight,
        },
    });

    get activeObject(): PaintObject | null {
        return this.objects.find(object => object.id === this.activeObjectId) ?? null;
    }

    get activeView(): PaintView | null {
        return this.views.find(view => view.id === this.activeViewId) ?? null;
    }

    get isCameraAtActiveView(): boolean {
        const view = this.activeView;
        return !!view && !this.cameraMovedFrom(view);
    }

    get currentViewName(): string {
        if (this.isCameraAtActiveView) return this.activeView?.name ?? "No saved view";
        return this.activeView ? "New view" : "No saved view";
    }

    get chartCount(): number {
        return this.objects.reduce((sum, object) => sum + object.charts.length, 0);
    }

    get seamCount(): number {
        return this.objects.reduce(
            (sum, object) => sum + object.charts.reduce((chartSum, chart) => chartSum + chart.seams.filter(Boolean).length, 0),
            0,
        );
    }

    get canUndo(): boolean {
        return this.undoStack.length > 0;
    }

    addObject(name?: string, recordHistory = true) {
        if (recordHistory) this.recordUndoSnapshot();
        this.ensureActiveView(this.viewportWidth, this.viewportHeight, false);
        const index = this.objects.length + 1;
        const object: PaintObject = {
            id: makeId("object"),
            name: name ?? `Object ${index}`,
            visible: true,
            locked: false,
            layerIndex: this.nextLayerIndex(),
            charts: [],
        };
        this.objects = [...this.objects, object];
        this.activeObjectId = object.id;
        this.meshVersion += 1;
    }

    selectObject(objectId: string) {
        if (this.objects.some(object => object.id === objectId)) {
            this.activeObjectId = objectId;
        }
    }

    setPlacementMode(mode: PlacementMode) {
        this.placementMode = mode;
    }

    setChartProjectionMode(mode: ChartProjectionMode) {
        this.chartProjectionMode = mode;
    }

    setTool(tool: PaintTool) {
        this.tool = tool;
    }

    setBrushColor(color: string) {
        this.brush = { ...this.brush, color };
    }

    setBrushWidth(width: number) {
        this.brush = { ...this.brush, width: clamp(width, 1, 72) };
    }

    setBrushOpacity(opacity: number) {
        this.brush = { ...this.brush, opacity: clamp(opacity, 0.05, 1) };
    }

    setDepthBrushRadius(radius: number) {
        this.depthBrushRadius = clamp(radius, 0.04, 0.85);
    }

    setDepthBrushStrength(strength: number) {
        this.depthBrushStrength = clamp(strength, 0.01, 0.2);
    }

    setSeamBrushRadius(radius: number) {
        this.seamBrushRadius = clamp(radius, 0.015, 0.22);
    }

    resetDiagnostics() {
        this.raycastCountForDiagnostics = 0;
    }

    beginUndoGroup() {
        if (!this.undoGroup) {
            this.undoGroup = {
                snapshot: this.captureSceneSnapshot(),
                dirty: false,
            };
        }
    }

    commitUndoGroup() {
        if (this.undoGroup?.dirty) {
            this.pushUndoSnapshot(this.undoGroup.snapshot);
        }
        this.undoGroup = null;
    }

    cancelUndoGroup() {
        this.undoGroup = null;
    }

    beginStroke(point: Vec2, width: number, height: number) {
        const undoSnapshot = this.captureSceneSnapshot();
        this.viewportWidth = Math.max(1, width);
        this.viewportHeight = Math.max(1, height);
        if (!this.activeObjectId) this.addObject(undefined, false);
        const object = this.activeObject;
        if (!object || object.locked || !object.visible) {
            this.pendingStrokeUndoSnapshot = null;
            return;
        }
        this.ensureActiveView(width, height, false);
        this.draftStroke = [point];
        this.pendingStrokeUndoSnapshot = undoSnapshot;
    }

    appendStrokePoint(point: Vec2) {
        if (!this.draftStroke || this.draftStroke.length === 0) return;
        const last = this.draftStroke[this.draftStroke.length - 1];
        const dx = point.x - last.x;
        const dy = point.y - last.y;
        if (dx * dx + dy * dy < 0.00008) return;
        this.draftStroke.push(point);
    }

    finishStroke() {
        const undoSnapshot = this.pendingStrokeUndoSnapshot ?? this.captureSceneSnapshot();
        const object = this.activeObject;
        const view = this.activeView;
        if (!this.draftStroke || this.draftStroke.length < 2 || !object || !view) {
            this.draftStroke = null;
            if (this.pendingStrokeUndoSnapshot) {
                this.restoreSceneSnapshot(this.pendingStrokeUndoSnapshot);
            }
            this.pendingStrokeUndoSnapshot = null;
            return;
        }

        const sourcePoints = samplePaintStrokeSpline(this.draftStroke);
        const strokeSamples = this.placeStrokeSamples(object, view, sourcePoints, this.placementMode);
        if (strokeSamples.samples.length < 2) {
            this.draftStroke = null;
            this.restoreSceneSnapshot(undoSnapshot);
            this.pendingStrokeUndoSnapshot = null;
            return;
        }

        this.pushUndoSnapshot(undoSnapshot);
        const stroke: PaintStroke = {
            id: makeId("stroke"),
            objectId: object.id,
            sourceViewId: view.id,
            placement: this.placementMode,
            samples: strokeSamples.samples,
            style: { ...this.brush },
            paintOrder: this.nextPaintOrder(object.id),
        };

        this.strokes = [...this.strokes, stroke];
        if (strokeSamples.occlusionClaim) {
            this.occlusionClaims = [...this.occlusionClaims, strokeSamples.occlusionClaim];
        }
        this.touchCharts(strokeSamples.touchedChartIds);
        this.draftStroke = null;
        this.pendingStrokeUndoSnapshot = null;
        this.meshVersion += 1;
    }

    undo(): boolean {
        const snapshot = this.undoStack.at(-1);
        if (!snapshot) return false;
        this.undoStack = this.undoStack.slice(0, -1);
        this.restoreSceneSnapshot(snapshot);
        return true;
    }

    undoStroke(): boolean {
        return this.undo();
    }

    deleteActiveObject(): boolean {
        return this.activeObjectId ? this.deleteObject(this.activeObjectId) : false;
    }

    deleteObject(objectId: string): boolean {
        const existing = this.objects.find(object => object.id === objectId);
        if (!existing) return false;

        this.recordUndoSnapshot();
        this.objects = this.objects.filter(object => object.id !== objectId);
        this.strokes = this.strokes.filter(stroke => stroke.objectId !== objectId);
        this.occlusionClaims = this.occlusionClaims.filter(claim => claim.objectId !== objectId);

        if (this.activeObjectId === objectId) {
            this.activeObjectId = this.objects[0]?.id ?? null;
        }
        this.draftStroke = null;
        this.meshVersion += 1;
        return true;
    }

    deleteActiveView(): boolean {
        return this.activeViewId ? this.deleteView(this.activeViewId) : false;
    }

    deleteView(viewId: string): boolean {
        const existing = this.views.find(view => view.id === viewId);
        if (!existing) return false;

        this.recordUndoSnapshot();
        const removedChartIds = new Set<string>();
        this.objects = this.objects.map(object => {
            const keptCharts = object.charts.filter(chart => {
                if (chart.sourceViewId !== viewId) return true;
                removedChartIds.add(chart.id);
                return false;
            });
            return { ...object, charts: keptCharts };
        });

        this.views = this.views.filter(view => view.id !== viewId);
        this.strokes = this.strokes.filter(stroke =>
            stroke.sourceViewId !== viewId
            && !stroke.samples.some(sample => removedChartIds.has(sample.surfaceRef.chartId))
        );
        this.occlusionClaims = this.occlusionClaims.filter(claim =>
            claim.viewId !== viewId
            && !removedChartIds.has(claim.frontChartId)
            && !claim.backRefs.some(ref => removedChartIds.has(ref.chartId))
        );

        if (this.activeViewId === viewId) {
            const nextView = this.views[0] ?? null;
            this.activeViewId = null;
            if (nextView) {
                this.selectView(nextView.id);
            }
        }
        this.draftStroke = null;
        this.meshVersion += 1;
        return true;
    }

    sculptDepthAt(point: Vec2, deltaOrReverse: number | boolean = this.depthBrushStrength): boolean {
        return this.sculptDepthAlong([point], deltaOrReverse);
    }

    sculptDepthAlong(points: Vec2[], deltaOrReverse: number | boolean = this.depthBrushStrength): boolean {
        const delta = typeof deltaOrReverse === "boolean"
            ? this.depthBrushStrength * (deltaOrReverse ? -1 : 1)
            : deltaOrReverse;
        return this.applyProjectedDepthEdit(points, delta);
    }

    brushDepthAt(point: Vec2, reverse = false): boolean {
        return this.brushDepthAlong([point], reverse);
    }

    brushDepthAlong(points: Vec2[], reverse = false): boolean {
        return this.applyProjectedDepthEdit(points, this.depthBrushStep(reverse));
    }

    depthBrushStep(reverse = false): number {
        return this.depthBrushStrength * DEPTH_BRUSH_STAMP_SCALE * (reverse ? -1 : 1);
    }

    private applyProjectedDepthEdit(points: Vec2[], delta: number): boolean {
        const object = this.activeObject;
        const view = this.currentEffectView();
        if (!object || !view || object.locked) return false;
        if (points.length === 0) return false;
        if (Math.abs(delta) <= 1e-6) return false;

        const charts = this.collectBrushFootprintCharts(object, view, points, this.depthBrushRadius, this.maxEffectSamplesForBrush());
        if (charts.length === 0) return false;

        this.recordUndoSnapshot();
        const touchedChartIds = new Set<string>();
        for (const chart of charts) {
            const sourceView = this.views.find(item => item.id === chart.sourceViewId);
            if (!sourceView) continue;
            if (editChartDepthAlongScreenPolyline(
                chart,
                sourceView,
                view,
                points,
                this.depthBrushRadius,
                this.viewportWidth,
                this.viewportHeight,
                delta,
            )) {
                touchedChartIds.add(chart.id);
            }
        }
        if (touchedChartIds.size === 0) return false;
        this.touchCharts(touchedChartIds);
        this.meshVersion += 1;
        return true;
    }

    markSeamAt(point: Vec2): boolean {
        return this.markSeamAlong([point]);
    }

    markSeamAlong(points: Vec2[]): boolean {
        const object = this.activeObject;
        const view = this.currentEffectView();
        if (!object || !view || object.locked) return false;
        const runs = this.collectHitRuns(object, view, points, this.maxEffectSamplesForBrush());
        if (runs.length === 0) return false;

        this.recordUndoSnapshot();
        const touchedChartIds = new Set<string>();
        for (const run of runs) {
            if (markChartSeamAlongPolyline(run.chart, run.points, this.seamBrushRadius)) {
                touchedChartIds.add(run.chart.id);
            }
        }
        if (touchedChartIds.size === 0) return false;
        this.touchCharts(touchedChartIds);
        this.meshVersion += 1;
        return true;
    }

    saveCurrentView(width = this.viewportWidth, height = this.viewportHeight, recordHistory = true): PaintView {
        if (recordHistory) this.recordUndoSnapshot();
        this.viewportWidth = Math.max(1, width);
        this.viewportHeight = Math.max(1, height);
        const view = this.captureCurrentView(`View ${this.views.length + 1}`, width, height);
        this.views = [...this.views, view];
        this.activeViewId = view.id;
        return view;
    }

    selectView(viewId: string) {
        const view = this.views.find(item => item.id === viewId);
        if (!view) return;
        this.activeViewId = view.id;
        this.orbit.long = view.long;
        this.orbit.lat = view.lat;
        this.orbit.radius = view.radius;
        this.orbit.offset = vec3.fromValues(view.offset[0], view.offset[1], view.offset[2]);
    }

    ensureActiveView(width = this.viewportWidth, height = this.viewportHeight, recordHistory = true) {
        this.viewportWidth = Math.max(1, width);
        this.viewportHeight = Math.max(1, height);
        const active = this.activeView;
        if (active && !this.cameraMovedFrom(active)) {
            if (active.width === this.viewportWidth && active.height === this.viewportHeight) return;
            if (!this.viewHasAuthoredContent(active.id)) {
                const refreshedView = this.captureCurrentView(active.name, this.viewportWidth, this.viewportHeight);
                this.views = this.views.map(view => view.id === active.id
                    ? {
                        ...refreshedView,
                        id: active.id,
                        createdAt: active.createdAt,
                    }
                    : view);
                return;
            }
        }
        this.saveCurrentView(width, height, recordHistory);
    }

    cameraMovedFrom(view: PaintView): boolean {
        const offset = Array.from(this.orbit.offset).slice(0, 3) as Vec3;
        const offsetDelta = distance3(offset, view.offset);
        return Math.abs(this.orbit.long - view.long) > 0.015
            || Math.abs(this.orbit.lat - view.lat) > 0.015
            || Math.abs(Math.log(this.orbit.radius / view.radius)) > 0.015
            || offsetDelta > 0.01;
    }

    surfaceRefWorldPoint(ref: SurfaceRef): Vec3 | null {
        const chart = this.findChart(ref.chartId);
        if (!chart) return null;
        const view = this.views.find(item => item.id === chart.sourceViewId);
        if (!view) return null;
        return chartPointToWorldFromView(chart, view, ref.uv);
    }

    projectSurfaceRef(ref: SurfaceRef, view: PaintView | null = this.activeView): Vec2 | null {
        if (!view) return null;
        const world = this.surfaceRefWorldPoint(ref);
        if (!world) return null;
        return projectVisiblePoint(view.viewProjMat, world);
    }

    buildRenderSegments(options: boolean | PaintRenderOptions = true, depthPreview?: DepthBrushPreview): RenderPrimitive[] {
        const renderOptions = normalizeRenderOptions(options, depthPreview);
        const segments: RenderPrimitive[] = [];
        const objectById = new Map(this.objects.map(object => [object.id, object]));
        const viewById = new Map(this.views.map(view => [view.id, view]));
        const chartById = new Map<string, PaintChart>();
        for (const object of this.objects) {
            for (const chart of object.charts) {
                chartById.set(chart.id, chart);
            }
        }

        appendOrientationSegments(segments);

        if (renderOptions.showPaintSurface) {
            for (const object of this.objects) {
                if (!object.visible) continue;
                for (const chart of object.charts) {
                    const sourceView = viewById.get(chart.sourceViewId);
                    if (!sourceView) continue;
                    const worldAt = (point: Vec2) => chartPointToWorldFromView(chart, sourceView, point);
                    appendPaintTriangles(segments, chart, worldAt);
                }
            }
        }

        if (renderOptions.showChartWireframe) {
            for (const object of this.objects) {
                if (!object.visible) continue;
                for (const chart of object.charts) {
                    const sourceView = viewById.get(chart.sourceViewId);
                    if (!sourceView) continue;
                    const worldAt = (point: Vec2) => chartPointToWorldFromView(chart, sourceView, point);
                    appendChartSegments(segments, chart, worldAt);
                }
            }
        }

        const worldPointForRef = (ref: SurfaceRef): Vec3 | null => {
            const chart = chartById.get(ref.chartId);
            if (!chart) return null;
            const view = viewById.get(chart.sourceViewId);
            if (!view) return null;
            return chartPointToWorldFromView(chart, view, ref.uv);
        };

        const committedStrokeRenderMode = renderOptions.strokeRenderMode === "view-depth"
            ? "view-depth"
            : "paint-order";
        for (const stroke of this.sortedStrokesForRender(committedStrokeRenderMode, objectById, worldPointForRef)) {
            appendStrokeRenderSegments(segments, stroke, worldPointForRef);
        }

        this.appendDraftStrokePreviewSegments(segments);

        if (renderOptions.depthPreview && renderOptions.showBrushLattice) {
            this.appendDepthBrushPreviewSegments(segments, renderOptions.depthPreview);
        }

        return segments;
    }

    private sortedStrokesForRender(
        mode: PaintStrokeRenderMode,
        objectById: Map<string, PaintObject>,
        worldPointForRef: (ref: SurfaceRef) => Vec3 | null,
    ): PaintStroke[] {
        const strokes = this.strokes.filter(stroke => {
            const object = objectById.get(stroke.objectId);
            return !!object?.visible;
        });

        if (mode === "view-depth") {
            return strokes
                .map(stroke => ({
                    stroke,
                    objectLayer: objectById.get(stroke.objectId)?.layerIndex ?? 0,
                    depth: this.strokeCameraDepth(stroke, worldPointForRef),
                }))
                .sort((a, b) =>
                    b.depth - a.depth
                    || a.objectLayer - b.objectLayer
                    || a.stroke.paintOrder - b.stroke.paintOrder
                )
                .map(entry => entry.stroke);
        }

        return strokes.slice().sort((a, b) =>
            (objectById.get(a.objectId)?.layerIndex ?? 0) - (objectById.get(b.objectId)?.layerIndex ?? 0)
            || a.paintOrder - b.paintOrder
        );
    }

    private strokeCameraDepth(stroke: PaintStroke, worldPointForRef: (ref: SurfaceRef) => Vec3 | null): number {
        let total = 0;
        let count = 0;
        const stride = Math.max(1, Math.floor(stroke.samples.length / 32));
        for (let i = 0; i < stroke.samples.length; i += stride) {
            const world = worldPointForRef(stroke.samples[i].surfaceRef);
            const depth = world ? projectedDepth(this.camera.viewProjMat, world) : null;
            if (depth === null) continue;
            total += depth;
            count += 1;
        }
        return count === 0 ? Number.NEGATIVE_INFINITY : total / count;
    }

    private appendDraftStrokePreviewSegments(segments: RenderPrimitive[]) {
        const object = this.activeObject;
        const view = this.activeView;
        if (!this.draftStroke || this.draftStroke.length < 2 || !object?.visible || object.locked || !view) return;

        const color = parseColor(this.brush.color, this.brush.opacity);
        const points = samplePaintStrokeSpline(this.draftStroke);
        let previous = points.length > 0
            ? this.draftStrokeWorldPoint(object, view, points[0])
            : null;
        for (let i = 1; i < points.length; i++) {
            const current = this.draftStrokeWorldPoint(object, view, points[i]);
            appendWorldSegment(segments, previous, current, color, this.brush.width);
            previous = current;
        }
    }

    private draftStrokeWorldPoint(object: PaintObject, view: PaintView, point: Vec2): Vec3 | null {
        if (this.placementMode === "snap") {
            const hit = this.raycastObjectSurface(object, view, point);
            if (hit) return hit.world;
        }

        if (this.placementMode === "paint-behind") {
            const hit = this.raycastObjectSurface(object, view, point);
            if (hit) {
                const depth = depthForProjectionAtPoint(
                    view,
                    point,
                    hit.viewDepth + OCCLUSION_GAP,
                    this.chartProjectionMode,
                );
                return viewPointToWorldAtProjectionDepth(view, point, depth, this.chartProjectionMode);
            }
            return viewPointToWorldAtProjectionDepth(
                view,
                point,
                this.defaultDepthForView(view) * 1.12,
                this.chartProjectionMode,
            );
        }

        if (this.placementMode === "occluding-surface") {
            const hit = this.raycastObjectSurface(object, view, point);
            if (hit) {
                const depth = depthForProjectionAtPoint(
                    view,
                    point,
                    Math.max(MIN_DEPTH, hit.viewDepth - OCCLUSION_GAP),
                    this.chartProjectionMode,
                );
                return viewPointToWorldAtProjectionDepth(view, point, depth, this.chartProjectionMode);
            }
            return viewPointToWorldAtProjectionDepth(
                view,
                point,
                this.defaultDepthForView(view) * 0.82,
                this.chartProjectionMode,
            );
        }

        return viewPointToWorldAtProjectionDepth(
            view,
            point,
            this.defaultDepthForView(view),
            this.chartProjectionMode,
        );
    }

    private currentEffectView(): PaintView {
        const width = Math.max(1, this.viewportWidth);
        const height = Math.max(1, this.viewportHeight);
        const active = this.activeView;
        if (active && !this.cameraMovedFrom(active) && active.width === width && active.height === height) {
            return active;
        }
        return this.captureCurrentView("Interaction view", width, height);
    }

    private captureCurrentView(name: string, width: number, height: number): PaintView {
        const offset = Array.from(this.orbit.offset).slice(0, 3) as Vec3;
        return {
            id: makeId("view"),
            name,
            long: this.orbit.long,
            lat: this.orbit.lat,
            radius: this.orbit.radius,
            offset,
            width,
            height,
            viewProjMat: Array.from(this.camera.viewProjMat),
            viewProjInvMat: Array.from(this.camera.viewProjInvMat),
            viewMat: Array.from(this.camera.viewMat),
            viewInvMat: Array.from(this.camera.viewInvMat),
            createdAt: Date.now(),
        };
    }

    private placeStrokeSamples(
        object: PaintObject,
        view: PaintView,
        points: Vec2[],
        placement: PlacementMode,
    ): { samples: PaintSample[]; occlusionClaim?: OcclusionClaim; touchedChartIds: Set<string> } {
        if (placement === "occluding-surface") {
            return this.placeOccludingSamples(object, view, points);
        }

        const samples: PaintSample[] = [];
        const touchedChartIds = new Set<string>();
        const paintRuns: ChartPaintRun[] = [];
        let fallbackChart: PaintChart | null = null;
        const fallbackDepth = this.defaultDepthForView(view);
        const paintDepthRadius = this.paintDepthRadiusForView(view);

        for (const point of points) {
            if (placement === "snap") {
                const hit = this.raycastObjectSurface(object, view, point);
                if (hit) {
                    const chart = this.findChart(hit.chartId);
                    if (chart) {
                        appendPaintRun(paintRuns, chart, {
                            point: hit.surfaceRef.uv,
                            depth: sampleChartDepth(chart, hit.surfaceRef.uv),
                        }, false, true);
                    }
                    samples.push({
                        sourcePoint: point,
                        surfaceRef: hit.surfaceRef,
                        placement,
                    });
                    continue;
                }
            }

            let depth = fallbackDepth;
            let depthIsViewRayDistance = false;
            if (placement === "paint-behind") {
                const hits = this.raycastObjectSurfaces(object, view, point);
                const firstHit = hits[0] ?? null;
                const backHit = firstHit
                    ? hits.find(hit => hit.viewDepth > firstHit.viewDepth + OCCLUSION_GAP * 0.5) ?? null
                    : null;
                if (backHit) {
                    const chart = this.findChart(backHit.chartId);
                    if (chart) {
                        appendPaintRun(paintRuns, chart, {
                            point: backHit.surfaceRef.uv,
                            depth: sampleChartDepth(chart, backHit.surfaceRef.uv),
                        }, false, true);
                    }
                    samples.push({
                        sourcePoint: point,
                        surfaceRef: backHit.surfaceRef,
                        placement,
                    });
                    continue;
                }
                if (firstHit) {
                    depth = firstHit.viewDepth + OCCLUSION_GAP;
                    depthIsViewRayDistance = true;
                }
            }

            const role: ChartRole = placement === "paint-behind" ? "behind" : "surface";
            fallbackChart ??= this.getOrCreateChart(object, view, role);
            appendPaintRun(paintRuns, fallbackChart, {
                point,
                depth: depthIsViewRayDistance
                    ? depthForProjectionAtPoint(view, point, depth, fallbackChart.projectionMode)
                    : depth,
            }, true, false);
            samples.push({
                sourcePoint: point,
                surfaceRef: { chartId: fallbackChart.id, uv: { ...point } },
                placement,
            });
        }

        for (const run of paintRuns) {
            if (applyStrokeToChartGeometry(run.chart, run.samples, paintDepthRadius, {
                updateDepth: run.updateDepth,
                requireCoverage: run.requireCoverage,
            })) {
                touchedChartIds.add(run.chart.id);
            }
        }

        return { samples, touchedChartIds };
    }

    private placeOccludingSamples(
        object: PaintObject,
        view: PaintView,
        points: Vec2[],
    ): { samples: PaintSample[]; occlusionClaim?: OcclusionClaim; touchedChartIds: Set<string> } {
        const chart = this.getOrCreateChart(object, view, "occluder");
        const touchedChartIds = new Set([chart.id]);
        const paintDepthRadius = this.paintDepthRadiusForView(view);
        const claim: OcclusionClaim = {
            id: makeId("occlusion"),
            objectId: object.id,
            viewId: view.id,
            frontChartId: chart.id,
            backRefs: [],
            mask: [],
            createdAt: Date.now(),
        };
        const samples: PaintSample[] = [];
        const paintSamples: ChartPaintSample[] = [];

        for (const point of points) {
            const backHit = this.raycastObjectSurface(object, view, point, chart.id);
            const depth = backHit
                ? depthForProjectionAtPoint(view, point, Math.max(MIN_DEPTH, backHit.viewDepth - OCCLUSION_GAP), chart.projectionMode)
                : this.defaultDepthForView(view) * 0.82;
            paintSamples.push({ point, depth });
            claim.mask.push({ ...point });
            if (backHit) claim.backRefs.push(backHit.surfaceRef);
            samples.push({
                sourcePoint: point,
                surfaceRef: { chartId: chart.id, uv: { ...point } },
                placement: "occluding-surface",
            });
        }

        applyStrokeToChartGeometry(chart, paintSamples, paintDepthRadius, {
            updateDepth: true,
            requireCoverage: false,
        });
        markChartSeamAlongPolyline(chart, points, this.seamBrushRadius);
        return { samples, occlusionClaim: claim, touchedChartIds };
    }

    private defaultDepthForView(view: PaintView): number {
        const camera = cameraCenter(view);
        return Math.max(MIN_DEPTH, distance3(camera, [0, 0, 0]));
    }

    private paintDepthRadiusForView(view: PaintView): number {
        const minDimension = Math.max(1, Math.min(view.width, view.height));
        return clamp(this.brush.width / minDimension * 1.55, 0.035, 0.28);
    }

    private maxEffectSamplesForBrush(): number {
        return Math.round(clamp(96 - this.depthBrushRadius * 36, 48, 96));
    }

    private collectHitRuns(
        object: PaintObject,
        view: PaintView,
        points: Vec2[],
        maxSamples: number,
    ): ChartUvRun[] {
        const runs: ChartUvRun[] = [];
        for (const point of resamplePaintPolyline(points, maxSamples)) {
            const hit = this.raycastObjectSurface(object, view, point);
            if (!hit) continue;
            const chart = this.findChart(hit.chartId);
            if (!chart) continue;
            appendUvRun(runs, chart, hit.surfaceRef.uv);
        }
        return runs;
    }

    private appendDepthBrushPreviewSegments(segments: RenderPrimitive[], preview: DepthBrushPreview) {
        const object = this.activeObject;
        const view = this.currentEffectView();
        if (!object || !view || object.locked || preview.points.length === 0) return;

        const color = depthPreviewColor(preview.tool, preview.delta, 0.95);
        const charts = this.collectBrushFootprintCharts(object, view, preview.points, this.depthBrushRadius, this.maxEffectSamplesForBrush());
        this.appendDepthBrushSurfaceRing(segments, object, view, preview.points.at(-1)!, this.depthBrushRadius, color);

        for (const chart of charts) {
            const sourceView = this.views.find(item => item.id === chart.sourceViewId);
            if (!sourceView) continue;
            appendDepthBrushPreview(
                segments,
                chart,
                sourceView,
                view,
                preview.points,
                this.depthBrushRadius,
                this.viewportWidth,
                this.viewportHeight,
                preview.delta,
                preview.tool,
            );
        }
    }

    private collectBrushFootprintCharts(
        object: PaintObject,
        view: PaintView,
        points: Vec2[],
        radius: number,
        maxSamples: number,
    ): PaintChart[] {
        const chartById = new Map(object.charts.map(chart => [chart.id, chart]));
        const chartIds = new Set<string>();
        for (const point of sampleBrushFootprint(points, radius, this.viewportWidth, this.viewportHeight, maxSamples)) {
            const hit = this.raycastObjectSurface(object, view, point);
            if (hit) chartIds.add(hit.chartId);
        }
        return [...chartIds]
            .map(chartId => chartById.get(chartId))
            .filter((chart): chart is PaintChart => !!chart);
    }

    private appendDepthBrushSurfaceRing(
        segments: RenderPrimitive[],
        object: PaintObject,
        view: PaintView,
        center: Vec2,
        radius: number,
        color: Vec4,
    ) {
        const steps = 48;
        let previousHit: SurfaceHit | null = null;
        for (let i = 0; i <= steps; i++) {
            const angle = i / steps * Math.PI * 2;
            const point = offsetBrushPoint(
                center,
                { x: Math.cos(angle), y: Math.sin(angle) },
                radius,
                this.viewportWidth,
                this.viewportHeight,
            );
            const hit = this.raycastObjectSurface(object, view, point);
            if (previousHit && hit && previousHit.chartId === hit.chartId) {
                appendWorldSegment(segments, previousHit.world, hit.world, color, DEPTH_PREVIEW_RING_WIDTH);
            }
            previousHit = hit;
        }
    }

    private viewHasAuthoredContent(viewId: string): boolean {
        return this.strokes.some(stroke => stroke.sourceViewId === viewId)
            || this.occlusionClaims.some(claim => claim.viewId === viewId)
            || this.objects.some(object => object.charts.some(chart => chart.sourceViewId === viewId));
    }

    private getOrCreateChart(object: PaintObject, view: PaintView, role: ChartRole): PaintChart {
        const projectionMode = this.chartProjectionMode;
        const existing = object.charts.find(chart =>
            chart.sourceViewId === view.id
            && chart.role === role
            && chart.projectionMode === projectionMode
        );
        if (existing) return existing;
        const chart = createChart({
            objectId: object.id,
            sourceViewId: view.id,
            role,
            projectionMode,
            depth: role === "occluder" ? this.defaultDepthForView(view) * 0.82 : this.defaultDepthForView(view),
        });
        object.charts.push(chart);
        return chart;
    }

    private raycastObjectSurface(
        object: PaintObject,
        view: PaintView,
        point: Vec2,
        excludeChartId?: string,
    ): SurfaceHit | null {
        return this.raycastObjectSurfaces(object, view, point, excludeChartId)[0] ?? null;
    }

    private raycastObjectSurfaces(
        object: PaintObject,
        view: PaintView,
        point: Vec2,
        excludeChartId?: string,
    ): SurfaceHit[] {
        if (!object.visible) return [];
        this.raycastCountForDiagnostics += 1;
        const ray = makeViewRay(view, point);
        if (!ray) return [];

        const hits: SurfaceHit[] = [];
        for (const chart of object.charts) {
            if (chart.id === excludeChartId) continue;
            if (!chartHasCoverage(chart)) continue;
            const sourceView = this.views.find(item => item.id === chart.sourceViewId);
            if (!sourceView) continue;
            const chartHits = raycastChart(chart, ray, uv => chartPointToWorldFromView(chart, sourceView, uv));
            for (const hit of chartHits) {
                hits.push({
                    objectId: object.id,
                    chartId: chart.id,
                    surfaceRef: { chartId: chart.id, uv: hit.uv },
                    world: hit.world,
                    viewDepth: hit.t,
                });
            }
        }

        return hits.sort((a, b) => a.viewDepth - b.viewDepth);
    }

    private findChart(chartId: string): PaintChart | null {
        for (const object of this.objects) {
            const chart = object.charts.find(item => item.id === chartId);
            if (chart) return chart;
        }
        return null;
    }

    private nextLayerIndex(): number {
        return this.objects.reduce((max, object) => Math.max(max, object.layerIndex), -1) + 1;
    }

    private nextPaintOrder(objectId: string): number {
        return this.strokes
            .filter(stroke => stroke.objectId === objectId)
            .reduce((max, stroke) => Math.max(max, stroke.paintOrder), -1) + 1;
    }

    private recordUndoSnapshot() {
        if (this.undoGroup) {
            this.undoGroup.dirty = true;
            return;
        }
        this.pushUndoSnapshot(this.captureSceneSnapshot());
    }

    private pushUndoSnapshot(snapshot: PaintSceneSnapshot) {
        this.undoStack = [...this.undoStack, snapshot];
    }

    private captureSceneSnapshot(): PaintSceneSnapshot {
        return {
            viewportWidth: this.viewportWidth,
            viewportHeight: this.viewportHeight,
            views: this.views.map(cloneView),
            objects: this.objects.map(cloneObject),
            strokes: this.strokes.map(cloneStroke),
            occlusionClaims: this.occlusionClaims.map(cloneOcclusionClaim),
            activeObjectId: this.activeObjectId,
            activeViewId: this.activeViewId,
            chartProjectionMode: this.chartProjectionMode,
        };
    }

    private restoreSceneSnapshot(snapshot: PaintSceneSnapshot) {
        this.viewportWidth = snapshot.viewportWidth;
        this.viewportHeight = snapshot.viewportHeight;
        this.views = snapshot.views.map(cloneView);
        this.objects = snapshot.objects.map(cloneObject);
        this.strokes = snapshot.strokes.map(cloneStroke);
        this.occlusionClaims = snapshot.occlusionClaims.map(cloneOcclusionClaim);
        this.activeObjectId = snapshot.activeObjectId;
        this.activeViewId = snapshot.activeViewId;
        this.chartProjectionMode = snapshot.chartProjectionMode;
        this.draftStroke = null;
        this.pendingStrokeUndoSnapshot = null;
        this.undoGroup = null;

        const activeView = this.activeView;
        if (activeView) {
            this.orbit.long = activeView.long;
            this.orbit.lat = activeView.lat;
            this.orbit.radius = activeView.radius;
            this.orbit.offset = vec3.fromValues(activeView.offset[0], activeView.offset[1], activeView.offset[2]);
        }

        this.meshVersion += 1;
    }

    private touchCharts(chartIds: Set<string>) {
        if (chartIds.size === 0) return;
        this.objects = this.objects.map(object => ({
            ...object,
            charts: object.charts.map(chart => chartIds.has(chart.id)
                ? {
                    ...chart,
                    depths: [...chart.depths],
                    coverage: [...chart.coverage],
                    paint: [...chart.paint],
                    seams: [...chart.seams],
                }
                : chart),
        }));
    }
}

function normalizeRenderOptions(
    options: boolean | PaintRenderOptions,
    depthPreview?: DepthBrushPreview,
): Required<PaintRenderOptions> {
    if (typeof options === "boolean") {
        return {
            showPaintSurface: false,
            showChartWireframe: options,
            showBrushLattice: !!depthPreview,
            depthPreview: depthPreview ?? null,
            strokeRenderMode: "paint-order",
        };
    }
    return {
        showPaintSurface: options.showPaintSurface ?? false,
        showChartWireframe: options.showChartWireframe ?? true,
        showBrushLattice: options.showBrushLattice ?? false,
        depthPreview: options.depthPreview ?? null,
        strokeRenderMode: options.strokeRenderMode ?? "surface",
    };
}

function createChart({
    objectId,
    sourceViewId,
    role,
    projectionMode,
    depth,
}: {
    objectId: string;
    sourceViewId: string;
    role: ChartRole;
    projectionMode: ChartProjectionMode;
    depth: number;
}): PaintChart {
    return {
        id: makeId("chart"),
        objectId,
        sourceViewId,
        role,
        projectionMode,
        width: CHART_RESOLUTION,
        height: CHART_RESOLUTION,
        depths: Array.from({ length: CHART_RESOLUTION * CHART_RESOLUTION }, () => depth),
        coverage: Array.from({ length: CHART_RESOLUTION * CHART_RESOLUTION }, () => 0),
        paint: Array.from({ length: CHART_RESOLUTION * CHART_RESOLUTION * 4 }, () => 0),
        seams: Array.from({ length: CHART_RESOLUTION * CHART_RESOLUTION }, () => false),
        createdAt: Date.now(),
    };
}

function cloneView(view: PaintView): PaintView {
    return {
        ...view,
        offset: [...view.offset] as Vec3,
        viewProjMat: [...view.viewProjMat],
        viewProjInvMat: [...view.viewProjInvMat],
        viewMat: [...view.viewMat],
        viewInvMat: [...view.viewInvMat],
    };
}

function cloneObject(object: PaintObject): PaintObject {
    return {
        ...object,
        charts: object.charts.map(cloneChart),
    };
}

function cloneChart(chart: PaintChart): PaintChart {
    return {
        ...chart,
        depths: [...chart.depths],
        coverage: [...chart.coverage],
        paint: [...chart.paint],
        seams: [...chart.seams],
    };
}

function cloneStroke(stroke: PaintStroke): PaintStroke {
    return {
        ...stroke,
        samples: stroke.samples.map(sample => ({
            ...sample,
            sourcePoint: { ...sample.sourcePoint },
            surfaceRef: {
                chartId: sample.surfaceRef.chartId,
                uv: { ...sample.surfaceRef.uv },
            },
        })),
        style: { ...stroke.style },
    };
}

function cloneOcclusionClaim(claim: OcclusionClaim): OcclusionClaim {
    return {
        ...claim,
        backRefs: claim.backRefs.map(ref => ({
            chartId: ref.chartId,
            uv: { ...ref.uv },
        })),
        mask: claim.mask.map(point => ({ ...point })),
    };
}

function resamplePaintPolyline(points: Vec2[], maxSamples: number): Vec2[] {
    if (points.length <= 1) return points.slice();
    const length = polylineLength(points);
    if (length <= 1e-6) return [points[0]];

    const sampleCount = Math.max(2, Math.min(maxSamples, Math.ceil(length / 0.015)));
    const spacing = length / (sampleCount - 1);
    const out: Vec2[] = [points[0]];

    let segmentStart = points[0];
    let segmentEndIndex = 1;
    let segmentEnd = points[segmentEndIndex];
    let segmentLength = distance2d(segmentStart, segmentEnd);
    let distanceIntoSegment = 0;

    for (let sampleIndex = 1; sampleIndex < sampleCount - 1; sampleIndex++) {
        const targetDistance = sampleIndex * spacing;

        while (distanceIntoSegment + segmentLength < targetDistance && segmentEndIndex < points.length - 1) {
            distanceIntoSegment += segmentLength;
            segmentStart = segmentEnd;
            segmentEndIndex += 1;
            segmentEnd = points[segmentEndIndex];
            segmentLength = distance2d(segmentStart, segmentEnd);
        }

        const local = segmentLength <= 1e-6
            ? 0
            : (targetDistance - distanceIntoSegment) / segmentLength;
        out.push({
            x: segmentStart.x + (segmentEnd.x - segmentStart.x) * local,
            y: segmentStart.y + (segmentEnd.y - segmentStart.y) * local,
        });
    }

    out.push(points[points.length - 1]);
    return out;
}

function samplePaintStrokeSpline(points: Vec2[]): Vec2[] {
    const length = polylineLength(points);

    if (points.length < MIN_STROKE_SPLINE_CONTROL_POINTS) {
        return resamplePaintPolyline(points, paintStrokePolylineSampleCount(length));
    }

    if (length <= 1e-6) return [points[0]];
    return sampleClampedCubicBSpline(points);
}

function paintStrokePolylineSampleCount(length: number): number {
    if (length <= 1e-6) return 1;
    return Math.max(2, Math.ceil(length / STROKE_SPLINE_SAMPLE_SPACING) + 1);
}

function sampleClampedCubicBSpline(controls: Vec2[]): Vec2[] {
    if (controls.length < MIN_STROKE_SPLINE_CONTROL_POINTS) {
        return resamplePaintPolyline(controls, paintStrokePolylineSampleCount(polylineLength(controls)));
    }

    const padded = [
        controls[0],
        controls[0],
        ...controls,
        controls[controls.length - 1],
        controls[controls.length - 1],
    ];
    const spanCount = Math.max(1, padded.length - 3);
    const samples: Vec2[] = [];

    for (let span = 0; span < spanCount; span++) {
        const p0 = padded[span];
        const p1 = padded[span + 1];
        const p2 = padded[span + 2];
        const p3 = padded[span + 3];
        const segmentCount = Math.max(
            1,
            Math.ceil(cubicBSplineSpanLength(p0, p1, p2, p3) / STROKE_SPLINE_SAMPLE_SPACING),
        );

        if (span === 0) {
            samples.push(cubicBSplinePoint(p0, p1, p2, p3, 0));
        }
        for (let i = 1; i <= segmentCount; i++) {
            samples.push(cubicBSplinePoint(p0, p1, p2, p3, i / segmentCount));
        }
    }

    samples[0] = controls[0];
    samples[samples.length - 1] = controls[controls.length - 1];
    return samples;
}

function cubicBSplineSpanLength(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2): number {
    const steps = 4;
    let length = 0;
    let previous = cubicBSplinePoint(p0, p1, p2, p3, 0);
    for (let i = 1; i <= steps; i++) {
        const current = cubicBSplinePoint(p0, p1, p2, p3, i / steps);
        length += distance2d(previous, current);
        previous = current;
    }
    return length;
}

function cubicBSplinePoint(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: number): Vec2 {
    const t2 = t * t;
    const t3 = t2 * t;
    const b0 = (-t3 + 3 * t2 - 3 * t + 1) / 6;
    const b1 = (3 * t3 - 6 * t2 + 4) / 6;
    const b2 = (-3 * t3 + 3 * t2 + 3 * t + 1) / 6;
    const b3 = t3 / 6;

    return {
        x: p0.x * b0 + p1.x * b1 + p2.x * b2 + p3.x * b3,
        y: p0.y * b0 + p1.y * b1 + p2.y * b2 + p3.y * b3,
    };
}

function polylineLength(points: Vec2[]): number {
    let length = 0;
    for (let i = 1; i < points.length; i++) {
        length += distance2d(points[i - 1], points[i]);
    }
    return length;
}

function distance2d(a: Vec2, b: Vec2): number {
    return Math.hypot(a.x - b.x, a.y - b.y);
}

function appendPaintRun(
    runs: ChartPaintRun[],
    chart: PaintChart,
    sample: ChartPaintSample,
    updateDepth: boolean,
    requireCoverage: boolean,
) {
    const previous = runs.at(-1);
    if (
        previous
        && previous.chart.id === chart.id
        && previous.updateDepth === updateDepth
        && previous.requireCoverage === requireCoverage
    ) {
        previous.samples.push(sample);
        return;
    }
    runs.push({ chart, samples: [sample], updateDepth, requireCoverage });
}

function appendUvRun(runs: ChartUvRun[], chart: PaintChart, point: Vec2) {
    const previous = runs.at(-1);
    if (previous?.chart.id === chart.id) {
        previous.points.push(point);
        return;
    }
    runs.push({ chart, points: [point] });
}

function editChartDepthAlongPolyline(chart: PaintChart, points: Vec2[], radius: number, delta: number): boolean {
    let changed = false;
    forEachGridPoint(chart, (index, uv) => {
        if (!isGridPointCovered(chart, index)) return;
        if (chart.seams[index]) return;
        const nearest = nearestUvOnPolyline(points, uv);
        if (!nearest || nearest.distance > radius) return;
        const t = nearest.distance / Math.max(radius, 1e-5);
        const influence = (1 - t * t) ** 2;
        const next = Math.max(MIN_DEPTH, chart.depths[index] + delta * influence);
        if (Math.abs(next - chart.depths[index]) > 1e-7) {
            chart.depths[index] = next;
            changed = true;
        }
    });
    return changed;
}

function editChartDepthAlongScreenPolyline(
    chart: PaintChart,
    sourceView: PaintView,
    editView: PaintView,
    points: Vec2[],
    radius: number,
    viewportWidth: number,
    viewportHeight: number,
    delta: number,
): boolean {
    let changed = false;
    forEachGridPoint(chart, (index, uv) => {
        if (!isGridPointCovered(chart, index)) return;
        if (chart.seams[index]) return;
        const world = chartPointToWorldFromView(chart, sourceView, uv);
        const projected = world ? projectVisiblePoint(editView.viewProjMat, world) : null;
        if (!projected) return;
        const nearest = nearestScreenPointOnPolyline(points, projected, viewportWidth, viewportHeight);
        if (!nearest || nearest.distance > radius) return;
        const t = nearest.distance / Math.max(radius, 1e-5);
        const influence = (1 - t * t) ** 2;
        const next = Math.max(MIN_DEPTH, chart.depths[index] + delta * influence);
        if (Math.abs(next - chart.depths[index]) > 1e-7) {
            chart.depths[index] = next;
            changed = true;
        }
    });
    return changed;
}

function applyStrokeToChartGeometry(
    chart: PaintChart,
    samples: ChartPaintSample[],
    radius: number,
    {
        updateDepth,
        requireCoverage,
    }: {
        updateDepth: boolean;
        requireCoverage: boolean;
    },
): boolean {
    if (samples.length === 0) return false;
    let changed = false;

    // Paint color stays in vector stroke samples. The chart grid stores only geometry coverage
    // and depth so brushstrokes do not get baked into a low-resolution color raster.
    forEachGridPoint(chart, (index, uv) => {
        if (requireCoverage && !isGridPointCovered(chart, index)) return;
        const nearest = nearestPaintSampleOnPolyline(samples, uv);
        if (!nearest || nearest.distance > radius) return;
        const t = nearest.distance / Math.max(radius, 1e-5);
        const influence = (1 - t * t) ** 2;

        if (updateDepth) {
            chart.depths[index] = lerp(chart.depths[index], Math.max(MIN_DEPTH, nearest.depth), influence);
            chart.coverage[index] = Math.max(chart.coverage[index] ?? 0, influence);
            changed = true;
        }
    });
    return changed;
}

function markChartSeamAlongPolyline(chart: PaintChart, points: Vec2[], radius: number): boolean {
    let changed = false;
    forEachGridPoint(chart, (index, uv) => {
        if (!isGridPointCovered(chart, index)) return;
        const nearest = nearestUvOnPolyline(points, uv);
        if (nearest && nearest.distance <= radius && !chart.seams[index]) {
            chart.seams[index] = true;
            changed = true;
        }
    });
    return changed;
}

function nearestUvOnPolyline(points: Vec2[], uv: Vec2): { distance: number } | null {
    if (points.length === 0) return null;
    if (points.length === 1) {
        return { distance: distance2d(points[0], uv) };
    }

    let bestDistance = Number.POSITIVE_INFINITY;
    for (let i = 1; i < points.length; i++) {
        const nearest = nearestPointOnSegment(uv, points[i - 1], points[i]);
        if (nearest.distance < bestDistance) {
            bestDistance = nearest.distance;
        }
    }
    return { distance: bestDistance };
}

function sampleBrushFootprint(
    points: Vec2[],
    radius: number,
    viewportWidth: number,
    viewportHeight: number,
    maxSamples: number,
): Vec2[] {
    const centers = resamplePaintPolyline(points, Math.max(1, Math.min(maxSamples, 8)));
    const samples: Vec2[] = [];
    for (const center of centers) {
        for (const offset of DEPTH_BRUSH_FOOTPRINT_OFFSETS) {
            samples.push(offsetBrushPoint(center, offset, radius, viewportWidth, viewportHeight));
        }
    }
    return samples;
}

function offsetBrushPoint(
    center: Vec2,
    offset: Vec2,
    radius: number,
    viewportWidth: number,
    viewportHeight: number,
): Vec2 {
    const minDimension = Math.max(1, Math.min(viewportWidth, viewportHeight));
    return {
        x: clamp(center.x + offset.x * radius * minDimension / Math.max(1, viewportWidth), -1, 1),
        y: clamp(center.y + offset.y * radius * minDimension / Math.max(1, viewportHeight), -1, 1),
    };
}

function nearestScreenPointOnPolyline(
    points: Vec2[],
    point: Vec2,
    viewportWidth: number,
    viewportHeight: number,
): { distance: number } | null {
    if (points.length === 0) return null;
    const scaledPoint = pointToBrushMetric(point, viewportWidth, viewportHeight);
    if (points.length === 1) {
        return { distance: distance2d(pointToBrushMetric(points[0], viewportWidth, viewportHeight), scaledPoint) };
    }

    let bestDistance = Number.POSITIVE_INFINITY;
    for (let i = 1; i < points.length; i++) {
        const nearest = nearestPointOnSegment(
            scaledPoint,
            pointToBrushMetric(points[i - 1], viewportWidth, viewportHeight),
            pointToBrushMetric(points[i], viewportWidth, viewportHeight),
        );
        bestDistance = Math.min(bestDistance, nearest.distance);
    }
    return { distance: bestDistance };
}

function pointToBrushMetric(point: Vec2, viewportWidth: number, viewportHeight: number): Vec2 {
    const minDimension = Math.max(1, Math.min(viewportWidth, viewportHeight));
    return {
        x: point.x * viewportWidth / minDimension,
        y: point.y * viewportHeight / minDimension,
    };
}

function nearestPaintSampleOnPolyline(
    samples: ChartPaintSample[],
    uv: Vec2,
): { distance: number; depth: number } | null {
    if (samples.length === 0) return null;
    if (samples.length === 1) {
        return {
            distance: distance2d(samples[0].point, uv),
            depth: samples[0].depth,
        };
    }

    let bestDistance = Number.POSITIVE_INFINITY;
    let bestDepth = samples[0].depth;
    for (let i = 1; i < samples.length; i++) {
        const previous = samples[i - 1];
        const current = samples[i];
        const nearest = nearestPointOnSegment(uv, previous.point, current.point);
        if (nearest.distance < bestDistance) {
            bestDistance = nearest.distance;
            bestDepth = lerp(previous.depth, current.depth, nearest.t);
        }
    }
    return { distance: bestDistance, depth: bestDepth };
}

function nearestPointOnSegment(point: Vec2, a: Vec2, b: Vec2): { distance: number; t: number } {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const lengthSquared = dx * dx + dy * dy;
    const t = lengthSquared <= 1e-10
        ? 0
        : clamp(((point.x - a.x) * dx + (point.y - a.y) * dy) / lengthSquared, 0, 1);
    return {
        t,
        distance: Math.hypot(point.x - (a.x + dx * t), point.y - (a.y + dy * t)),
    };
}

function sampleChartDepth(chart: PaintChart, uv: Vec2): number {
    const x = clamp((uv.x * 0.5 + 0.5) * (chart.width - 1), 0, chart.width - 1);
    const y = clamp((uv.y * 0.5 + 0.5) * (chart.height - 1), 0, chart.height - 1);
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const x1 = Math.min(chart.width - 1, x0 + 1);
    const y1 = Math.min(chart.height - 1, y0 + 1);
    const fx = x - x0;
    const fy = y - y0;
    const a = chart.depths[y0 * chart.width + x0];
    const b = chart.depths[y0 * chart.width + x1];
    const c = chart.depths[y1 * chart.width + x0];
    const d = chart.depths[y1 * chart.width + x1];
    return lerp(lerp(a, b, fx), lerp(c, d, fx), fy);
}

function chartPointToWorldFromView(chart: PaintChart, view: PaintView, uv: Vec2): Vec3 | null {
    const ray = makeViewRay(view, uv);
    if (!ray) return null;
    const depth = sampleChartDepth(chart, uv);
    if ((chart.projectionMode ?? "ray-depth") === "ray-depth") {
        return add3(ray.origin, scale3(ray.direction, depth));
    }
    const normal = viewForward(view);
    const denominator = dot3(ray.direction, normal);
    if (Math.abs(denominator) <= 1e-6) return null;
    const rayDistance = depth / denominator;
    if (!Number.isFinite(rayDistance) || rayDistance <= MIN_DEPTH) return null;
    return add3(ray.origin, scale3(ray.direction, rayDistance));
}

function viewPointToWorldAtProjectionDepth(
    view: PaintView,
    point: Vec2,
    depth: number,
    projectionMode: ChartProjectionMode,
): Vec3 | null {
    const ray = makeViewRay(view, point);
    if (!ray) return null;
    if (projectionMode === "ray-depth") {
        return add3(ray.origin, scale3(ray.direction, Math.max(MIN_DEPTH, depth)));
    }
    const denominator = dot3(ray.direction, viewForward(view));
    if (Math.abs(denominator) <= 1e-6) return null;
    const rayDistance = Math.max(MIN_DEPTH, depth) / denominator;
    if (!Number.isFinite(rayDistance) || rayDistance <= MIN_DEPTH) return null;
    return add3(ray.origin, scale3(ray.direction, rayDistance));
}

function depthForProjectionAtPoint(
    view: PaintView,
    point: Vec2,
    rayDistance: number,
    projectionMode: ChartProjectionMode,
): number {
    if (projectionMode === "ray-depth") return rayDistance;
    const ray = makeViewRay(view, point);
    if (!ray) return rayDistance;
    return Math.max(MIN_DEPTH, rayDistance * dot3(ray.direction, viewForward(view)));
}

function chartHasCoverage(chart: PaintChart): boolean {
    return chart.coverage.some(value => value > COVERAGE_EPSILON);
}

function chartHasPaint(chart: PaintChart): boolean {
    for (let i = 3; i < chart.paint.length; i += 4) {
        if ((chart.paint[i] ?? 0) > PAINT_EPSILON) return true;
    }
    return false;
}

function isGridPointCovered(chart: PaintChart, index: number): boolean {
    return (chart.coverage[index] ?? 0) > COVERAGE_EPSILON;
}

function isGridEdgeCovered(chart: PaintChart, a: number, b: number): boolean {
    return isGridPointCovered(chart, a) && isGridPointCovered(chart, b);
}

function isGridTriangleCovered(chart: PaintChart, a: number, b: number, c: number): boolean {
    return isGridPointCovered(chart, a) && isGridPointCovered(chart, b) && isGridPointCovered(chart, c);
}

function forEachGridPoint(chart: PaintChart, fn: (index: number, uv: Vec2) => void) {
    for (let y = 0; y < chart.height; y++) {
        for (let x = 0; x < chart.width; x++) {
            fn(y * chart.width + x, gridUv(chart, x, y));
        }
    }
}

function gridUv(chart: PaintChart, x: number, y: number): Vec2 {
    return {
        x: chart.width <= 1 ? 0 : x / (chart.width - 1) * 2 - 1,
        y: chart.height <= 1 ? 0 : y / (chart.height - 1) * 2 - 1,
    };
}

function paintTriangleColor(chart: PaintChart, a: number, b: number, c: number): Vec4 | null {
    const alphaA = chart.paint[a * 4 + 3] ?? 0;
    const alphaB = chart.paint[b * 4 + 3] ?? 0;
    const alphaC = chart.paint[c * 4 + 3] ?? 0;
    const alpha = (alphaA + alphaB + alphaC) / 3;
    const alphaWeight = alphaA + alphaB + alphaC;
    if (alpha <= PAINT_EPSILON || alphaWeight <= PAINT_EPSILON) return null;
    return [
        ((chart.paint[a * 4] ?? 0) * alphaA + (chart.paint[b * 4] ?? 0) * alphaB + (chart.paint[c * 4] ?? 0) * alphaC) / alphaWeight,
        ((chart.paint[a * 4 + 1] ?? 0) * alphaA + (chart.paint[b * 4 + 1] ?? 0) * alphaB + (chart.paint[c * 4 + 1] ?? 0) * alphaC) / alphaWeight,
        ((chart.paint[a * 4 + 2] ?? 0) * alphaA + (chart.paint[b * 4 + 2] ?? 0) * alphaB + (chart.paint[c * 4 + 2] ?? 0) * alphaC) / alphaWeight,
        clamp(alpha, 0, 1),
    ];
}

function appendOrientationSegments(segments: RenderPrimitive[]) {
    const steps = Math.round(GRID_EXTENT / GRID_STEP);
    for (let i = -steps; i <= steps; i++) {
        const position = i * GRID_STEP;
        const isMajor = i === 0 || Math.abs(i) % 4 === 0;
        const color = isMajor
            ? [0.58, 0.66, 0.66, 0.2] as Vec4
            : [0.58, 0.66, 0.66, 0.09] as Vec4;
        const width = isMajor ? 1.1 : 0.75;
        segments.push({
            a: [position, -GRID_EXTENT, GRID_PLANE_Z],
            b: [position, GRID_EXTENT, GRID_PLANE_Z],
            color,
            width,
        });
        segments.push({
            a: [-GRID_EXTENT, position, GRID_PLANE_Z],
            b: [GRID_EXTENT, position, GRID_PLANE_Z],
            color,
            width,
        });
    }

    const axisLength = GRID_EXTENT * 0.86;
    segments.push({ a: [-axisLength, 0, 0], b: [axisLength, 0, 0], color: [0.92, 0.42, 0.38, 0.62], width: 2.15 });
    segments.push({ a: [0, -axisLength, 0], b: [0, axisLength, 0], color: [0.48, 0.82, 0.55, 0.62], width: 2.15 });
    segments.push({ a: [0, 0, -axisLength * 0.18], b: [0, 0, axisLength], color: [0.48, 0.62, 0.9, 0.62], width: 2.15 });
}

function appendPaintTriangles(
    segments: RenderPrimitive[],
    chart: PaintChart,
    worldAt: (uv: Vec2) => Vec3 | null,
) {
    if (!chartHasPaint(chart)) return;

    for (let y = 1; y < chart.height; y++) {
        for (let x = 1; x < chart.width; x++) {
            const i00 = (y - 1) * chart.width + x - 1;
            const i10 = (y - 1) * chart.width + x;
            const i01 = y * chart.width + x - 1;
            const i11 = y * chart.width + x;

            if (isGridTriangleCovered(chart, i00, i10, i11)) {
                appendPaintTriangle(
                    segments,
                    worldAt(gridUv(chart, x - 1, y - 1)),
                    worldAt(gridUv(chart, x, y - 1)),
                    worldAt(gridUv(chart, x, y)),
                    paintTriangleColor(chart, i00, i10, i11),
                );
            }
            if (isGridTriangleCovered(chart, i00, i11, i01)) {
                appendPaintTriangle(
                    segments,
                    worldAt(gridUv(chart, x - 1, y - 1)),
                    worldAt(gridUv(chart, x, y)),
                    worldAt(gridUv(chart, x - 1, y)),
                    paintTriangleColor(chart, i00, i11, i01),
                );
            }
        }
    }
}

function appendChartSegments(
    segments: RenderPrimitive[],
    chart: PaintChart,
    worldAt: (uv: Vec2) => Vec3 | null,
) {
    if (!chartHasCoverage(chart)) return;

    const color = chart.role === "occluder"
        ? [1, 0.48, 0.32, 0.38] as Vec4
        : chart.role === "behind"
            ? [0.46, 0.55, 1, 0.24] as Vec4
            : [0.44, 0.92, 0.82, 0.18] as Vec4;
    const stride = 4;

    for (let y = 0; y < chart.height; y += stride) {
        for (let x = 1; x < chart.width; x++) {
            const aIndex = y * chart.width + x - 1;
            const bIndex = y * chart.width + x;
            if (!isGridEdgeCovered(chart, aIndex, bIndex)) continue;
            appendWorldSegment(segments, worldAt(gridUv(chart, x - 1, y)), worldAt(gridUv(chart, x, y)), color, 1.15);
        }
    }
    for (let x = 0; x < chart.width; x += stride) {
        for (let y = 1; y < chart.height; y++) {
            const aIndex = (y - 1) * chart.width + x;
            const bIndex = y * chart.width + x;
            if (!isGridEdgeCovered(chart, aIndex, bIndex)) continue;
            appendWorldSegment(segments, worldAt(gridUv(chart, x, y - 1)), worldAt(gridUv(chart, x, y)), color, 1.15);
        }
    }

    forEachGridPoint(chart, (index, uv) => {
        if (!chart.seams[index]) return;
        const a = worldAt({ x: uv.x - 0.018, y: uv.y - 0.018 });
        const b = worldAt({ x: uv.x + 0.018, y: uv.y + 0.018 });
        const c = worldAt({ x: uv.x - 0.018, y: uv.y + 0.018 });
        const d = worldAt({ x: uv.x + 0.018, y: uv.y - 0.018 });
        appendWorldSegment(segments, a, b, [1, 0.16, 0.16, 0.95], 2.4);
        appendWorldSegment(segments, c, d, [1, 0.16, 0.16, 0.95], 2.4);
    });
}

function appendDepthBrushPreview(
    segments: RenderPrimitive[],
    chart: PaintChart,
    sourceView: PaintView,
    editView: PaintView,
    points: Vec2[],
    radius: number,
    viewportWidth: number,
    viewportHeight: number,
    delta: number,
    tool: DepthTool,
) {
    if (!chartHasCoverage(chart) || points.length === 0) return;
    appendDepthBrushInfluenceTrails(segments, chart, sourceView, editView, points, radius, viewportWidth, viewportHeight, delta, tool);
}

function appendDepthBrushRing(
    segments: RenderPrimitive[],
    worldAt: (uv: Vec2) => Vec3 | null,
    center: Vec2,
    radius: number,
    color: Vec4,
) {
    const steps = 72;
    let previousUv: Vec2 | null = null;
    let previousWorld: Vec3 | null = null;

    for (let i = 0; i <= steps; i++) {
        const angle = i / steps * Math.PI * 2;
        const uv = {
            x: center.x + Math.cos(angle) * radius,
            y: center.y + Math.sin(angle) * radius,
        };
        const inBounds = isUvInBounds(uv);
        const world = inBounds ? worldAt(uv) : null;
        if (previousUv && previousWorld && world) {
            appendWorldSegment(segments, previousWorld, world, color, DEPTH_PREVIEW_RING_WIDTH);
        }
        previousUv = inBounds ? uv : null;
        previousWorld = world;
    }
}

function appendDepthBrushInfluenceTrails(
    segments: RenderPrimitive[],
    chart: PaintChart,
    sourceView: PaintView,
    editView: PaintView,
    points: Vec2[],
    radius: number,
    viewportWidth: number,
    viewportHeight: number,
    delta: number,
    tool: DepthTool,
) {
    const stride = Math.max(1, Math.round(chart.width / 24));
    const isDragging = Math.abs(delta) > 1e-6;

    for (let y = 0; y < chart.height; y += stride) {
        for (let x = 0; x < chart.width; x += stride) {
            const index = y * chart.width + x;
            if (!isGridPointCovered(chart, index)) continue;
            if (chart.seams[index]) continue;

            const uv = gridUv(chart, x, y);
            const world = chartPointToWorldFromView(chart, sourceView, uv);
            const projected = world ? projectVisiblePoint(editView.viewProjMat, world) : null;
            const nearest = projected
                ? nearestScreenPointOnPolyline(points, projected, viewportWidth, viewportHeight)
                : null;
            if (!nearest || nearest.distance > radius) continue;

            const t = nearest.distance / Math.max(radius, 1e-5);
            const influence = (1 - t * t) ** 2;
            const ray = makeViewRay(sourceView, uv);
            if (!world || !ray) continue;

            const color = depthPreviewColor(tool, delta, 0.35 + influence * 0.58);
            if (isDragging) {
                const appliedDelta = delta * influence;
                const before = add3(world, scale3(ray.direction, -appliedDelta));
                appendWorldSegment(segments, before, world, color, DEPTH_PREVIEW_TRAIL_WIDTH);
            } else {
                const tip = add3(world, scale3(ray.direction, DEPTH_PREVIEW_IDLE_LENGTH * (0.35 + influence * 0.65)));
                appendWorldSegment(segments, world, tip, color, DEPTH_PREVIEW_TRAIL_WIDTH);
            }
        }
    }
}

function depthPreviewColor(tool: DepthTool, delta: number, alpha: number): Vec4 {
    if (Math.abs(delta) <= 1e-6) {
        return tool === "depth-pull"
            ? [1, 0.88, 0.36, alpha]
            : [0.5, 0.78, 1, alpha];
    }
    if (tool === "depth-pull") {
        return delta > 0
            ? [1, 0.72, 0.16, alpha]
            : [0.2, 0.6, 1, alpha];
    }
    return delta > 0
        ? [0.18, 0.68, 1, alpha]
        : [1, 0.78, 0.18, alpha];
}

function isUvInBounds(uv: Vec2): boolean {
    return uv.x >= -1 && uv.x <= 1 && uv.y >= -1 && uv.y <= 1;
}

function appendPaintTriangle(
    segments: RenderPrimitive[],
    a: Vec3 | null,
    b: Vec3 | null,
    c: Vec3 | null,
    color: Vec4 | null,
) {
    if (!a || !b || !c || !color) return;
    segments.push({ kind: "triangle", a, b, c, color });
}

function appendStrokeRenderSegments(
    segments: RenderPrimitive[],
    stroke: PaintStroke,
    worldPointForRef: (ref: SurfaceRef) => Vec3 | null,
) {
    const color = parseColor(stroke.style.color, stroke.style.opacity);
    let previous = stroke.samples.length > 0
        ? worldPointForRef(stroke.samples[0].surfaceRef)
        : null;
    for (let i = 1; i < stroke.samples.length; i++) {
        const current = worldPointForRef(stroke.samples[i].surfaceRef);
        appendWorldSegment(segments, previous, current, color, stroke.style.width);
        previous = current;
    }
}

function appendWorldSegment(segments: RenderPrimitive[], a: Vec3 | null, b: Vec3 | null, color: Vec4, width?: number) {
    if (!a || !b) return;
    segments.push({ a, b, color, width });
}

function raycastChart(
    chart: PaintChart,
    ray: { origin: Vec3; direction: Vec3 },
    worldAt: (uv: Vec2) => Vec3 | null,
): Array<{ uv: Vec2; world: Vec3; t: number }> {
    const hits: Array<{ uv: Vec2; world: Vec3; t: number }> = [];
    const uvs = new Array<Vec2>(chart.width * chart.height);
    const worlds = new Array<Vec3 | null | undefined>(chart.width * chart.height);

    for (let y = 0; y < chart.height; y++) {
        for (let x = 0; x < chart.width; x++) {
            const index = y * chart.width + x;
            uvs[index] = gridUv(chart, x, y);
        }
    }

    const worldAtIndex = (index: number): Vec3 | null => {
        if (worlds[index] !== undefined) return worlds[index];
        const world = worldAt(uvs[index]);
        worlds[index] = world;
        return world;
    };

    for (let y = 1; y < chart.height; y++) {
        for (let x = 1; x < chart.width; x++) {
            const i00 = (y - 1) * chart.width + x - 1;
            const i10 = (y - 1) * chart.width + x;
            const i01 = y * chart.width + x - 1;
            const i11 = y * chart.width + x;
            const uv00 = uvs[i00];
            const uv10 = uvs[i10];
            const uv01 = uvs[i01];
            const uv11 = uvs[i11];
            if (isGridTriangleCovered(chart, i00, i10, i11)) {
                const p00 = worldAtIndex(i00);
                const p10 = worldAtIndex(i10);
                const p11 = worldAtIndex(i11);
                if (p00 && p10 && p11) {
                    appendTriangleHit(hits, ray, p00, p10, p11, uv00, uv10, uv11);
                }
            }
            if (isGridTriangleCovered(chart, i00, i11, i01)) {
                const p00 = worldAtIndex(i00);
                const p11 = worldAtIndex(i11);
                const p01 = worldAtIndex(i01);
                if (p00 && p11 && p01) {
                    appendTriangleHit(hits, ray, p00, p11, p01, uv00, uv11, uv01);
                }
            }
        }
    }
    return hits.sort((a, b) => a.t - b.t);
}

function appendTriangleHit(
    hits: Array<{ uv: Vec2; world: Vec3; t: number }>,
    ray: { origin: Vec3; direction: Vec3 },
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    uv0: Vec2,
    uv1: Vec2,
    uv2: Vec2,
) {
    const hit = intersectRayTriangle(ray.origin, ray.direction, p0, p1, p2);
    if (!hit) return;
    const w0 = 1 - hit.u - hit.v;
    hits.push({
        t: hit.t,
        world: add3(ray.origin, scale3(ray.direction, hit.t)),
        uv: {
            x: uv0.x * w0 + uv1.x * hit.u + uv2.x * hit.v,
            y: uv0.y * w0 + uv1.y * hit.u + uv2.y * hit.v,
        },
    });
}

function intersectRayTriangle(
    origin: Vec3,
    direction: Vec3,
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
): { t: number; u: number; v: number } | null {
    const edge1 = sub3(p1, p0);
    const edge2 = sub3(p2, p0);
    const h = cross3(direction, edge2);
    const a = dot3(edge1, h);
    if (Math.abs(a) < 1e-7) return null;
    const f = 1 / a;
    const s = sub3(origin, p0);
    const u = f * dot3(s, h);
    if (u < 0 || u > 1) return null;
    const q = cross3(s, edge1);
    const v = f * dot3(direction, q);
    if (v < 0 || u + v > 1) return null;
    const t = f * dot3(edge2, q);
    if (!Number.isFinite(t) || t <= 1e-5) return null;
    return { t, u, v };
}

function makeViewRay(view: PaintView, point: Vec2): { origin: Vec3; direction: Vec3 } | null {
    const near = unprojectNdc(view.viewProjInvMat, point.x, point.y, 0.02);
    const far = unprojectNdc(view.viewProjInvMat, point.x, point.y, 0.98);
    if (!near || !far) return null;
    return {
        origin: cameraCenter(view),
        direction: normalize3(sub3(far, near), [0, 0, -1]),
    };
}

function cameraCenter(view: PaintView): Vec3 {
    return [view.viewInvMat[12], view.viewInvMat[13], view.viewInvMat[14]];
}

function viewForward(view: PaintView): Vec3 {
    return makeViewRay(view, { x: 0, y: 0 })?.direction ?? [0, 0, -1];
}

function unprojectNdc(invViewProjMat: number[], x: number, y: number, z: number): Vec3 | null {
    const worldX = invViewProjMat[0] * x + invViewProjMat[4] * y + invViewProjMat[8] * z + invViewProjMat[12];
    const worldY = invViewProjMat[1] * x + invViewProjMat[5] * y + invViewProjMat[9] * z + invViewProjMat[13];
    const worldZ = invViewProjMat[2] * x + invViewProjMat[6] * y + invViewProjMat[10] * z + invViewProjMat[14];
    const worldW = invViewProjMat[3] * x + invViewProjMat[7] * y + invViewProjMat[11] * z + invViewProjMat[15];
    if (!Number.isFinite(worldW) || Math.abs(worldW) <= 1e-6) return null;
    return [worldX / worldW, worldY / worldW, worldZ / worldW];
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

function projectedDepth(viewProjMat: number[] | Float32Array, p: Vec3): number | null {
    const clipZ = viewProjMat[2] * p[0] + viewProjMat[6] * p[1] + viewProjMat[10] * p[2] + viewProjMat[14];
    const clipW = viewProjMat[3] * p[0] + viewProjMat[7] * p[1] + viewProjMat[11] * p[2] + viewProjMat[15];
    if (!Number.isFinite(clipW) || Math.abs(clipW) <= 1e-6) return null;
    const depth = clipZ / clipW;
    return Number.isFinite(depth) ? depth : null;
}

function parseColor(color: string, opacity: number): Vec4 {
    const hex = color.startsWith("#") ? color.slice(1) : color;
    if (hex.length === 6) {
        const r = Number.parseInt(hex.slice(0, 2), 16) / 255;
        const g = Number.parseInt(hex.slice(2, 4), 16) / 255;
        const b = Number.parseInt(hex.slice(4, 6), 16) / 255;
        if ([r, g, b].every(Number.isFinite)) return [r, g, b, opacity];
    }
    return [1, 1, 1, opacity];
}

function add3(a: Vec3, b: Vec3): Vec3 {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function sub3(a: Vec3, b: Vec3): Vec3 {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale3(v: Vec3, scale: number): Vec3 {
    return [v[0] * scale, v[1] * scale, v[2] * scale];
}

function dot3(a: Vec3, b: Vec3): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross3(a: Vec3, b: Vec3): Vec3 {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
}

function distance3(a: Vec3, b: Vec3): number {
    return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function normalize3(v: Vec3, fallback: Vec3): Vec3 {
    const length = Math.hypot(v[0], v[1], v[2]);
    if (!Number.isFinite(length) || length <= 1e-8) return [...fallback] as Vec3;
    return [v[0] / length, v[1] / length, v[2] / length];
}

function lerp(a: number, b: number, t: number): number {
    return a + (b - a) * t;
}

function clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value));
}

function makeId(prefix: string): string {
    return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}
