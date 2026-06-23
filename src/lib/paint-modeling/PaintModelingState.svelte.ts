import { Camera } from "../viewer/Camera.svelte.ts";
import { CameraOrbit } from "../viewer/CameraOrbit.svelte.ts";
import {
    defaultDepthForPaintView,
    getOrCreatePaintChart,
    paintDepthRadiusForView,
    raycastPaintObjectSurfaceBatchWithViews,
    raycastPaintObjectSurfaceWithViews,
    raycastPaintObjectSurfacesWithViews,
} from "./state/chartAccess.ts";
import type { ChartPaintRun } from "./state/chartPainting.ts";
import { touchPaintCharts } from "./state/chartMutation.ts";
import {
    MAX_BRUSH_WIDTH,
    MIN_BRUSH_WIDTH,
} from "./state/constants.ts";
import {
    BASE_PAINT_LAYER_ID,
    createBasePaintLayer,
    createPaintLayer,
} from "./state/paintLayers.ts";
import { makeId } from "./state/sceneData.ts";
import {
    cameraMovedFromPaintView,
    capturePaintView,
    deletePaintObject,
    deletePaintView,
    selectPaintView,
    viewHasAuthoredContent,
} from "./state/sceneLifecycle.ts";
import {
    capturePaintSceneSnapshot,
    restorePaintSceneSnapshot,
    type PaintSceneSnapshot,
} from "./state/sceneHistory.ts";
import {
    buildDraftPaintRenderSegments,
    buildPaintRenderSegments,
    paintSurfaceRefWorldPoint,
    projectPaintSurfaceRef,
    type RenderAssemblyContext,
} from "./state/renderAssembly.ts";
import { planPaintChartSeams } from "./state/seamEditing.ts";
import { planFinishedStroke } from "./state/strokeSession.ts";
import { samplePaintStrokeSpline } from "./state/strokeSampling.ts";
import type {
    SnapPlacementPlan,
    StrokePlacementContext,
} from "./state/strokePlacement.ts";
import { PaintSurfaceRaycastCache } from "./state/surfaceRaycastCache.ts";
import { clamp } from "./state/vectorMath.ts";
import type {
    BrushStyle,
    BrushMode,
    ChartRole,
    ChartProjectionMode,
    OcclusionClaim,
    PaintChart,
    PaintLayer,
    PaintObject,
    PaintStroke,
    PaintView,
    PaintRenderOptions,
    PlacementMode,
    StrokeGeometryMode,
    RenderPrimitive,
    SurfaceHit,
    SurfaceRef,
    Vec2,
    Vec3,
} from "./types.ts";


const DEFAULT_COLOR_BRUSH: BrushStyle = {
    color: "#ffd27a",
    width: 18,
    opacity: 1,
    geometryMode: "billboard",
};
const DEFAULT_BRUSH_WIDTH_BY_MODE: Record<BrushMode, number> = {
    color: DEFAULT_COLOR_BRUSH.width,
    surface: Math.min(MAX_BRUSH_WIDTH, DEFAULT_COLOR_BRUSH.width * 4),
    depth: Math.min(MAX_BRUSH_WIDTH, DEFAULT_COLOR_BRUSH.width * 2),
};

export class PaintModelingState {
    viewportWidth = $state(1);
    viewportHeight = $state(1);

    views = $state<PaintView[]>([]);
    objects = $state<PaintObject[]>([]);
    paintLayers = $state<PaintLayer[]>([createBasePaintLayer()]);
    strokes = $state<PaintStroke[]>([]);
    occlusionClaims = $state<OcclusionClaim[]>([]);
    activeObjectId = $state<string | null>(null);
    activeViewId = $state<string | null>(null);
    activePaintLayerId = $state(BASE_PAINT_LAYER_ID);
    placementMode = $state<PlacementMode>("snap");
    chartProjectionMode = $state<ChartProjectionMode>("view-plane");
    brushMode = $state<BrushMode>("color");
    brush = $state<BrushStyle>({ ...DEFAULT_COLOR_BRUSH });
    draftStroke = $state<Vec2[] | null>(null);
    undoStack = $state<PaintSceneSnapshot[]>([]);
    meshVersion = $state(0);
    raycastCountForDiagnostics = 0;
    raycastCacheBuildCountForDiagnostics = 0;
    private readonly surfaceRaycastCache = new PaintSurfaceRaycastCache(() => {
        this.raycastCacheBuildCountForDiagnostics += 1;
    });
    private brushWidthByMode: Record<BrushMode, number> = { ...DEFAULT_BRUSH_WIDTH_BY_MODE };
    private pendingStrokeUndoSnapshot: PaintSceneSnapshot | null = null;
    private pendingStrokeView: PaintView | null = null;
    private pendingGpuChartPaintRuns: ChartPaintRun[] = [];
    private undoGroup: { snapshot: PaintSceneSnapshot; dirty: boolean } | null = null;

    readonly orbit = new CameraOrbit();
    readonly camera = new Camera({
        controlScheme: this.orbit,
        projectionAspect: "screen",
        screenDims: {
            width: () => this.viewportWidth,
            height: () => this.viewportHeight,
        },
    });

    get activeObject(): PaintObject | null { return this.objects.find(object => object.id === this.activeObjectId) ?? null; }

    get activeView(): PaintView | null { return this.views.find(view => view.id === this.activeViewId) ?? null; }

    get activePaintLayer(): PaintLayer | null {
        return this.paintLayers.find(layer => layer.id === this.activePaintLayerId)
            ?? this.paintLayers[0]
            ?? null;
    }

    get renderDepthSortKey(): string {
        const offset = Array.from(this.orbit.offset).slice(0, 3);
        return [
            this.viewportWidth,
            this.viewportHeight,
            this.orbit.long.toFixed(4),
            this.orbit.lat.toFixed(4),
            this.orbit.radius.toFixed(4),
            offset.map(value => value.toFixed(4)).join(","),
        ].join(":");
    }

    get isCameraAtActiveView(): boolean {
        const view = this.activeView;
        return !!view
            && !this.cameraMovedFrom(view)
            && view.width === this.viewportWidth
            && view.height === this.viewportHeight;
    }

    get currentViewName(): string {
        if (this.isCameraAtActiveView) return this.activeView?.name ?? "No saved view";
        return this.activeView ? "New view" : "No saved view";
    }

    get chartCount(): number { return this.objects.reduce((sum, object) => sum + object.charts.length, 0); }

    get seamCount(): number {
        return this.objects.reduce(
            (sum, object) => sum + object.charts.reduce((chartSum, chart) => chartSum + chart.seams.filter(Boolean).length, 0),
            0,
        );
    }

    get canUndo(): boolean { return this.undoStack.length > 0; }

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
        if (this.objects.some(object => object.id === objectId)) this.activeObjectId = objectId;
    }

    addPaintLayer(recordHistory = true): PaintLayer {
        if (recordHistory) this.recordUndoSnapshot();
        const layer = createPaintLayer(this.nextPaintLayerOrder());
        this.paintLayers = [...this.paintLayers, layer];
        this.activePaintLayerId = layer.id;
        return layer;
    }

    selectPaintLayer(layerId: string) {
        if (this.paintLayers.some(layer => layer.id === layerId)) this.activePaintLayerId = layerId;
    }

    setPlacementMode(mode: PlacementMode) { this.placementMode = mode; }

    setBrushMode(mode: BrushMode) {
        this.brushWidthByMode[this.brushMode] = this.brush.width;
        this.brushMode = mode;
        this.brush.width = this.brushWidthByMode[mode];
    }

    setChartProjectionMode(mode: ChartProjectionMode) { this.chartProjectionMode = mode; }

    setBrushColor(color: string) { this.brush.color = color; }

    setBrushGeometryMode(mode: StrokeGeometryMode) { this.brush.geometryMode = mode; }

    setBrushWidth(width: number) {
        const clampedWidth = clamp(width, MIN_BRUSH_WIDTH, MAX_BRUSH_WIDTH);
        this.brushWidthByMode[this.brushMode] = clampedWidth;
        this.brush.width = clampedWidth;
    }

    setBrushOpacity(_opacity: number) { this.brush.opacity = 1; }

    resetDiagnostics() {
        this.raycastCountForDiagnostics = 0;
        this.raycastCacheBuildCountForDiagnostics = 0;
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

    cancelUndoGroup() { this.undoGroup = null; }

    consumeGpuChartPaintRuns(): ChartPaintRun[] {
        const runs = this.pendingGpuChartPaintRuns;
        this.pendingGpuChartPaintRuns = [];
        return runs;
    }

    beginStroke(point: Vec2, width: number, height: number) {
        const undoSnapshot = this.captureSceneSnapshot();
        this.viewportWidth = Math.max(1, width);
        this.viewportHeight = Math.max(1, height);
        if (!this.activeObjectId && this.brushMode !== "depth") this.addObject(undefined, false);
        const object = this.activeObject;
        if (!object || object.locked || !object.visible) {
            this.pendingStrokeUndoSnapshot = null;
            this.pendingStrokeView = null;
            return;
        }

        if (this.brushMode === "depth") {
            this.pendingStrokeView = this.currentEffectView();
        } else {
            this.ensureActiveView(width, height, false);
            this.pendingStrokeView = this.activeView;
        }
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

    draftStrokeSourcePoints(): Vec2[] | null {
        const object = this.activeObject;
        if (!this.draftStroke || this.draftStroke.length < 2 || !object || object.locked || !object.visible || !this.activeView) {
            return null;
        }
        return samplePaintStrokeSpline(this.draftStroke);
    }

    finishStroke(options: { snapPlacementPlan?: SnapPlacementPlan } = {}) {
        const result = planFinishedStroke({
            draftStroke: this.draftStroke,
            pendingStrokeUndoSnapshot: this.pendingStrokeUndoSnapshot,
            undoSnapshot: this.pendingStrokeUndoSnapshot ?? this.captureSceneSnapshot(),
            object: this.activeObject,
            view: this.pendingStrokeView ?? this.activeView,
            brushMode: this.brushMode,
            placementMode: this.placementMode,
            brush: this.brush,
            placementContext: this.strokePlacementContext(),
            nextPaintOrder: objectId => this.nextPaintOrder(objectId),
            paintLayerId: this.activePaintLayer?.id ?? BASE_PAINT_LAYER_ID,
            snapPlacementPlan: options.snapPlacementPlan,
        });

        this.draftStroke = null;
        this.pendingStrokeUndoSnapshot = null;
        this.pendingStrokeView = null;
        if (result.kind === "discard") {
            if (result.restoreSnapshot) this.restoreSceneSnapshot(result.restoreSnapshot);
            return;
        }

        this.pushUndoSnapshot(result.undoSnapshot);
        this.pendingGpuChartPaintRuns.push(...result.gpuChartPaintRuns);
        this.touchCharts(result.touchedChartIds);
        if (result.kind === "stroke") {
            this.strokes = [...this.strokes, result.stroke];
            if (result.occlusionClaim) {
                this.occlusionClaims = [...this.occlusionClaims, result.occlusionClaim];
            }
        }
        this.meshVersion += 1;
    }

    undo(): boolean {
        const snapshot = this.undoStack.at(-1);
        if (!snapshot) return false;
        this.undoStack = this.undoStack.slice(0, -1);
        this.restoreSceneSnapshot(snapshot);
        return true;
    }

    undoStroke(): boolean { return this.undo(); }

    deleteActiveObject(): boolean { return this.activeObjectId ? this.deleteObject(this.activeObjectId) : false; }

    deleteObject(objectId: string): boolean {
        const deletion = deletePaintObject(
            objectId,
            this.objects,
            this.strokes,
            this.occlusionClaims,
            this.activeObjectId,
        );
        if (!deletion) return false;

        this.recordUndoSnapshot();
        this.objects = deletion.objects;
        this.strokes = deletion.strokes;
        this.occlusionClaims = deletion.occlusionClaims;
        this.activeObjectId = deletion.activeObjectId;
        this.draftStroke = null;
        this.pendingStrokeView = null;
        this.meshVersion += 1;
        return true;
    }

    deleteActiveView(): boolean { return this.activeViewId ? this.deleteView(this.activeViewId) : false; }

    deleteView(viewId: string): boolean {
        const deletion = deletePaintView(
            viewId,
            this.objects,
            this.views,
            this.strokes,
            this.occlusionClaims,
            this.activeViewId,
        );
        if (!deletion) return false;

        this.recordUndoSnapshot();
        this.objects = deletion.objects;
        this.views = deletion.views;
        this.strokes = deletion.strokes;
        this.occlusionClaims = deletion.occlusionClaims;
        this.activeViewId = null;
        if (deletion.selectViewId) {
            this.selectView(deletion.selectViewId);
        }
        this.draftStroke = null;
        this.pendingStrokeView = null;
        this.meshVersion += 1;
        return true;
    }

    markSeamAt(point: Vec2): boolean { return this.markSeamAlong([point]); }

    markSeamAlong(points: Vec2[]): boolean {
        const object = this.activeObject;
        const view = this.currentEffectView();
        if (!object || !view || object.locked) return false;
        const seamEdit = planPaintChartSeams(object, this.views, view, points);
        this.raycastCountForDiagnostics += seamEdit.raycastCount;
        if (!seamEdit.hasHits) return false;

        this.recordUndoSnapshot();
        const touchedChartIds = seamEdit.apply();
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
        const view = selectPaintView(this.views, this.orbit, viewId);
        if (!view) return;
        this.activeViewId = view.id;
    }

    ensureActiveView(width = this.viewportWidth, height = this.viewportHeight, recordHistory = true) {
        this.viewportWidth = Math.max(1, width);
        this.viewportHeight = Math.max(1, height);
        const active = this.activeView;
        if (active && !this.cameraMovedFrom(active)) {
            if (active.width === this.viewportWidth && active.height === this.viewportHeight) return;
            if (!viewHasAuthoredContent(active.id, this.objects, this.strokes, this.occlusionClaims)) {
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

    cameraMovedFrom(view: PaintView): boolean { return cameraMovedFromPaintView(view, this.orbit); }

    surfaceRefWorldPoint(ref: SurfaceRef): Vec3 | null { return paintSurfaceRefWorldPoint(this.objects, this.views, ref); }

    projectSurfaceRef(ref: SurfaceRef, view: PaintView | null = this.activeView): Vec2 | null {
        return projectPaintSurfaceRef(this.objects, this.views, ref, view);
    }

    buildRenderSegments(options: boolean | PaintRenderOptions = true): RenderPrimitive[] {
        return buildPaintRenderSegments(this.renderAssemblyContext(), options);
    }

    buildDraftRenderSegments(): RenderPrimitive[] { return buildDraftPaintRenderSegments(this.renderAssemblyContext()); }

    private renderAssemblyContext(): RenderAssemblyContext {
        return {
            objects: this.objects,
            views: this.views,
            strokes: this.strokes,
            paintLayers: this.paintLayers,
            renderView: this.currentEffectView(),
            activeObject: this.activeObject,
            activeView: this.pendingStrokeView ?? this.activeView,
            draftStroke: this.draftStroke,
            brushMode: this.brushMode,
            placementMode: this.placementMode,
            chartProjectionMode: this.chartProjectionMode,
            brush: this.brush,
            defaultDepthForView: view => this.defaultDepthForView(view),
            raycastObjectSurface: (object, view, point, excludeChartId) =>
                this.raycastObjectSurface(object, view, point, excludeChartId),
        };
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
        return capturePaintView(name, width, height, this.orbit, this.camera);
    }

    private strokePlacementContext(): StrokePlacementContext {
        return {
            getOrCreateChart: (object, view, role, projectionMode) =>
                this.getOrCreateChart(object, view, role, projectionMode),
            findView: viewId => this.views.find(view => view.id === viewId) ?? null,
            defaultDepthForView: view => this.defaultDepthForView(view),
            paintDepthRadiusForView: view => this.paintDepthRadiusForView(view),
            raycastObjectSurface: (object, view, point, excludeChartId) =>
                this.raycastObjectSurface(object, view, point, excludeChartId),
            raycastObjectSurfaceBatch: (object, view, points, excludeChartId) =>
                this.raycastObjectSurfaceBatch(object, view, points, excludeChartId),
            raycastObjectSurfaces: (object, view, point) =>
                this.raycastObjectSurfaces(object, view, point),
        };
    }

    private defaultDepthForView(view: PaintView): number { return defaultDepthForPaintView(view); }

    private paintDepthRadiusForView(view: PaintView): number { return paintDepthRadiusForView(view, this.brush.width); }

    private getOrCreateChart(object: PaintObject, view: PaintView, role: ChartRole, projectionMode = this.chartProjectionMode): PaintChart {
        return getOrCreatePaintChart(object, view, role, projectionMode);
    }

    private raycastObjectSurface(object: PaintObject, view: PaintView, point: Vec2, excludeChartId?: string): SurfaceHit | null {
        this.raycastCountForDiagnostics += 1;
        return raycastPaintObjectSurfaceWithViews(object, this.views, view, point, excludeChartId, this.surfaceRaycastCache);
    }

    private raycastObjectSurfaceBatch(object: PaintObject, view: PaintView, points: Vec2[], excludeChartId?: string): Array<SurfaceHit | null> {
        if (points.length > 0) this.raycastCountForDiagnostics += 1;
        return raycastPaintObjectSurfaceBatchWithViews(object, this.views, view, points, excludeChartId, this.surfaceRaycastCache);
    }

    private raycastObjectSurfaces(object: PaintObject, view: PaintView, point: Vec2, excludeChartId?: string): SurfaceHit[] {
        this.raycastCountForDiagnostics += 1;
        return raycastPaintObjectSurfacesWithViews(object, this.views, view, point, excludeChartId, this.surfaceRaycastCache);
    }


    private nextLayerIndex(): number { return this.objects.reduce((max, object) => Math.max(max, object.layerIndex), -1) + 1; }

    private nextPaintLayerOrder(): number { return this.paintLayers.reduce((max, layer) => Math.max(max, layer.order), -1) + 1; }

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

    private pushUndoSnapshot(snapshot: PaintSceneSnapshot) { this.undoStack = [...this.undoStack, snapshot]; }

    private captureSceneSnapshot(): PaintSceneSnapshot { return capturePaintSceneSnapshot(this); }

    private restoreSceneSnapshot(snapshot: PaintSceneSnapshot) {
        Object.assign(this, restorePaintSceneSnapshot(snapshot));
        this.draftStroke = null;
        this.pendingStrokeUndoSnapshot = null;
        this.pendingStrokeView = null;
        this.pendingGpuChartPaintRuns = [];
        this.undoGroup = null;

        const activeView = this.activeView;
        if (activeView) selectPaintView(this.views, this.orbit, activeView.id);
        this.meshVersion += 1;
    }

    private touchCharts(chartIds: Set<string>) { this.objects = touchPaintCharts(this.objects, chartIds); }
}
