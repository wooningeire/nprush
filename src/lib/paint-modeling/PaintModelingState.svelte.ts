import { Camera } from "../viewer/Camera.svelte.ts";
import { CameraOrbit } from "../viewer/CameraOrbit.svelte.ts";
import {
    MAX_BRUSH_WIDTH,
    MIN_BRUSH_WIDTH,
} from "./state/constants.ts";
import {
    BASE_PAINT_LAYER_ID,
    createBasePaintLayer,
    createPaintLayer,
} from "./state/paintLayers.ts";
import { reorderPaintLayers, reorderPaintObjects, reorderPaintViews } from "./state/reorder.ts";
import { makeId } from "./state/sceneData.ts";
import {
    addDeformationLineToStroke,
    raycastStrokeSurface,
    sculptStrokeMesh,
} from "./state/strokeMesh.ts";
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
    type RenderAssemblyContext,
} from "./state/renderAssembly.ts";
import { planFinishedStroke } from "./state/strokeSession.ts";
import { samplePaintStrokeSpline } from "./state/strokeSampling.ts";
import { clamp } from "./state/vectorMath.ts";
import type {
    BrushStyle,
    DeformationLine,
    PaintLayer,
    PaintObject,
    PaintStroke,
    PaintView,
    PaintRenderOptions,
    RenderPrimitive,
    RibbonUv,
    StrokeSurfaceHit,
    Vec2,
    Vec3,
} from "./types.ts";

const DEFAULT_BRUSH: BrushStyle = {
    color: "#ffd27a",
    width: 18,
    opacity: 1,
};

export class PaintModelingState {
    viewportWidth = $state(1);
    viewportHeight = $state(1);

    views = $state<PaintView[]>([]);
    objects = $state<PaintObject[]>([]);
    paintLayers = $state<PaintLayer[]>([createBasePaintLayer()]);
    strokes = $state<PaintStroke[]>([]);
    activeObjectId = $state<string | null>(null);
    activeViewId = $state<string | null>(null);
    activePaintLayerId = $state(BASE_PAINT_LAYER_ID);
    brush = $state<BrushStyle>({ ...DEFAULT_BRUSH });
    draftStroke = $state<Vec2[] | null>(null);
    undoStack = $state<PaintSceneSnapshot[]>([]);
    meshVersion = $state(0);
    private pendingStrokeUndoSnapshot: PaintSceneSnapshot | null = null;
    private pendingStrokeView: PaintView | null = null;
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

    reorderPaintLayer(layerId: string, targetLayerId: string): boolean {
        const paintLayers = reorderPaintLayers(this.paintLayers, layerId, targetLayerId);
        if (!paintLayers) return false;

        this.recordUndoSnapshot();
        this.paintLayers = paintLayers;
        this.meshVersion += 1;
        return true;
    }

    reorderObject(objectId: string, targetObjectId: string): boolean {
        const objects = reorderPaintObjects(this.objects, objectId, targetObjectId);
        if (!objects) return false;

        this.recordUndoSnapshot();
        this.objects = objects;
        this.meshVersion += 1;
        return true;
    }

    reorderView(viewId: string, targetViewId: string): boolean {
        const views = reorderPaintViews(this.views, viewId, targetViewId);
        if (!views) return false;

        this.recordUndoSnapshot();
        this.views = views;
        return true;
    }

    setBrushColor(color: string) { this.brush.color = color; }

    setBrushWidth(width: number) { this.brush.width = clamp(width, MIN_BRUSH_WIDTH, MAX_BRUSH_WIDTH); }

    setBrushOpacity(_opacity: number) { this.brush.opacity = 1; }

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

    beginStroke(point: Vec2, width: number, height: number) {
        const undoSnapshot = this.captureSceneSnapshot();
        this.viewportWidth = Math.max(1, width);
        this.viewportHeight = Math.max(1, height);
        if (!this.activeObjectId) this.addObject(undefined, false);
        const object = this.activeObject;
        if (!object || object.locked || !object.visible) {
            this.pendingStrokeUndoSnapshot = null;
            this.pendingStrokeView = null;
            return;
        }

        this.ensureActiveView(width, height, false);
        this.pendingStrokeView = this.activeView;
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

    finishStroke() {
        const result = planFinishedStroke({
            draftStroke: this.draftStroke,
            pendingStrokeUndoSnapshot: this.pendingStrokeUndoSnapshot,
            undoSnapshot: this.pendingStrokeUndoSnapshot ?? this.captureSceneSnapshot(),
            object: this.activeObject,
            view: this.pendingStrokeView ?? this.activeView,
            brush: this.brush,
            nextPaintOrder: objectId => this.nextPaintOrder(objectId),
            paintLayerId: this.activePaintLayer?.id ?? BASE_PAINT_LAYER_ID,
        });

        this.draftStroke = null;
        this.pendingStrokeUndoSnapshot = null;
        this.pendingStrokeView = null;
        if (result.kind === "discard") {
            if (result.restoreSnapshot) this.restoreSceneSnapshot(result.restoreSnapshot);
            return;
        }

        this.pushUndoSnapshot(result.undoSnapshot);
        this.strokes = [...this.strokes, result.stroke];
        this.meshVersion += 1;
    }

    sculptStrokeAt(strokeId: string, center: RibbonUv, delta: Vec3, radius?: number): boolean {
        if (!this.strokes.some(stroke => stroke.id === strokeId)) return false;
        this.recordUndoSnapshot();
        this.strokes = this.strokes.map(stroke => stroke.id === strokeId
            ? sculptStrokeMesh(stroke, center, delta, radius)
            : stroke);
        this.meshVersion += 1;
        return true;
    }

    addDeformationLine(strokeId: string, points: RibbonUv[]): boolean {
        if (!this.strokes.some(stroke => stroke.id === strokeId) || points.length < 2) return false;
        this.recordUndoSnapshot();
        const line: DeformationLine = {
            id: makeId("deform"),
            points: points.map(point => ({ ...point })),
        };
        this.strokes = this.strokes.map(stroke => stroke.id === strokeId
            ? addDeformationLineToStroke(stroke, line)
            : stroke);
        this.meshVersion += 1;
        return true;
    }

    raycastStrokeAt(point: Vec2, view: PaintView | null = this.currentEffectView()): StrokeSurfaceHit | null {
        if (!view) return null;
        return raycastStrokeSurface(this.objects, this.strokes, view, point);
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
            this.activeObjectId,
        );
        if (!deletion) return false;

        this.recordUndoSnapshot();
        this.objects = deletion.objects;
        this.strokes = deletion.strokes;
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
            this.views,
            this.strokes,
            this.activeViewId,
        );
        if (!deletion) return false;

        this.recordUndoSnapshot();
        this.views = deletion.views;
        this.strokes = deletion.strokes;
        this.activeViewId = null;
        if (deletion.selectViewId) {
            this.selectView(deletion.selectViewId);
        }
        this.draftStroke = null;
        this.pendingStrokeView = null;
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
            if (!viewHasAuthoredContent(active.id, this.strokes)) {
                const refreshedView = this.captureCurrentView(active.name, this.viewportWidth, this.viewportHeight);
                this.views = this.views.map(view => view.id === active.id
                    ? {
                        ...refreshedView,
                        id: active.id,
                        order: active.order,
                        createdAt: active.createdAt,
                    }
                    : view);
                return;
            }
        }
        this.saveCurrentView(width, height, recordHistory);
    }

    cameraMovedFrom(view: PaintView): boolean { return cameraMovedFromPaintView(view, this.orbit); }

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
            brush: this.brush,
        };
    }

    private currentEffectView(): PaintView | null {
        const width = Math.max(1, this.viewportWidth);
        const height = Math.max(1, this.viewportHeight);
        const active = this.activeView;
        if (active && !this.cameraMovedFrom(active) && active.width === width && active.height === height) {
            return active;
        }
        return this.captureCurrentView("Interaction view", width, height);
    }

    private captureCurrentView(name: string, width: number, height: number): PaintView {
        return capturePaintView(name, this.nextViewOrder(), width, height, this.orbit, this.camera);
    }

    private nextLayerIndex(): number { return this.objects.reduce((max, object) => Math.max(max, object.layerIndex), -1) + 1; }

    private nextPaintLayerOrder(): number { return this.paintLayers.reduce((max, layer) => Math.max(max, layer.order), -1) + 1; }

    private nextViewOrder(): number { return this.views.reduce((max, view) => Math.max(max, view.order), -1) + 1; }

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
        this.undoGroup = null;

        const activeView = this.activeView;
        if (activeView) selectPaintView(this.views, this.orbit, activeView.id);
        this.meshVersion += 1;
    }
}
