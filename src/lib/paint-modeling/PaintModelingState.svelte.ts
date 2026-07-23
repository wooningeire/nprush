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
import { reorderPaintLayers, reorderPaintObjects } from "./state/reorder.ts";
import { makeId } from "./state/sceneData.ts";
import { deletePaintObject } from "./state/sceneLifecycle.ts";
import {
    capturePaintSceneSnapshot,
    restorePaintSceneSnapshot,
    type PaintSceneSnapshot,
} from "./state/sceneHistory.ts";
import { captureProjectionSnapshot } from "./state/projectionSnapshot.ts";
import {
    buildDraftPaintRenderSegments,
    buildPaintRenderSegments,
    type RenderAssemblyContext,
} from "./state/renderAssembly.ts";
import {
    planFinishedStroke,
    planFinishedStrokeWithRibbon,
} from "./state/strokeSession.ts";
import { samplePaintStrokeSpline } from "./state/strokeSampling.ts";
import { clamp } from "./state/vectorMath.ts";
import {
    createDefaultConstructionPlane,
    movePointToViewDepth,
    normalizedPlane,
    viewDepthForPoint,
    viewFacingNormal,
} from "./state/constructionPlane.ts";
import {
    BrushPlacementMode,
    type BrushPlacementMode as BrushPlacementModeValue,
    type BrushStyle,
    type ConstructionPlane,
    type PaintLayer,
    type PaintObject,
    type PaintRibbon,
    type PaintStroke,
    type PaintRenderOptions,
    type ProjectionSnapshot,
    type RenderPrimitive,
    type Vec2,
    type Vec3,
} from "./types.ts";

const DEFAULT_BRUSH: BrushStyle = {
    color: "#ffd27a",
    width: 18,
    opacity: 1,
};

export class PaintModelingState {
    viewportWidth = $state(1);
    viewportHeight = $state(1);

    objects = $state<PaintObject[]>([]);
    paintLayers = $state<PaintLayer[]>([createBasePaintLayer()]);
    strokes = $state<PaintStroke[]>([]);
    activeObjectId = $state<string | null>(null);
    activePaintLayerId = $state(BASE_PAINT_LAYER_ID);
    brush = $state<BrushStyle>({ ...DEFAULT_BRUSH });
    brushPlacementMode = $state<BrushPlacementModeValue>(BrushPlacementMode.View);
    constructionPlane = $state<ConstructionPlane>(createDefaultConstructionPlane());
    draftStroke = $state<Vec2[] | null>(null);
    undoStack = $state<PaintSceneSnapshot[]>([]);
    ribbonVersion = $state(0);
    private pendingStrokeUndoSnapshot: PaintSceneSnapshot | null = null;
    private pendingStrokeProjection: ProjectionSnapshot | null = null;
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

    get canUndo(): boolean { return this.undoStack.length > 0; }

    addObject(name?: string, recordHistory = true) {
        if (recordHistory) this.recordUndoSnapshot();
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
        this.ribbonVersion += 1;
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
        this.ribbonVersion += 1;
        return true;
    }

    reorderObject(objectId: string, targetObjectId: string): boolean {
        const objects = reorderPaintObjects(this.objects, objectId, targetObjectId);
        if (!objects) return false;

        this.recordUndoSnapshot();
        this.objects = objects;
        this.ribbonVersion += 1;
        return true;
    }

    setBrushColor(color: string) { this.brush.color = color; }

    setBrushWidth(width: number) { this.brush.width = clamp(width, MIN_BRUSH_WIDTH, MAX_BRUSH_WIDTH); }

    setBrushOpacity(_opacity: number) { this.brush.opacity = 1; }

    setBrushPlacementMode(mode: BrushPlacementModeValue) { this.brushPlacementMode = mode; }

    get constructionPlaneViewDepth(): number {
        return viewDepthForPoint(this.camera.viewInvMat, this.constructionPlane.origin);
    }

    setConstructionPlaneViewDepth(depth: number) {
        this.constructionPlane.origin = movePointToViewDepth(
            this.camera.viewInvMat,
            this.constructionPlane.origin,
            depth,
        );
    }

    setConstructionPlaneNormal(normal: Vec3) {
        this.constructionPlane.normal = normalizedPlane(
            this.constructionPlane.origin,
            normal,
            this.constructionPlane.normal,
        ).normal;
    }

    setConstructionPlane(origin: Vec3, normal: Vec3) {
        this.constructionPlane = normalizedPlane(origin, normal, this.constructionPlane.normal);
    }

    alignConstructionPlaneToView() {
        this.constructionPlane.normal = viewFacingNormal(this.camera.viewInvMat);
    }

    flipConstructionPlaneNormal() {
        const [x, y, z] = this.constructionPlane.normal;
        this.constructionPlane.normal = [-x, -y, -z];
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

    beginStroke(point: Vec2, width: number, height: number) {
        const undoSnapshot = this.captureSceneSnapshot();
        this.viewportWidth = Math.max(1, width);
        this.viewportHeight = Math.max(1, height);
        if (!this.activeObjectId) this.addObject(undefined, false);
        const object = this.activeObject;
        if (!object || object.locked || !object.visible) {
            this.pendingStrokeUndoSnapshot = null;
            this.pendingStrokeProjection = null;
            return;
        }

        this.pendingStrokeProjection = this.currentProjectionSnapshot();
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
        if (
            !this.draftStroke
            || this.draftStroke.length < 2
            || !object
            || object.locked
            || !object.visible
            || !this.strokePlacementProjection()
        ) {
            return null;
        }
        return samplePaintStrokeSpline(this.draftStroke);
    }

    strokePlacementProjection(): ProjectionSnapshot | null { return this.pendingStrokeProjection; }

    finishStroke() {
        const result = planFinishedStroke({
            draftStroke: this.draftStroke,
            pendingStrokeUndoSnapshot: this.pendingStrokeUndoSnapshot,
            undoSnapshot: this.pendingStrokeUndoSnapshot ?? this.captureSceneSnapshot(),
            object: this.activeObject,
            sourceProjection: this.pendingStrokeProjection,
            brush: this.brush,
            nextPaintOrder: objectId => this.nextPaintOrder(objectId),
            paintLayerId: this.activePaintLayer?.id ?? BASE_PAINT_LAYER_ID,
        });

        this.draftStroke = null;
        this.pendingStrokeUndoSnapshot = null;
        this.pendingStrokeProjection = null;
        if (result.kind === "discard") {
            if (result.restoreSnapshot) this.restoreSceneSnapshot(result.restoreSnapshot);
            return;
        }

        this.pushUndoSnapshot(result.undoSnapshot);
        this.strokes = [...this.strokes, result.stroke];
        this.ribbonVersion += 1;
    }

    finishStrokeWithRibbon(sourcePoints: Vec2[], ribbon: PaintRibbon): boolean {
        const result = planFinishedStrokeWithRibbon({
            draftStroke: this.draftStroke,
            pendingStrokeUndoSnapshot: this.pendingStrokeUndoSnapshot,
            undoSnapshot: this.pendingStrokeUndoSnapshot ?? this.captureSceneSnapshot(),
            object: this.activeObject,
            sourceProjection: this.pendingStrokeProjection,
            brush: this.brush,
            nextPaintOrder: objectId => this.nextPaintOrder(objectId),
            paintLayerId: this.activePaintLayer?.id ?? BASE_PAINT_LAYER_ID,
            sourcePoints,
            ribbon,
        });

        this.draftStroke = null;
        this.pendingStrokeUndoSnapshot = null;
        this.pendingStrokeProjection = null;
        if (result.kind === "discard") {
            if (result.restoreSnapshot) this.restoreSceneSnapshot(result.restoreSnapshot);
            return false;
        }

        this.pushUndoSnapshot(result.undoSnapshot);
        this.strokes = [...this.strokes, result.stroke];
        this.ribbonVersion += 1;
        return true;
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
        this.pendingStrokeProjection = null;
        this.ribbonVersion += 1;
        return true;
    }

    buildRenderSegments(options: boolean | PaintRenderOptions = true): RenderPrimitive[] {
        return buildPaintRenderSegments(this.renderAssemblyContext(), options);
    }

    buildDraftRenderSegments(): RenderPrimitive[] { return buildDraftPaintRenderSegments(this.renderAssemblyContext()); }

    private renderAssemblyContext(): RenderAssemblyContext {
        return {
            objects: this.objects,
            strokes: this.strokes,
            paintLayers: this.paintLayers,
            renderProjection: this.currentProjectionSnapshot(),
            activeObject: this.activeObject,
            activeProjection: this.pendingStrokeProjection,
            draftStroke: this.draftStroke,
            brush: this.brush,
        };
    }

    private currentProjectionSnapshot(): ProjectionSnapshot {
        return captureProjectionSnapshot(
            this.viewportWidth,
            this.viewportHeight,
            this.camera,
        );
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
        this.pendingStrokeProjection = null;
        this.undoGroup = null;
        this.ribbonVersion += 1;
    }
}
