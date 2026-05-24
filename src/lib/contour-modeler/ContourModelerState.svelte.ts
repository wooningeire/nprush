import { vec3 } from "wgpu-matrix";
import { Camera } from "../viewer/Camera.svelte.ts";
import { CameraOrbit } from "../viewer/CameraOrbit.svelte.ts";
import { makeContourStroke } from "./contourGeometry.ts";
import { DEFAULT_IMPLICIT_BODY_PARAMS, cloneImplicitBodyParams } from "./implicitBody.ts";
import { fitImplicitBody, type ContourGpuFitEvaluator, type FitImplicitBodyOptions } from "./fitSolver.ts";
import { fitImplicitBodyInWorker, supportsContourFitWorker } from "./fitWorkerClient.ts";
import type {
    ContourStroke,
    ContourStrokeKind,
    ContourView,
    ImplicitShape,
    Vec2,
    Vec3,
} from "./types.ts";

export class ContourModelerState {
    viewportWidth = $state(1);
    viewportHeight = $state(1);

    activeKind = $state<ContourStrokeKind>("edge");
    views = $state<ContourView[]>([]);
    strokes = $state<ContourStroke[]>([]);
    shapes = $state<ImplicitShape[]>([]);
    activeViewId = $state<string | null>(null);
    activeShapeId = $state<string | null>(null);
    activeDepthStrokeId = $state<string | null>(null);
    draftStroke = $state<Vec2[] | null>(null);
    fitProgress = $state(0);
    meshVersion = $state(0);
    private fitAbortController: AbortController | null = null;

    readonly orbit = new CameraOrbit();
    readonly camera = new Camera({
        controlScheme: this.orbit,
        screenDims: {
            width: () => this.viewportWidth,
            height: () => this.viewportHeight,
        },
    });

    get activeShape(): ImplicitShape | null {
        return this.shapes.find(shape => shape.id === this.activeShapeId) ?? null;
    }

    get activeView(): ContourView | null {
        return this.views.find(view => view.id === this.activeViewId) ?? null;
    }

    get isCameraAtActiveView(): boolean {
        const active = this.activeView;
        return !!active && !this.cameraMovedFrom(active);
    }

    get currentViewName(): string {
        if (this.isCameraAtActiveView) return this.activeView?.name ?? "No saved view";
        return this.activeView ? "New view" : "No saved view";
    }

    get isFitting(): boolean {
        return this.activeShape?.fitStatus === "fitting";
    }

    get visibleStrokes(): ContourStroke[] {
        if (!this.activeViewId || !this.activeShapeId || !this.isCameraAtActiveView) return [];
        return this.strokes.filter(stroke =>
            stroke.shapeId === this.activeShapeId
            && stroke.viewId === this.activeViewId
        );
    }

    get guideStrokes(): ContourStroke[] {
        if (!this.activeShapeId) return [];
        return this.strokes.filter(stroke =>
            stroke.shapeId === this.activeShapeId
            && (!this.isCameraAtActiveView || stroke.viewId !== this.activeViewId)
        );
    }

    get depthEditableStrokes(): ContourStroke[] {
        return this.guideStrokes;
    }

    get activeDepthStroke(): ContourStroke | null {
        return this.strokes.find(stroke => stroke.id === this.activeDepthStrokeId) ?? null;
    }

    get activeDepthValue(): number {
        const stroke = this.activeDepthStroke;
        if (!stroke) return 0;
        const view = this.views.find(item => item.id === stroke.viewId);
        return this.strokeDepthValue(stroke, view);
    }

    addShape() {
        const index = this.shapes.length + 1;
        const params = cloneImplicitBodyParams(DEFAULT_IMPLICIT_BODY_PARAMS);
        const shape: ImplicitShape = {
            id: makeId("shape"),
            name: `Body ${index}`,
            params,
            mesh: null,
            fitStatus: "idle",
            fitLoss: null,
            strokeIds: [],
        };

        this.shapes = [...this.shapes, shape];
        this.activeShapeId = shape.id;
        this.meshVersion += 1;
    }

    beginStroke(point: Vec2, width: number, height: number) {
        if (!this.activeShapeId) this.addShape();
        this.viewportWidth = Math.max(1, width);
        this.viewportHeight = Math.max(1, height);
        this.ensureActiveView(width, height);
        this.draftStroke = [point];
    }

    appendStrokePoint(point: Vec2) {
        if (!this.draftStroke || this.draftStroke.length === 0) return;
        const last = this.draftStroke[this.draftStroke.length - 1];
        const dx = point.x - last.x;
        const dy = point.y - last.y;
        if (dx * dx + dy * dy < 0.00008) return;
        this.draftStroke = [...this.draftStroke, point];
    }

    finishStroke() {
        if (!this.draftStroke || this.draftStroke.length < 2 || !this.activeShapeId || !this.activeViewId) {
            this.draftStroke = null;
            return;
        }

        const stroke = makeContourStroke({
            id: makeId("stroke"),
            kind: this.activeKind,
            viewId: this.activeViewId,
            shapeId: this.activeShapeId,
            points: this.draftStroke,
            depthNdc: this.depthAtWorldOriginForView(this.activeView),
            depthOffset: 0,
        });

        this.strokes = [...this.strokes, stroke];
        this.shapes = this.shapes.map(shape =>
            shape.id === this.activeShapeId
                ? { ...shape, strokeIds: [...shape.strokeIds, stroke.id], fitStatus: "idle" }
                : shape
        );
        this.draftStroke = null;
    }

    undoStroke() {
        if (!this.activeShapeId) return;
        const activeStrokeIds = new Set(this.activeShape?.strokeIds ?? []);
        const last = [...this.strokes].reverse().find(stroke => activeStrokeIds.has(stroke.id));
        if (!last) return;
        this.strokes = this.strokes.filter(stroke => stroke.id !== last.id);
        if (this.activeDepthStrokeId === last.id) this.activeDepthStrokeId = null;
        this.shapes = this.shapes.map(shape =>
            shape.id === this.activeShapeId
                ? { ...shape, strokeIds: shape.strokeIds.filter(id => id !== last.id), fitStatus: "idle" }
                : shape
        );
    }

    clearActiveShape() {
        const shape = this.activeShape;
        if (!shape) return;
        const strokeIds = new Set(shape.strokeIds);
        const params = cloneImplicitBodyParams(DEFAULT_IMPLICIT_BODY_PARAMS);
        this.strokes = this.strokes.filter(stroke => !strokeIds.has(stroke.id));
        if (this.activeDepthStrokeId && strokeIds.has(this.activeDepthStrokeId)) {
            this.activeDepthStrokeId = null;
        }
        this.shapes = this.shapes.map(item =>
            item.id === shape.id
                ? {
                    ...item,
                    params,
                    mesh: null,
                    fitStatus: "idle",
                    fitLoss: null,
                    strokeIds: [],
                }
                : item
        );
        this.meshVersion += 1;
    }

    selectShape(shapeId: string) {
        if (this.shapes.some(shape => shape.id === shapeId)) {
            this.activeShapeId = shapeId;
        }
    }

    selectView(viewId: string) {
        const view = this.views.find(item => item.id === viewId);
        if (!view) return;
        this.activeViewId = view.id;
        this.activeDepthStrokeId = null;
        this.orbit.long = view.long;
        this.orbit.lat = view.lat;
        this.orbit.radius = view.radius;
        this.orbit.offset = vec3.fromValues(view.offset[0], view.offset[1], view.offset[2]);
    }

    cancelFit() {
        this.fitAbortController?.abort();
    }

    selectDepthStroke(strokeId: string | null) {
        if (strokeId !== null && !this.depthEditableStrokes.some(stroke => stroke.id === strokeId)) return;
        this.activeDepthStrokeId = strokeId;
    }

    strokeDepthValue(stroke: ContourStroke, view?: ContourView | null): number {
        return clampDepth(stroke.depthNdc ?? this.depthAtWorldOriginForView(view ?? this.views.find(item => item.id === stroke.viewId)));
    }

    setStrokeDepth(strokeId: string, depthNdc: number, locked = true) {
        const nextDepth = clampDepth(depthNdc);
        let changed = false;
        this.strokes = this.strokes.map(stroke => {
            if (stroke.id !== strokeId) return stroke;
            changed = true;
            return {
                ...stroke,
                depthNdc: nextDepth,
                depthOffset: 0,
                depthLocked: locked,
                depthSamplesNdc: undefined,
                depthSamplesOffset: undefined,
                depthSamplesLocked: undefined,
            };
        });
        if (!changed) return;
        this.activeDepthStrokeId = strokeId;
        this.shapes = this.shapes.map(shape =>
            shape.id === this.activeShapeId
                ? { ...shape, fitStatus: "idle", fitLoss: null }
                : shape
        );
    }

    brushStrokeDepth(edits: Array<{ strokeId: string; pointIndex: number; influence: number; delta?: number }>, delta = 0) {
        if (edits.length === 0) return;
        const editsByStroke = new Map<string, Array<{ pointIndex: number; influence: number; delta: number }>>();
        for (const edit of edits) {
            if (edit.influence <= 0) continue;
            const editDelta = edit.delta ?? delta;
            if (Math.abs(editDelta) <= 1e-6) continue;
            const group = editsByStroke.get(edit.strokeId) ?? [];
            group.push({ pointIndex: edit.pointIndex, influence: edit.influence, delta: editDelta });
            editsByStroke.set(edit.strokeId, group);
        }
        if (editsByStroke.size === 0) return;

        let changed = false;
        this.strokes = this.strokes.map(stroke => {
            const strokeEdits = editsByStroke.get(stroke.id);
            if (!strokeEdits) return stroke;

            const view = this.views.find(item => item.id === stroke.viewId);
            const maxOffset = this.depthOffsetLimit(view);
            const offsets = Array.from({ length: stroke.resampledPoints.length }, (_, index) =>
                clampDepthOffset(stroke.depthSamplesOffset?.[index] ?? stroke.depthOffset ?? 0, maxOffset)
            );
            const locks = Array.from({ length: stroke.resampledPoints.length }, (_, index) =>
                stroke.depthSamplesLocked?.[index] ?? false
            );
            const deltas = new Array(stroke.resampledPoints.length).fill(0) as number[];
            const smoothRadius = 3;

            for (const edit of strokeEdits) {
                if (edit.pointIndex < 0 || edit.pointIndex >= offsets.length) continue;
                for (
                    let i = Math.max(0, edit.pointIndex - smoothRadius);
                    i <= Math.min(offsets.length - 1, edit.pointIndex + smoothRadius);
                    i++
                ) {
                    const distance = Math.abs(i - edit.pointIndex);
                    const lineFalloff = 1 - distance / (smoothRadius + 1);
                    const candidateDelta = edit.delta * edit.influence * lineFalloff * lineFalloff;
                    if (Math.abs(candidateDelta) > Math.abs(deltas[i])) {
                        deltas[i] = candidateDelta;
                    }
                }
            }

            for (let i = 0; i < offsets.length; i++) {
                if (Math.abs(deltas[i]) <= 1e-6) continue;
                offsets[i] = clampDepthOffset(offsets[i] + deltas[i], maxOffset);
                locks[i] = true;
                changed = true;
            }

            return {
                ...stroke,
                depthOffset: 0,
                depthSamplesNdc: undefined,
                depthSamplesOffset: offsets,
                depthSamplesLocked: locks,
                depthLocked: true,
            };
        });

        if (!changed) return;
        this.shapes = this.shapes.map(shape =>
            shape.id === this.activeShapeId
                ? { ...shape, fitStatus: "idle", fitLoss: null }
                : shape
        );
    }

    resetGuideDepths() {
        const editableIds = new Set(this.depthEditableStrokes.map(stroke => stroke.id));
        if (editableIds.size === 0) return;
        this.strokes = this.strokes.map(stroke => {
            if (!editableIds.has(stroke.id)) return stroke;
            const view = this.views.find(item => item.id === stroke.viewId);
            return {
                ...stroke,
                depthNdc: this.depthAtWorldOriginForView(view),
                depthOffset: 0,
                depthLocked: false,
                depthSamplesNdc: undefined,
                depthSamplesOffset: undefined,
                depthSamplesLocked: undefined,
            };
        });
    }

    nudgeActiveDepth(delta: number) {
        const stroke = this.activeDepthStroke;
        if (!stroke) return;
        this.setStrokeDepth(stroke.id, this.activeDepthValue + delta);
    }

    resetActiveDepth() {
        const stroke = this.activeDepthStroke;
        if (!stroke) return;
        const view = this.views.find(item => item.id === stroke.viewId);
        this.setStrokeDepth(stroke.id, this.depthAtWorldOriginForView(view), false);
    }

    async fitActiveShape(gpuEvaluator?: ContourGpuFitEvaluator | null) {
        const shape = this.activeShape;
        if (!shape || shape.fitStatus === "fitting") return;

        const shapeStrokes = this.strokes.filter(stroke => stroke.shapeId === shape.id);
        if (shapeStrokes.length === 0) return;

        this.fitAbortController = new AbortController();
        this.fitProgress = 0;
        this.updateActiveShape({ fitStatus: "fitting", fitLoss: null });

        try {
            const fitOptions: FitImplicitBodyOptions = {
                initialParams: shape.params,
                strokes: shapeStrokes,
                views: this.views.map(view => ({
                    id: view.id,
                    viewProjMat: view.viewProjMat,
                    viewProjInvMat: view.viewProjInvMat,
                    viewInvMat: view.viewInvMat,
                    width: view.width,
                    height: view.height,
                })),
                signal: this.fitAbortController.signal,
                onProgress: (progress, bestLoss) => {
                    this.fitProgress = progress;
                    this.updateActiveShape({ fitLoss: bestLoss });
                },
            };

            const result = gpuEvaluator
                ? await this.fitWithGpuThenWorkerFallback(fitOptions, gpuEvaluator)
                : await this.fitWithWorkerFallback(fitOptions);

            this.updateActiveShape({
                params: result.params,
                mesh: result.mesh,
                fitStatus: "fitted",
                fitLoss: result.loss,
            });
            this.meshVersion += 1;
        } catch (e) {
            if ((e as DOMException).name === "AbortError") {
                this.updateActiveShape({ fitStatus: "canceled" });
            } else {
                console.error("[contour modeler] fit failed", e);
                this.updateActiveShape({ fitStatus: "failed" });
            }
        } finally {
            this.fitProgress = 0;
            this.fitAbortController = null;
        }
    }

    ensureActiveView(width = this.viewportWidth, height = this.viewportHeight) {
        this.viewportWidth = Math.max(1, width);
        this.viewportHeight = Math.max(1, height);

        const active = this.activeView;
        if (active && !this.cameraMovedFrom(active)) return;

        const index = this.views.length + 1;
        const view = this.captureCurrentView(`View ${index}`, width, height);
        this.views = [...this.views, view];
        this.activeViewId = view.id;
    }

    private captureCurrentView(name: string, width: number, height: number): ContourView {
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

    cameraMovedFrom(view: ContourView): boolean {
        const offset = Array.from(this.orbit.offset).slice(0, 3) as Vec3;
        const offsetDelta = Math.hypot(
            offset[0] - view.offset[0],
            offset[1] - view.offset[1],
            offset[2] - view.offset[2],
        );
        return Math.abs(this.orbit.long - view.long) > 0.015
            || Math.abs(this.orbit.lat - view.lat) > 0.015
            || Math.abs(Math.log(this.orbit.radius / view.radius)) > 0.015
            || offsetDelta > 0.01;
    }

    private depthAtWorldOriginForView(view: ContourView | null | undefined): number {
        if (!view) return 0.5;
        const clipZ = view.viewProjMat[14];
        const clipW = view.viewProjMat[15];
        if (!Number.isFinite(clipW) || Math.abs(clipW) <= 1e-6) return 0.5;
        return clampDepth(clipZ / clipW);
    }

    private depthOffsetLimit(view: ContourView | null | undefined): number {
        return Math.max(0.25, Math.min(1.6, (view?.radius ?? this.orbit.radius) * 0.85));
    }

    private updateActiveShape(patch: Partial<ImplicitShape>) {
        if (!this.activeShapeId) return;
        this.shapes = this.shapes.map(shape =>
            shape.id === this.activeShapeId ? { ...shape, ...patch } : shape
        );
    }

    private async fitWithGpuThenWorkerFallback(
        options: FitImplicitBodyOptions,
        gpuEvaluator: ContourGpuFitEvaluator,
    ) {
        try {
            return await fitImplicitBody({
                ...options,
                gpuEvaluator,
                cpuFallbackOnGpuError: false,
            });
        } catch (error) {
            if ((error as DOMException).name === "AbortError") throw error;
            console.warn("[contour modeler] GPU fit failed; retrying in worker", error);
            return this.fitWithWorkerFallback(options);
        }
    }

    private fitWithWorkerFallback(options: FitImplicitBodyOptions) {
        if (supportsContourFitWorker()) {
            return fitImplicitBodyInWorker(options);
        }
        return fitImplicitBody({
            ...options,
            gpuEvaluator: null,
        });
    }
}

function clampDepth(depth: number): number {
    return Math.max(0.02, Math.min(0.98, depth));
}

function clampDepthOffset(offset: number, limit: number): number {
    return Math.max(-limit, Math.min(limit, offset));
}

function makeId(prefix: string): string {
    return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}
