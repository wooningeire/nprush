import { chartHasCoverage } from "./chartPainting.ts";
import {
    chartPointToWorldFromView,
    depthForProjectionAtPoint,
    projectVisiblePoint,
    viewPointToWorldAtProjectionDepth,
} from "./projection.ts";
import {
    appendChartSegments,
    appendChartSurfaceFieldSegments,
    appendStrokeRenderSegments,
    appendWorldStrokeRun,
    parseColor,
} from "./renderGeometry.ts";
import { MIN_DEPTH, OCCLUSION_GAP } from "./constants.ts";
import { samplePaintStrokeSpline } from "./strokeSampling.ts";
import type {
    BrushMode,
    BrushStyle,
    ChartProjectionMode,
    PaintChart,
    PaintObject,
    PaintRenderOptions,
    PaintStroke,
    PaintView,
    PlacementMode,
    RenderPrimitive,
    SurfaceHit,
    SurfaceRef,
    Vec2,
    Vec3,
} from "../types.ts";

export type RenderAssemblyContext = {
    objects: PaintObject[],
    views: PaintView[],
    strokes: PaintStroke[],
    activeObject: PaintObject | null,
    activeView: PaintView | null,
    draftStroke: Vec2[] | null,
    brushMode: BrushMode,
    placementMode: PlacementMode,
    chartProjectionMode: ChartProjectionMode,
    brush: BrushStyle,
    defaultDepthForView: (view: PaintView) => number,
    raycastObjectSurface: (
        object: PaintObject,
        view: PaintView,
        point: Vec2,
        excludeChartId?: string,
    ) => SurfaceHit | null,
};

export const buildPaintRenderSegments = (
    context: RenderAssemblyContext,
    options: boolean | PaintRenderOptions = true,
): RenderPrimitive[] => {
    const renderOptions = normalizeRenderOptions(options);
    const segments: RenderPrimitive[] = [];
    const objectById = new Map(context.objects.map(object => [object.id, object]));
    const viewById = new Map(context.views.map(view => [view.id, view]));
    const chartById = buildChartMap(context.objects);

    if (renderOptions.showChartWireframe) {
        for (const object of context.objects) {
            if (!object.visible) continue;
            for (const chart of object.charts) {
                const sourceView = viewById.get(chart.sourceViewId);
                if (!sourceView) continue;
                const worldAt = (point: Vec2) => chartPointToWorldFromView(chart, sourceView, point);
                appendChartSegments(segments, chart, worldAt);
            }
        }
    }

    if (renderOptions.showSurfaceField) {
        for (const object of context.objects) {
            if (!object.visible) continue;
            for (const chart of object.charts) {
                const sourceView = viewById.get(chart.sourceViewId);
                if (!sourceView) continue;
                const worldAt = (point: Vec2) => chartPointToWorldFromView(chart, sourceView, point);
                appendChartSurfaceFieldSegments(segments, chart, sourceView, worldAt);
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

    for (const stroke of sortedStrokesForRender(context.strokes, objectById)) {
        appendStrokeRenderSegments(segments, stroke, worldPointForRef);
    }

    if (renderOptions.showDraftStroke) {
        appendDraftStrokePreviewSegments(segments, context);
    }

    return segments;
};

export const buildDraftPaintRenderSegments = (context: RenderAssemblyContext): RenderPrimitive[] => {
    const segments: RenderPrimitive[] = [];
    appendDraftStrokePreviewSegments(segments, context);
    return segments;
};

export const projectPaintSurfaceRef = (
    objects: PaintObject[],
    views: PaintView[],
    ref: SurfaceRef,
    view: PaintView | null,
): Vec2 | null => {
    if (!view) return null;
    const world = paintSurfaceRefWorldPoint(objects, views, ref);
    if (!world) return null;
    return projectVisiblePoint(view.viewProjMat, world);
};

export const paintSurfaceRefWorldPoint = (
    objects: PaintObject[],
    views: PaintView[],
    ref: SurfaceRef,
): Vec3 | null => {
    const chart = findChart(objects, ref.chartId);
    if (!chart) return null;
    const view = views.find(item => item.id === chart.sourceViewId);
    if (!view) return null;
    return chartPointToWorldFromView(chart, view, ref.uv);
};

const normalizeRenderOptions = (options: boolean | PaintRenderOptions): Required<PaintRenderOptions> => {
    if (typeof options === "boolean") {
        return {
            showChartWireframe: options,
            showSurfaceField: false,
            showDraftStroke: true,
        };
    }
    return {
        showChartWireframe: options.showChartWireframe ?? true,
        showSurfaceField: options.showSurfaceField ?? false,
        showDraftStroke: options.showDraftStroke ?? true,
    };
};

const buildChartMap = (objects: PaintObject[]): Map<string, PaintChart> => {
    const chartById = new Map<string, PaintChart>();
    for (const object of objects) {
        for (const chart of object.charts) {
            chartById.set(chart.id, chart);
        }
    }
    return chartById;
};

const sortedStrokesForRender = (
    strokes: PaintStroke[],
    objectById: Map<string, PaintObject>,
): PaintStroke[] => {
    const visibleStrokes = strokes.filter(stroke => {
        const object = objectById.get(stroke.objectId);
        return !!object?.visible;
    });

    return visibleStrokes.slice().sort((a, b) =>
        (objectById.get(a.objectId)?.layerIndex ?? 0) - (objectById.get(b.objectId)?.layerIndex ?? 0)
        || a.paintOrder - b.paintOrder
    );
};

const appendDraftStrokePreviewSegments = (
    segments: RenderPrimitive[],
    context: RenderAssemblyContext,
) => {
    const object = context.activeObject;
    const view = context.activeView;
    if (!context.draftStroke || context.draftStroke.length < 2 || !object?.visible || object.locked || !view) return;

    const color = context.brushMode === "surface"
        ? [0.44, 0.92, 0.82, 0.68] as [number, number, number, number]
        : parseColor(context.brush.color, context.brush.opacity);
    const points = samplePaintStrokeSpline(context.draftStroke);
    const previewDepth = draftStrokePreviewDepth(context, object, view);
    appendWorldStrokeRun(
        segments,
        points.map(point => draftStrokePreviewWorldPoint(context, object, view, point, previewDepth)),
        color,
        context.brush.width,
    );
};

const draftStrokePreviewDepth = (
    context: RenderAssemblyContext,
    object: PaintObject,
    view: PaintView,
): number => {
    if (
        context.brushMode === "surface"
        || context.placementMode !== "snap"
        || !context.draftStroke
        || context.draftStroke.length === 0
    ) {
        return context.defaultDepthForView(view);
    }

    const cursor = context.draftStroke.at(-1)!;
    const hit = context.raycastObjectSurface(object, view, cursor);
    return hit
        ? depthForProjectionAtPoint(view, cursor, hit.viewDepth, context.chartProjectionMode)
        : context.defaultDepthForView(view);
};

const draftStrokePreviewWorldPoint = (
    context: RenderAssemblyContext,
    object: PaintObject,
    view: PaintView,
    point: Vec2,
    previewDepth: number,
): Vec3 | null => {
    if (context.brushMode === "surface") {
        return viewPointToWorldAtProjectionDepth(view, point, previewDepth, "view-plane");
    }
    if (context.placementMode === "snap") {
        return viewPointToWorldAtProjectionDepth(view, point, previewDepth, context.chartProjectionMode);
    }
    return draftStrokeWorldPoint(context, object, view, point);
};

const draftStrokeWorldPoint = (
    context: RenderAssemblyContext,
    object: PaintObject,
    view: PaintView,
    point: Vec2,
): Vec3 | null => {
    if (context.placementMode === "snap") {
        const hit = context.raycastObjectSurface(object, view, point);
        if (hit) return hit.world;
    }

    if (context.placementMode === "paint-behind") {
        const hit = context.raycastObjectSurface(object, view, point);
        if (hit) {
            const depth = depthForProjectionAtPoint(
                view,
                point,
                hit.viewDepth + OCCLUSION_GAP,
                context.chartProjectionMode,
            );
            return viewPointToWorldAtProjectionDepth(view, point, depth, context.chartProjectionMode);
        }
        return viewPointToWorldAtProjectionDepth(
            view,
            point,
            context.defaultDepthForView(view) * 1.12,
            context.chartProjectionMode,
        );
    }

    if (context.placementMode === "occluding-surface") {
        const hit = context.raycastObjectSurface(object, view, point);
        if (hit) {
            const depth = depthForProjectionAtPoint(
                view,
                point,
                Math.max(MIN_DEPTH, hit.viewDepth - OCCLUSION_GAP),
                context.chartProjectionMode,
            );
            return viewPointToWorldAtProjectionDepth(view, point, depth, context.chartProjectionMode);
        }
        return viewPointToWorldAtProjectionDepth(
            view,
            point,
            context.defaultDepthForView(view) * 0.82,
            context.chartProjectionMode,
        );
    }

    return viewPointToWorldAtProjectionDepth(
        view,
        point,
        context.defaultDepthForView(view),
        context.chartProjectionMode,
    );
};

const findChart = (objects: PaintObject[], chartId: string): PaintChart | null => {
    for (const object of objects) {
        const chart = object.charts.find(item => item.id === chartId);
        if (chart) return chart;
    }
    return null;
};
