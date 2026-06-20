import { createChart } from "./sceneData.ts";
import {
    raycastPaintObjectSurfaceBatch,
    raycastPaintObjectSurfaces,
} from "./objectRaycast.ts";
import { cameraCenter } from "./projection.ts";
import type { PaintSurfaceRaycastCache } from "./surfaceRaycastCache.ts";
import { MIN_DEPTH } from "./constants.ts";
import { clamp, distance3 } from "./vectorMath.ts";
import type {
    ChartProjectionMode,
    ChartRole,
    PaintChart,
    PaintObject,
    PaintView,
    SurfaceHit,
    Vec2,
} from "../types.ts";

export const getOrCreatePaintChart = (
    object: PaintObject,
    view: PaintView,
    role: ChartRole,
    projectionMode: ChartProjectionMode,
): PaintChart => {
    const existing = object.charts.find(chart =>
        chart.sourceViewId === view.id
        && chart.role === role
        && chart.projectionMode === projectionMode
    );
    if (existing) return existing;

    const defaultDepth = defaultDepthForPaintView(view);
    const chart = createChart({
        objectId: object.id,
        sourceViewId: view.id,
        role,
        projectionMode,
        depth: role === "occluder" ? defaultDepth * 0.82 : defaultDepth,
    });
    object.charts.push(chart);
    return chart;
};

export const raycastPaintObjectSurfaceWithViews = (
    object: PaintObject,
    views: PaintView[],
    view: PaintView,
    point: Vec2,
    excludeChartId?: string,
    raycastCache?: PaintSurfaceRaycastCache,
): SurfaceHit | null =>
    raycastPaintObjectSurfacesWithViews(object, views, view, point, excludeChartId, raycastCache)[0] ?? null;

export const raycastPaintObjectSurfacesWithViews = (
    object: PaintObject,
    views: PaintView[],
    view: PaintView,
    point: Vec2,
    excludeChartId?: string,
    raycastCache?: PaintSurfaceRaycastCache,
): SurfaceHit[] => {
    if (!object.visible) return [];
    return raycastPaintObjectSurfaces(object, views, view, point, excludeChartId, raycastCache);
};

export const raycastPaintObjectSurfaceBatchWithViews = (
    object: PaintObject,
    views: PaintView[],
    view: PaintView,
    points: Vec2[],
    excludeChartId?: string,
    raycastCache?: PaintSurfaceRaycastCache,
): Array<SurfaceHit | null> => {
    if (!object.visible) return points.map(() => null);
    return raycastPaintObjectSurfaceBatch(object, views, view, points, excludeChartId, raycastCache);
};
export const findPaintChart = (
    objects: PaintObject[],
    chartId: string,
): PaintChart | null => {
    for (const object of objects) {
        const chart = object.charts.find(item => item.id === chartId);
        if (chart) return chart;
    }
    return null;
};

export const defaultDepthForPaintView = (view: PaintView): number => {
    const camera = cameraCenter(view);
    return Math.max(MIN_DEPTH, distance3(camera, [0, 0, 0]));
};

export const paintDepthRadiusForView = (
    view: PaintView,
    brushWidth: number,
): number => {
    const minDimension = Math.max(1, Math.min(view.width, view.height));
    return clamp(brushWidth / minDimension * 1.55, 0.035, 0.28);
};
