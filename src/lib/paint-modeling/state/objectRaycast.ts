import { chartHasCoverage } from "./chartPainting.ts";
import {
    createChartRaycastCache,
    raycastCachedChart,
} from "./chartRaycast.ts";
import {
    chartPointToWorldFromView,
    makeViewRay,
} from "./projection.ts";
import type {
    PaintObject,
    PaintView,
    SurfaceHit,
    Vec2,
} from "../types.ts";
import type {
    PaintSurfaceRaycastCache,
    SurfaceRaycastTarget,
} from "./surfaceRaycastCache.ts";

export const raycastPaintObjectSurface = (
    object: PaintObject,
    views: PaintView[],
    view: PaintView,
    point: Vec2,
    excludeChartId?: string,
    raycastCache?: PaintSurfaceRaycastCache,
): SurfaceHit | null =>
    raycastPaintObjectSurfaces(object, views, view, point, excludeChartId, raycastCache)[0] ?? null;

export const raycastPaintObjectSurfaces = (
    object: PaintObject,
    views: PaintView[],
    view: PaintView,
    point: Vec2,
    excludeChartId?: string,
    raycastCache?: PaintSurfaceRaycastCache,
): SurfaceHit[] =>
    raycastPaintObjectSurfacesBatch(object, views, view, [point], excludeChartId, raycastCache)[0] ?? [];

export const raycastPaintObjectSurfaceBatch = (
    object: PaintObject,
    views: PaintView[],
    view: PaintView,
    points: Vec2[],
    excludeChartId?: string,
    raycastCache?: PaintSurfaceRaycastCache,
): Array<SurfaceHit | null> =>
    raycastPaintObjectSurfacesBatch(object, views, view, points, excludeChartId, raycastCache)
        .map(hits => hits[0] ?? null);

export const raycastPaintObjectSurfacesBatch = (
    object: PaintObject,
    views: PaintView[],
    view: PaintView,
    points: Vec2[],
    excludeChartId?: string,
    raycastCache?: PaintSurfaceRaycastCache,
): SurfaceHit[][] => {
    if (!object.visible) return points.map(() => []);

    const targets = raycastCache
        ? raycastCache.targetsForObject(object, views, excludeChartId)
        : buildRaycastTargets(object, views, excludeChartId);
    if (targets.length === 0) return points.map(() => []);

    return points.map(point => {
        const ray = makeViewRay(view, point);
        if (!ray) return [];

        const hits: SurfaceHit[] = [];
        for (const target of targets) {
            const chartHits = raycastCachedChart(target.cache, ray);
            for (const hit of chartHits) {
                hits.push({
                    objectId: object.id,
                    chartId: target.chartId,
                    surfaceRef: { chartId: target.chartId, uv: hit.uv },
                    world: hit.world,
                    viewDepth: hit.t,
                });
            }
        }

        return hits.sort((a, b) => a.viewDepth - b.viewDepth);
    });
};

const buildRaycastTargets = (
    object: PaintObject,
    views: PaintView[],
    excludeChartId?: string,
): SurfaceRaycastTarget[] => {
    const targets: SurfaceRaycastTarget[] = [];
    for (const chart of object.charts) {
        if (chart.id === excludeChartId) continue;
        if (!chartHasCoverage(chart)) continue;

        const sourceView = views.find(item => item.id === chart.sourceViewId);
        if (!sourceView) continue;

        targets.push({
            chartId: chart.id,
            cache: createChartRaycastCache(
                chart,
                uv => chartPointToWorldFromView(chart, sourceView, uv),
            ),
        });
    }
    return targets;
};