import type { ChartPaintRun, ChartPaintSample } from "./chartPainting.ts";
import {
    applyStrokeToChartGeometry,
    appendPaintRun,
    markChartSeamAlongPolyline,
} from "./chartPainting.ts";
import { MIN_DEPTH, OCCLUSION_GAP, SEAM_BRUSH_RADIUS } from "./constants.ts";
import { depthForProjectionAtPoint } from "./projection.ts";
import { makeId } from "./sceneData.ts";
import {
    carryStrokeDepths,
    depthForCarriedSnapAtPoint,
    snapCarryDepthAtPoint,
    type SnapCarryDepth,
} from "./snapDepthCarry.ts";
import type {
    ChartProjectionMode,
    ChartRole,
    OcclusionClaim,
    PaintChart,
    PaintObject,
    PaintSample,
    PaintView,
    PlacementMode,
    SurfaceHit,
    Vec2,
} from "../types.ts";

export type SurfaceBrushPlacement = {
    touchedChartIds: Set<string>,
    gpuChartPaintRuns: ChartPaintRun[],
};

export type StrokePlacementResult = {
    samples: PaintSample[],
    occlusionClaim?: OcclusionClaim,
    touchedChartIds: Set<string>,
    gpuChartPaintRuns: ChartPaintRun[],
};

export type StrokePlacementContext = {
    getOrCreateChart: (
        object: PaintObject,
        view: PaintView,
        role: ChartRole,
        projectionMode?: ChartProjectionMode,
    ) => PaintChart,
    defaultDepthForView: (view: PaintView) => number,
    paintDepthRadiusForView: (view: PaintView) => number,
    raycastObjectSurface: (
        object: PaintObject,
        view: PaintView,
        point: Vec2,
        excludeChartId?: string,
    ) => SurfaceHit | null,
    raycastObjectSurfaceBatch: (
        object: PaintObject,
        view: PaintView,
        points: Vec2[],
        excludeChartId?: string,
    ) => Array<SurfaceHit | null>,
    raycastObjectSurfaces: (object: PaintObject, view: PaintView, point: Vec2) => SurfaceHit[],
};

export type SnapPlacementPlan = {
    hits: Array<SurfaceHit | null>,
    carriedDepths: Array<SnapCarryDepth | null>,
};

export function placeSurfaceBrushMask(
    context: StrokePlacementContext,
    object: PaintObject,
    view: PaintView,
    points: Vec2[],
): SurfaceBrushPlacement {
    const chart = context.getOrCreateChart(object, view, "surface", "view-plane");
    const depth = context.defaultDepthForView(view);
    const radius = context.paintDepthRadiusForView(view);
    const paintSamples = points.map(point => ({ point, depth }));
    const changed = applyStrokeToChartGeometry(
        chart,
        paintSamples,
        radius,
        {
            requireCoverage: false,
        },
    );

    return {
        touchedChartIds: changed ? new Set([chart.id]) : new Set(),
        gpuChartPaintRuns: changed
            ? [{
                chart,
                samples: paintSamples,
                radius,
                requireCoverage: false,
                depthWriteMode: "blend",
            }]
            : [],
    };
}

export function placeStrokeSamples(
    context: StrokePlacementContext,
    object: PaintObject,
    view: PaintView,
    points: Vec2[],
    placement: PlacementMode,
    snapPlacementPlan?: SnapPlacementPlan,
): StrokePlacementResult {
    if (placement === "occluding-surface") {
        return placeOccludingSamples(context, object, view, points);
    }

    const samples: PaintSample[] = [];
    const touchedChartIds = new Set<string>();
    const paintRuns: ChartPaintRun[] = [];
    const gpuChartPaintRuns: ChartPaintRun[] = [];
    let fallbackChart: PaintChart | null = null;
    const fallbackDepth = context.defaultDepthForView(view);
    const paintDepthRadius = context.paintDepthRadiusForView(view);
    const activeSnapPlacementPlan = placement === "snap"
        ? snapPlacementPlan ?? planSnapPlacement(context, object, view, points)
        : null;

    for (let pointIndex = 0; pointIndex < points.length; pointIndex++) {
        const point = points[pointIndex];
        if (placement === "snap") {
            const hit = activeSnapPlacementPlan?.hits[pointIndex] ?? null;
            if (hit) {
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
            const hits = context.raycastObjectSurfaces(object, view, point);
            const firstHit = hits[0] ?? null;
            const backHit = firstHit
                ? hits.find(hit => hit.viewDepth > firstHit.viewDepth + OCCLUSION_GAP * 0.5) ?? null
                : null;
            if (backHit) {
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
        fallbackChart ??= context.getOrCreateChart(object, view, role);
        const carriedDepth = placement === "snap"
            ? depthForCarriedSnapAtPoint(
                view,
                point,
                activeSnapPlacementPlan?.carriedDepths[pointIndex] ?? null,
                fallbackChart.projectionMode,
            )
            : null;
        appendPaintRun(paintRuns, fallbackChart, {
            point,
            depth: carriedDepth
                ?? (depthIsViewRayDistance
                    ? depthForProjectionAtPoint(view, point, depth, fallbackChart.projectionMode)
                    : depth),
        }, paintDepthRadius, false, placement === "snap" ? "replace" : "blend");
        samples.push({
            sourcePoint: point,
            surfaceRef: { chartId: fallbackChart.id, uv: { ...point } },
            placement,
        });
    }

    for (const run of paintRuns) {
        if (applyStrokeToChartGeometry(run.chart, run.samples, paintDepthRadius, {
            requireCoverage: run.requireCoverage,
            depthWriteMode: run.depthWriteMode,
        })) {
            touchedChartIds.add(run.chart.id);
            gpuChartPaintRuns.push(run);
        }
    }

    return { samples, touchedChartIds, gpuChartPaintRuns };
}

function placeOccludingSamples(
    context: StrokePlacementContext,
    object: PaintObject,
    view: PaintView,
    points: Vec2[],
): StrokePlacementResult {
    const chart = context.getOrCreateChart(object, view, "occluder");
    const touchedChartIds = new Set([chart.id]);
    const paintDepthRadius = context.paintDepthRadiusForView(view);
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
        const backHit = context.raycastObjectSurface(object, view, point, chart.id);
        const depth = backHit
            ? depthForProjectionAtPoint(
                view,
                point,
                Math.max(MIN_DEPTH, backHit.viewDepth - OCCLUSION_GAP),
                chart.projectionMode,
            )
            : context.defaultDepthForView(view) * 0.82;
        paintSamples.push({ point, depth });
        claim.mask.push({ ...point });
        if (backHit) claim.backRefs.push(backHit.surfaceRef);
        samples.push({
            sourcePoint: point,
            surfaceRef: { chartId: chart.id, uv: { ...point } },
            placement: "occluding-surface",
        });
    }

    const painted = applyStrokeToChartGeometry(chart, paintSamples, paintDepthRadius, {
        requireCoverage: false,
    });
    markChartSeamAlongPolyline(chart, points, SEAM_BRUSH_RADIUS);

    return {
        samples,
        occlusionClaim: claim,
        touchedChartIds,
        gpuChartPaintRuns: painted
            ? [{
                chart,
                samples: paintSamples,
                radius: paintDepthRadius,
                requireCoverage: false,
                depthWriteMode: "blend",
            }]
            : [],
    };
}

function planSnapPlacement(
    context: StrokePlacementContext,
    object: PaintObject,
    view: PaintView,
    points: Vec2[],
): SnapPlacementPlan {
    const hits = context.raycastObjectSurfaceBatch(object, view, points);
    const directDepths = hits.map(hit => hit
        ? snapCarryDepthAtPoint(hit.viewDepth)
        : null);

    return {
        hits,
        carriedDepths: carryStrokeDepths(directDepths, points),
    };
}
