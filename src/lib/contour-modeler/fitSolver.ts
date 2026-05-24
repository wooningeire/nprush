import {
    DEFAULT_IMPLICIT_BODY_PARAMS,
    bodyAxes,
    cloneImplicitBodyParams,
    implicitBodyPoint,
    implicitBodySdf,
    projectPoint,
    sanitizeImplicitBodyParams,
} from "./implicitBody.ts";
import { extractImplicitBodyMesh } from "./marchingTetrahedra.ts";
import { strokeWorldPointAtIndex } from "./viewGuides.ts";
import {
    buildRenderedFitTarget,
    evaluateRenderedFeatureLoss,
    type RenderedFitTarget,
} from "./renderedFeatureLoss.ts";
import type {
    ContourStroke,
    FitSample,
    FitView,
    ImplicitBodyParams,
    Vec2,
    Vec3,
} from "./types.ts";

export interface ContourGpuFitEvaluator {
    evaluateCandidates(
        candidates: ImplicitBodyParams[],
        target: RenderedFitTarget,
    ): Promise<number[]>;
}

export interface FitImplicitBodyOptions {
    initialParams?: ImplicitBodyParams;
    strokes: ContourStroke[];
    views: FitView[];
    gpuEvaluator?: ContourGpuFitEvaluator | null;
    signal?: AbortSignal;
    meshResolution?: number;
    iterations?: number;
    candidatesPerIteration?: number;
    cpuFallbackOnGpuError?: boolean;
    onProgress?: (progress: number, bestLoss: number) => void;
}

export interface FitImplicitBodyResult {
    params: ImplicitBodyParams;
    loss: number;
    mesh: ReturnType<typeof extractImplicitBodyMesh>;
}

interface FitRegularization {
    views: Array<{
        targetBounds: ReturnType<typeof pointsBounds>;
        view: FitView;
    }>;
    depthAnchors: DepthAnchor[];
}

interface DepthAnchor {
    point: Vec3;
    weight: number;
}

export async function fitImplicitBody({
    initialParams = DEFAULT_IMPLICIT_BODY_PARAMS,
    strokes,
    views,
    gpuEvaluator,
    signal,
    meshResolution = 28,
    iterations = 7,
    candidatesPerIteration = 22,
    cpuFallbackOnGpuError = true,
    onProgress,
}: FitImplicitBodyOptions): Promise<FitImplicitBodyResult> {
    const target = buildRenderedFitTarget(strokes, views);
    const regularization = buildFitRegularization(strokes, views);
    let best = sanitizeImplicitBodyParams(estimateInitialParamsFromContours(strokes, views, initialParams));
    let bestLoss = await evaluateCandidates([best], target, gpuEvaluator, cpuFallbackOnGpuError, regularization).then(losses => losses[0] ?? 0);

    let radius = 1;
    const random = mulberry32(0x516d6f64);

    onProgress?.(0, bestLoss);

    for (let iteration = 0; iteration < iterations; iteration++) {
        if (signal?.aborted) throw new DOMException("Fit canceled", "AbortError");

        const candidates: ImplicitBodyParams[] = [best];
        for (let i = 1; i < candidatesPerIteration; i++) {
            candidates.push(mutateCandidate(best, radius, random));
        }

        const losses = await evaluateCandidates(candidates, target, gpuEvaluator, cpuFallbackOnGpuError, regularization);

        let iterationBest = best;
        let iterationBestLoss = bestLoss;
        for (let i = 0; i < candidates.length; i++) {
            const loss = losses[i] ?? Infinity;
            if (loss < iterationBestLoss) {
                iterationBest = candidates[i];
                iterationBestLoss = loss;
            }
        }

        if (iterationBestLoss < bestLoss) {
            best = iterationBest;
            bestLoss = iterationBestLoss;
        } else {
            radius *= 0.62;
        }

        if (iteration > 0 && iteration % 3 === 0) radius *= 0.72;
        onProgress?.((iteration + 1) / iterations * 0.88, bestLoss);
        await new Promise<void>(resolve => setTimeout(resolve, 0));
    }

    if (signal?.aborted) throw new DOMException("Fit canceled", "AbortError");

    onProgress?.(0.92, bestLoss);
    const mesh = extractImplicitBodyMesh(best, { resolution: meshResolution });
    onProgress?.(1, bestLoss);

    return { params: best, loss: bestLoss, mesh };
}

export function buildFitSamples(strokes: ContourStroke[], views: FitView[]): FitSample[] {
    const viewIndexById = new Map(views.map((view, index) => [view.id, index]));
    const samples: FitSample[] = [];

    for (const stroke of strokes) {
        const viewIndex = viewIndexById.get(stroke.viewId);
        if (viewIndex === undefined) continue;
        const stride = Math.max(1, Math.ceil(stroke.resampledPoints.length / 80));
        for (let i = 0; i < stroke.resampledPoints.length; i += stride) {
            samples.push({
                point: stroke.resampledPoints[i],
                kind: stroke.kind,
                viewIndex,
                weight: stroke.weight,
            });
        }
    }

    // Keep each fit pass interactive. Dense strokes are resampled before this
    // point; a few hundred screen-space targets are enough for the v1 body while
    // keeping GPU dispatches and worker batches responsive.
    return samples.slice(0, 120);
}

export function estimateInitialParamsFromContours(
    strokes: ContourStroke[],
    views: FitView[],
    fallback: ImplicitBodyParams = DEFAULT_IMPLICIT_BODY_PARAMS,
): ImplicitBodyParams {
    const view = selectPrimaryFitView(strokes, views);
    if (!view) return cloneImplicitBodyParams(fallback);

    const viewStrokes = strokes.filter(stroke => stroke.viewId === view.id);
    const edges = viewStrokes
        .filter(stroke => stroke.kind === "edge")
        .flatMap(stroke => stroke.resampledPoints);
    const all = viewStrokes.flatMap(stroke => stroke.resampledPoints);
    const source = edges.length >= 4 ? edges : all;
    if (source.length < 2) return cloneImplicitBodyParams(fallback);

    const bbox = pointsBounds(source);
    const h = Math.max(0.05, bbox.maxY - bbox.minY);
    const w = Math.max(0.05, bbox.maxX - bbox.minX);
    const cameraFit = estimateCameraFacingFrame(view, bbox);
    const scale = cameraFit?.ndcToWorldScale ?? estimateNdcToWorldScale(view.viewProjMat);

    const topWidth = bandWidth(source, bbox.maxY - h * 0.3, Infinity);
    const bottomWidth = bandWidth(source, -Infinity, bbox.minY + h * 0.3);
    const contourWidth = averageStrokeWidth(viewStrokes.filter(stroke => stroke.kind === "contour"));

    let radiusTop = (topWidth > 0.02 ? topWidth : w * 0.22) * scale * 0.5;
    let radiusBottom = (bottomWidth > 0.02 ? bottomWidth : w) * scale * 0.5;
    if (contourWidth > 0.02) {
        radiusBottom = Math.max(radiusBottom, contourWidth * scale * 0.45);
    }

    const topWorldWidth = cameraFit
        ? worldWidthAtNdc(view, bbox.centerX, bbox.maxY - h * 0.18, topWidth > 0.02 ? topWidth : w * 0.22)
        : null;
    const bottomWorldWidth = cameraFit
        ? worldWidthAtNdc(view, bbox.centerX, bbox.minY + h * 0.18, bottomWidth > 0.02 ? bottomWidth : w)
        : null;
    const contourWorldWidth = cameraFit && contourWidth > 0.02
        ? worldWidthAtNdc(view, bbox.centerX, bbox.centerY, contourWidth)
        : null;

    const params = sanitizeImplicitBodyParams({
        center: cameraFit?.center ?? [...fallback.center],
        axisX: cameraFit?.axisX ?? fallback.axisX,
        axisY: cameraFit?.axisY ?? fallback.axisY,
        axisZ: cameraFit?.axisZ ?? fallback.axisZ,
        height: Math.max(cameraFit?.height ?? h * scale, 0.16),
        radiusBottom,
        radiusTop,
        bulge: fallback.bulge,
        ovalX: Math.max(0.65, Math.min(1.45, fallback.ovalX)),
        ovalZ: Math.max(0.65, Math.min(1.45, fallback.ovalZ)),
        boxiness: estimateEdgeBoxiness(strokes, fallback.boxiness ?? 0),
    });

    const depthAnchors = buildDepthAnchors(strokes, views);
    if (depthAnchors.length >= 4) {
        const anchorBounds = points3Bounds(depthAnchors.map(anchor => anchor.point));
        params.center = [
            params.center[0] * 0.65 + anchorBounds.center[0] * 0.35,
            params.center[1] * 0.65 + anchorBounds.center[1] * 0.35,
            params.center[2] * 0.65 + anchorBounds.center[2] * 0.35,
        ];
    }

    if (topWorldWidth !== null) params.radiusTop = Math.max(0.02, topWorldWidth * 0.5);
    if (bottomWorldWidth !== null) params.radiusBottom = Math.max(0.03, bottomWorldWidth * 0.5);
    if (contourWorldWidth !== null) {
        params.radiusBottom = Math.max(params.radiusBottom, contourWorldWidth * 0.45);
    }

    return sanitizeImplicitBodyParams(params);
}

async function evaluateCandidates(
    candidates: ImplicitBodyParams[],
    target: RenderedFitTarget,
    gpuEvaluator?: ContourGpuFitEvaluator | null,
    cpuFallbackOnGpuError = true,
    regularization?: FitRegularization | null,
): Promise<number[]> {
    if (target.views.every(view => view.edgeSamples.length === 0 && view.contourSamples.length === 0)) {
        return candidates.map(candidate => regularizationLoss(candidate, regularization));
    }

    if (gpuEvaluator) {
        try {
            const losses = await gpuEvaluator.evaluateCandidates(candidates, target);
            if (losses.length === candidates.length && losses.every(Number.isFinite)) {
                return losses.map((loss, index) => loss + regularizationLoss(candidates[index], regularization));
            }
        } catch (e) {
            if (!cpuFallbackOnGpuError) throw e;
            console.warn("[contour fit] GPU evaluator failed; falling back to CPU", e);
        }
    }

    return candidates.map(candidate =>
        evaluateRenderedFeatureLoss(candidate, target) + regularizationLoss(candidate, regularization)
    );
}

function mutateCandidate(
    best: ImplicitBodyParams,
    radius: number,
    random: () => number,
): ImplicitBodyParams {
    const n = () => (random() * 2 - 1) * radius;
    const axes = mutateAxes(best, n() * 0.3, n() * 0.44, n() * 0.22);
    return sanitizeImplicitBodyParams({
        center: [
            best.center[0] + n() * 0.12,
            best.center[1] + n() * 0.12,
            best.center[2] + n() * 0.12,
        ],
        axisX: axes.x,
        axisY: axes.y,
        axisZ: axes.z,
        height: best.height * (1 + n() * 0.28),
        radiusBottom: best.radiusBottom * (1 + n() * 0.35),
        radiusTop: best.radiusTop * (1 + n() * 0.35),
        bulge: best.bulge + n() * 0.08,
        ovalX: best.ovalX * (1 + n() * 0.18),
        ovalZ: best.ovalZ * (1 + n() * 0.18),
        boxiness: (best.boxiness ?? 0) + n() * 0.22,
    });
}

function selectPrimaryFitView(strokes: ContourStroke[], views: FitView[]): FitView | null {
    if (views.length === 0) return null;
    let best = views[0];
    let bestScore = -Infinity;
    for (const view of views) {
        let edgePoints = 0;
        let contourPoints = 0;
        for (const stroke of strokes) {
            if (stroke.viewId !== view.id) continue;
            if (stroke.kind === "edge") edgePoints += stroke.resampledPoints.length;
            else contourPoints += stroke.resampledPoints.length;
        }
        const score = edgePoints * 2 + contourPoints;
        if (score > bestScore) {
            best = view;
            bestScore = score;
        }
    }
    return bestScore > 0 ? best : views[0];
}

function pointsBounds(points: { x: number; y: number }[]) {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const p of points) {
        minX = Math.min(minX, p.x);
        minY = Math.min(minY, p.y);
        maxX = Math.max(maxX, p.x);
        maxY = Math.max(maxY, p.y);
    }
    return {
        minX,
        minY,
        maxX,
        maxY,
        centerX: (minX + maxX) * 0.5,
        centerY: (minY + maxY) * 0.5,
    };
}

function points3Bounds(points: Vec3[]) {
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    for (const p of points) {
        minX = Math.min(minX, p[0]);
        minY = Math.min(minY, p[1]);
        minZ = Math.min(minZ, p[2]);
        maxX = Math.max(maxX, p[0]);
        maxY = Math.max(maxY, p[1]);
        maxZ = Math.max(maxZ, p[2]);
    }
    return {
        min: [minX, minY, minZ] as Vec3,
        max: [maxX, maxY, maxZ] as Vec3,
        center: [
            (minX + maxX) * 0.5,
            (minY + maxY) * 0.5,
            (minZ + maxZ) * 0.5,
        ] as Vec3,
    };
}

function bandWidth(points: { x: number; y: number }[], minY: number, maxY: number): number {
    const band = points.filter(p => p.y >= minY && p.y <= maxY);
    if (band.length < 2) return 0;
    const b = pointsBounds(band);
    return b.maxX - b.minX;
}

function averageStrokeWidth(strokes: ContourStroke[]): number {
    if (strokes.length === 0) return 0;
    let total = 0;
    let count = 0;
    for (const stroke of strokes) {
        if (stroke.resampledPoints.length < 2) continue;
        const b = pointsBounds(stroke.resampledPoints);
        total += b.maxX - b.minX;
        count += 1;
    }
    return count === 0 ? 0 : total / count;
}

function estimateEdgeBoxiness(strokes: ContourStroke[], fallback: number): number {
    const edgeStrokes = strokes.filter(stroke => stroke.kind === "edge" && stroke.resampledPoints.length >= 3);
    if (edgeStrokes.length < 3) return fallback;

    let straightLength = 0;
    let totalLength = 0;
    let axisAlignedLength = 0;
    for (const stroke of edgeStrokes) {
        for (let i = 1; i < stroke.resampledPoints.length; i++) {
            const a = stroke.resampledPoints[i - 1];
            const b = stroke.resampledPoints[i];
            const dx = b.x - a.x;
            const dy = b.y - a.y;
            const len = Math.hypot(dx, dy);
            if (len <= 1e-6) continue;
            totalLength += len;
            const axisAlignment = Math.max(Math.abs(dx), Math.abs(dy)) / len;
            if (axisAlignment > 0.86) axisAlignedLength += len;
        }

        const chord = Math.sqrt((stroke.resampledPoints[0].x - stroke.resampledPoints[stroke.resampledPoints.length - 1].x) ** 2
            + (stroke.resampledPoints[0].y - stroke.resampledPoints[stroke.resampledPoints.length - 1].y) ** 2);
        const length = polylineLength(stroke.resampledPoints);
        if (length > 1e-6 && chord / length > 0.92) straightLength += length;
    }

    if (totalLength <= 1e-6) return fallback;
    const straightRatio = straightLength / totalLength;
    const axisRatio = axisAlignedLength / totalLength;
    return Math.max(fallback, Math.min(0.9, straightRatio * 0.52 + axisRatio * 0.38));
}

function polylineLength(points: { x: number; y: number }[]): number {
    let total = 0;
    for (let i = 1; i < points.length; i++) {
        total += Math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y);
    }
    return total;
}

function estimateNdcToWorldScale(viewProjMat: number[]): number {
    // The app's camera uses a 90 degree perspective projection with square
    // aspect. At the origin, NDC units are approximately equal to camera depth.
    const originViewZLike = viewProjMat[11];
    return Math.max(0.45, Math.min(2.5, Math.abs(originViewZLike) || 1));
}

function buildFitRegularization(
    strokes: ContourStroke[],
    views: FitView[],
): FitRegularization | null {
    const regularizedViews = views.flatMap(view => {
        const viewStrokes = strokes.filter(stroke => stroke.viewId === view.id);
        const edges = viewStrokes
            .filter(stroke => stroke.kind === "edge")
            .flatMap(stroke => stroke.resampledPoints);
        const all = viewStrokes.flatMap(stroke => stroke.resampledPoints);
        const source = edges.length >= 4 ? edges : all;
        if (source.length < 2) return [];
        return [{
            targetBounds: pointsBounds(source),
            view,
        }];
    });

    const depthAnchors = buildDepthAnchors(strokes, views);

    return regularizedViews.length > 0 || depthAnchors.length > 0
        ? { views: regularizedViews, depthAnchors }
        : null;
}

function regularizationLoss(
    params: ImplicitBodyParams,
    regularization?: FitRegularization | null,
): number {
    if (!regularization) return 0;
    let boundsLoss = 0;
    for (const item of regularization.views) {
        const candidateBounds = projectedBodyBounds(params, item.view);
        if (!candidateBounds) {
            boundsLoss += 0.08;
            continue;
        }

        const target = item.targetBounds;
        const targetW = Math.max(0.001, target.maxX - target.minX);
        const targetH = Math.max(0.001, target.maxY - target.minY);
        const candidateW = Math.max(0.001, candidateBounds.maxX - candidateBounds.minX);
        const candidateH = Math.max(0.001, candidateBounds.maxY - candidateBounds.minY);
        const centerDx = candidateBounds.centerX - target.centerX;
        const centerDy = candidateBounds.centerY - target.centerY;
        const sizeDx = Math.log(candidateW / targetW);
        const sizeDy = Math.log(candidateH / targetH);

        boundsLoss += centerDx * centerDx * 0.4
            + centerDy * centerDy * 0.4
            + sizeDx * sizeDx * 0.08
            + sizeDy * sizeDy * 0.08;
    }

    const averagedBoundsLoss = regularization.views.length === 0 ? 0 : boundsLoss / regularization.views.length;
    return averagedBoundsLoss + depthAnchorLoss(params, regularization.depthAnchors);
}

function buildDepthAnchors(strokes: ContourStroke[], views: FitView[]): DepthAnchor[] {
    const viewById = new Map(views.map(view => [view.id, view]));
    const anchors: DepthAnchor[] = [];

    for (const stroke of strokes) {
        if (!hasEditedDepth(stroke)) continue;
        const view = viewById.get(stroke.viewId);
        if (!view?.viewProjInvMat) continue;

        const hasSampleLocks = !!stroke.depthSamplesLocked?.some(Boolean);
        const stride = hasSampleLocks ? 1 : Math.max(1, Math.ceil(stroke.resampledPoints.length / 28));
        for (let i = 0; i < stroke.resampledPoints.length; i += stride) {
            if (hasSampleLocks && !stroke.depthSamplesLocked?.[i]) continue;
            const world = strokeWorldPointAtIndex(stroke, view, i);
            if (!world) continue;
            anchors.push({
                point: world,
                weight: stroke.kind === "contour" ? 1 : 0.72,
            });
        }
    }

    return anchors.slice(0, 120);
}

function hasEditedDepth(stroke: ContourStroke): boolean {
    return !!stroke.depthLocked || !!stroke.depthSamplesLocked?.some(Boolean);
}

function depthAnchorLoss(params: ImplicitBodyParams, anchors: DepthAnchor[]): number {
    if (anchors.length === 0) return 0;
    const scale = Math.max(0.1, params.height, params.radiusBottom, params.radiusTop);
    let weightedLoss = 0;
    let totalWeight = 0;
    for (const anchor of anchors) {
        const d = implicitBodySdf(params, anchor.point) / scale;
        weightedLoss += Math.min(1, d * d) * anchor.weight;
        totalWeight += anchor.weight;
    }
    return totalWeight <= 1e-8 ? 0 : weightedLoss / totalWeight * 0.32;
}

function projectedBodyBounds(
    params: ImplicitBodyParams,
    view: FitView,
): ReturnType<typeof pointsBounds> | null {
    const points: Vec2[] = [];
    for (let r = 0; r <= 10; r++) {
        const t = r / 10;
        for (let s = 0; s < 24; s++) {
            const theta = s / 24 * Math.PI * 2;
            const projected = projectPoint(view.viewProjMat, implicitBodyPoint(params, t, theta));
            if (projected) points.push(projected);
        }
    }
    if (points.length < 2) return null;
    return pointsBounds(points);
}

function estimateCameraFacingFrame(
    view: FitView,
    bbox: ReturnType<typeof pointsBounds>,
): {
    center: Vec3;
    axisX: Vec3;
    axisY: Vec3;
    axisZ: Vec3;
    height: number;
    ndcToWorldScale: number;
} | null {
    if (!view.viewProjInvMat || !view.viewInvMat) return null;

    const depth = ndcDepthAtWorldOrigin(view.viewProjMat);
    const center = unprojectNdc(view.viewProjInvMat, bbox.centerX, bbox.centerY, depth);
    const top = unprojectNdc(view.viewProjInvMat, bbox.centerX, bbox.maxY, depth);
    const bottom = unprojectNdc(view.viewProjInvMat, bbox.centerX, bbox.minY, depth);
    const left = unprojectNdc(view.viewProjInvMat, bbox.minX, bbox.centerY, depth);
    const right = unprojectNdc(view.viewProjInvMat, bbox.maxX, bbox.centerY, depth);
    if (!center || !top || !bottom || !left || !right) return null;

    const viewAxes = axesFromViewInv(view.viewInvMat);
    const axisY = normalize3(sub3(top, bottom), viewAxes.up);
    const axisX = normalize3(sub3(right, left), viewAxes.right);
    const axisZ = viewAxes.forward;
    const height = distance3(top, bottom);
    const ndcToWorldScale = distance3(left, right) / Math.max(0.001, bbox.maxX - bbox.minX);

    return {
        center,
        axisX,
        axisY,
        axisZ,
        height,
        ndcToWorldScale,
    };
}

function worldWidthAtNdc(
    view: FitView,
    centerX: number,
    y: number,
    ndcWidth: number,
): number | null {
    if (!view.viewProjInvMat) return null;
    const depth = ndcDepthAtWorldOrigin(view.viewProjMat);
    const half = Math.max(0.001, ndcWidth * 0.5);
    const left = unprojectNdc(view.viewProjInvMat, centerX - half, y, depth);
    const right = unprojectNdc(view.viewProjInvMat, centerX + half, y, depth);
    if (!left || !right) return null;
    return distance3(left, right);
}

function ndcDepthAtWorldOrigin(viewProjMat: number[]): number {
    const clipZ = viewProjMat[14];
    const clipW = viewProjMat[15];
    if (!Number.isFinite(clipW) || Math.abs(clipW) <= 1e-6) return 0;
    return Math.max(-0.95, Math.min(0.95, clipZ / clipW));
}

function unprojectNdc(
    viewProjInvMat: number[],
    x: number,
    y: number,
    z: number,
): Vec3 | null {
    const w = 1;
    const outX = viewProjInvMat[0] * x + viewProjInvMat[4] * y + viewProjInvMat[8] * z + viewProjInvMat[12] * w;
    const outY = viewProjInvMat[1] * x + viewProjInvMat[5] * y + viewProjInvMat[9] * z + viewProjInvMat[13] * w;
    const outZ = viewProjInvMat[2] * x + viewProjInvMat[6] * y + viewProjInvMat[10] * z + viewProjInvMat[14] * w;
    const outW = viewProjInvMat[3] * x + viewProjInvMat[7] * y + viewProjInvMat[11] * z + viewProjInvMat[15] * w;
    if (!Number.isFinite(outW) || Math.abs(outW) <= 1e-6) return null;
    return [outX / outW, outY / outW, outZ / outW];
}

function axesFromViewInv(viewInvMat: number[]): { right: Vec3; up: Vec3; forward: Vec3 } {
    return {
        right: normalize3([viewInvMat[0], viewInvMat[1], viewInvMat[2]], [1, 0, 0]),
        up: normalize3([viewInvMat[4], viewInvMat[5], viewInvMat[6]], [0, 1, 0]),
        forward: normalize3([viewInvMat[8], viewInvMat[9], viewInvMat[10]], [0, 0, 1]),
    };
}

function mutateAxes(
    params: ImplicitBodyParams,
    pitch: number,
    yaw: number,
    roll: number,
): { x: Vec3; y: Vec3; z: Vec3 } {
    let { x, y, z } = bodyAxes(params);

    x = rotateVecAroundAxis(x, y, yaw);
    z = rotateVecAroundAxis(z, y, yaw);

    y = rotateVecAroundAxis(y, x, pitch);
    z = rotateVecAroundAxis(z, x, pitch);

    x = rotateVecAroundAxis(x, z, roll);
    y = rotateVecAroundAxis(y, z, roll);

    y = normalize3(y, [0, 1, 0]);
    x = normalize3(cross3(y, z), x);
    z = normalize3(cross3(x, y), z);
    return { x, y, z };
}

function rotateVecAroundAxis(v: Vec3, axis: Vec3, angle: number): Vec3 {
    const a = normalize3(axis, [0, 1, 0]);
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const dot = dot3(v, a);
    const cross = cross3(a, v);
    return [
        v[0] * cos + cross[0] * sin + a[0] * dot * (1 - cos),
        v[1] * cos + cross[1] * sin + a[1] * dot * (1 - cos),
        v[2] * cos + cross[2] * sin + a[2] * dot * (1 - cos),
    ];
}

function sub3(a: Vec3, b: Vec3): Vec3 {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function distance3(a: Vec3, b: Vec3): number {
    return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
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

function normalize3(v: Vec3, fallback: Vec3): Vec3 {
    const len = Math.hypot(v[0], v[1], v[2]);
    if (!Number.isFinite(len) || len <= 1e-8) return [...fallback] as Vec3;
    return [v[0] / len, v[1] / len, v[2] / len];
}

function mulberry32(seed: number): () => number {
    return () => {
        seed |= 0;
        seed = seed + 0x6d2b79f5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}
