import {
    implicitBodyPoint,
    projectPoint,
} from "./implicitBody.ts";
import { isClosedPolyline } from "./contourGeometry.ts";
import type {
    ContourStroke,
    FitView,
    ImplicitBodyParams,
    Vec2,
} from "./types.ts";

export interface StrokeDistanceField {
    width: number;
    height: number;
    maxAxis: number;
    values: Float32Array;
}

export interface StrokeSample {
    point: Vec2;
    weight: number;
}

export interface ViewFeatureTarget {
    view: FitView;
    edgeField: StrokeDistanceField;
    contourField: StrokeDistanceField;
    edgeSamples: StrokeSample[];
    contourSamples: StrokeSample[];
    contourStrokes: ContourStroke[];
    edgeOutline: Vec2[];
    targetBounds: Bounds | null;
    hasClosedEdge: boolean;
}

export interface RenderedFitTarget {
    views: ViewFeatureTarget[];
}

interface Bounds {
    minX: number;
    minY: number;
    maxX: number;
    maxY: number;
    centerX: number;
    centerY: number;
}

const FIELD_MAX_DISTANCE = 4;

export function buildRenderedFitTarget(
    strokes: ContourStroke[],
    views: FitView[],
    maxAxis = 256,
): RenderedFitTarget {
    return {
        views: views.map(view => buildViewFeatureTarget(strokes, view, maxAxis)),
    };
}

export function evaluateRenderedFeatureLoss(
    params: ImplicitBodyParams,
    target: RenderedFitTarget,
): number {
    let weightedLoss = 0;
    let totalWeight = 0;

    for (const viewTarget of target.views) {
        const edgeLoss = kindFeatureLoss(
            predictedEdgeFeatures(params, viewTarget),
            viewTarget.edgeField,
            viewTarget.edgeSamples,
        );
        if (viewTarget.edgeSamples.length > 0) {
            weightedLoss += edgeLoss * 1;
            totalWeight += 1;
        }

        const contourLoss = kindFeatureLoss(
            predictedContourFeatures(params, viewTarget),
            viewTarget.contourField,
            viewTarget.contourSamples,
        );
        if (viewTarget.contourSamples.length > 0) {
            weightedLoss += contourLoss * 0.85;
            totalWeight += 0.85;
        }
    }

    return totalWeight <= 1e-8 ? 0 : weightedLoss / totalWeight;
}

export function strokeDistanceAt(field: StrokeDistanceField, point: Vec2): number {
    const x = (point.x * 0.5 + 0.5) * (field.width - 1);
    const y = (-point.y * 0.5 + 0.5) * (field.height - 1);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return FIELD_MAX_DISTANCE;
    if (x < 0 || y < 0 || x > field.width - 1 || y > field.height - 1) return FIELD_MAX_DISTANCE;

    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const x1 = Math.min(field.width - 1, x0 + 1);
    const y1 = Math.min(field.height - 1, y0 + 1);
    const tx = x - x0;
    const ty = y - y0;
    const a = field.values[y0 * field.width + x0];
    const b = field.values[y0 * field.width + x1];
    const c = field.values[y1 * field.width + x0];
    const d = field.values[y1 * field.width + x1];
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

function buildViewFeatureTarget(
    strokes: ContourStroke[],
    view: FitView,
    maxAxis: number,
): ViewFeatureTarget {
    const viewStrokes = strokes.filter(stroke => stroke.viewId === view.id);
    const edgeStrokes = viewStrokes.filter(stroke => stroke.kind === "edge");
    const contourStrokes = viewStrokes.filter(stroke => stroke.kind === "contour");
    const edgeOutline = edgeOutlineFromStrokes(edgeStrokes);

    return {
        view,
        edgeField: edgeOutline.length >= 3
            ? buildPolylineDistanceField(edgeOutline, view, maxAxis, true)
            : buildStrokeDistanceField(edgeStrokes, view, maxAxis),
        contourField: buildStrokeDistanceField(contourStrokes, view, maxAxis),
        edgeSamples: edgeOutline.length >= 3
            ? edgeOutline.map(point => ({ point, weight: 1 }))
            : strokeSamples(edgeStrokes),
        contourSamples: strokeSamples(contourStrokes),
        contourStrokes,
        edgeOutline,
        targetBounds: pointBounds((edgeStrokes.length > 0 ? edgeStrokes : viewStrokes).flatMap(stroke => stroke.resampledPoints)),
        hasClosedEdge: edgeStrokes.some(stroke => isClosedPolyline(stroke.resampledPoints)),
    };
}

function buildStrokeDistanceField(
    strokes: ContourStroke[],
    view: FitView,
    maxAxis: number,
): StrokeDistanceField {
    return buildPolylineDistanceField(
        strokes.flatMap(stroke => polylineSegments(stroke.resampledPoints)),
        view,
        maxAxis,
        false,
    );
}

function buildPolylineDistanceField(
    polylineOrSegments: Vec2[] | Array<{ a: Vec2; b: Vec2 }>,
    view: FitView,
    maxAxis: number,
    closed: boolean,
): StrokeDistanceField {
    const { width, height } = optimizationGridForView(view, maxAxis);
    const values = new Float32Array(width * height);
    const segments = Array.isArray(polylineOrSegments) && polylineOrSegments.length > 0 && "a" in polylineOrSegments[0]
        ? polylineOrSegments as Array<{ a: Vec2; b: Vec2 }>
        : polylineSegments(polylineOrSegments as Vec2[], closed);

    if (segments.length === 0) {
        values.fill(FIELD_MAX_DISTANCE);
        return { width, height, maxAxis: Math.max(width, height), values };
    }

    for (let py = 0; py < height; py++) {
        for (let px = 0; px < width; px++) {
            const point = ndcFromPixel(px, py, width, height);
            let best = FIELD_MAX_DISTANCE;
            for (const segment of segments) {
                best = Math.min(best, pointSegmentDistance(point, segment.a, segment.b));
            }
            values[py * width + px] = best;
        }
    }

    return { width, height, maxAxis: Math.max(width, height), values };
}

function optimizationGridForView(view: FitView, maxAxis: number): { width: number; height: number } {
    const viewWidth = Math.max(1, view.width ?? 1);
    const viewHeight = Math.max(1, view.height ?? 1);
    if (viewWidth >= viewHeight) {
        return {
            width: maxAxis,
            height: Math.max(1, Math.round(maxAxis * viewHeight / viewWidth)),
        };
    }
    return {
        width: Math.max(1, Math.round(maxAxis * viewWidth / viewHeight)),
        height: maxAxis,
    };
}

function kindFeatureLoss(
    predicted: Vec2[],
    field: StrokeDistanceField,
    samples: StrokeSample[],
): number {
    if (samples.length === 0) return 0;
    if (predicted.length === 0) return 1;

    let predictedToStroke = 0;
    for (const point of predicted) {
        const d = strokeDistanceAt(field, point);
        predictedToStroke += d * d;
    }
    predictedToStroke /= predicted.length;

    let strokeToPredicted = 0;
    let totalWeight = 0;
    for (const sample of samples) {
        const d2 = nearestPointDistance2(sample.point, predicted);
        strokeToPredicted += d2 * sample.weight;
        totalWeight += sample.weight;
    }
    strokeToPredicted = totalWeight <= 1e-8 ? 0 : strokeToPredicted / totalWeight;

    return predictedToStroke * 0.45 + strokeToPredicted * 0.55;
}

function predictedEdgeFeatures(params: ImplicitBodyParams, target: ViewFeatureTarget): Vec2[] {
    const surface = projectedSurfaceSamples(params, target.view, 30, 72);
    const hull = convexHull(surface);
    const bins: Array<{ minX: number; maxX: number; minPoint: Vec2; maxPoint: Vec2 } | null> = Array.from({ length: 96 }, () => null);

    for (const point of surface) {
        if (!isFinitePoint(point)) continue;
        const bin = Math.max(0, Math.min(bins.length - 1, Math.floor((-point.y * 0.5 + 0.5) * bins.length)));
        const item = bins[bin];
        if (!item) {
            bins[bin] = { minX: point.x, maxX: point.x, minPoint: point, maxPoint: point };
            continue;
        }
        if (point.x < item.minX) {
            item.minX = point.x;
            item.minPoint = point;
        }
        if (point.x > item.maxX) {
            item.maxX = point.x;
            item.maxPoint = point;
        }
    }

    const points: Vec2[] = sampleClosedPolyline(hull, 160);
    for (const bin of bins) {
        if (!bin) continue;
        points.push(bin.minPoint, bin.maxPoint);
    }

    if (target.hasClosedEdge) {
        points.push(...projectedRing(params, target.view, 0, 48));
        points.push(...projectedRing(params, target.view, 1, 48));
    }

    return points;
}

function predictedContourFeatures(params: ImplicitBodyParams, target: ViewFeatureTarget): Vec2[] {
    const points: Vec2[] = [];
    for (const stroke of target.contourStrokes) {
        const bounds = pointBounds(stroke.resampledPoints);
        if (!bounds) continue;

        if (isClosedPolyline(stroke.resampledPoints)) {
            points.push(...projectedRing(params, target.view, contourTFromStroke(bounds, target.targetBounds), 64));
            continue;
        }

        if ((bounds.maxY - bounds.minY) > (bounds.maxX - bounds.minX) * 1.25) {
            points.push(...projectedMeridian(params, target.view, meridianThetaFromStroke(params, target.view, bounds), 36));
        } else {
            points.push(...projectedRing(params, target.view, contourTFromStroke(bounds, target.targetBounds), 48));
        }
    }
    return points;
}

function projectedSurfaceSamples(
    params: ImplicitBodyParams,
    view: FitView,
    tSteps: number,
    thetaSteps: number,
): Vec2[] {
    const points: Vec2[] = [];
    for (let ti = 0; ti <= tSteps; ti++) {
        const t = ti / tSteps;
        for (let si = 0; si < thetaSteps; si++) {
            const projected = projectPoint(view.viewProjMat, implicitBodyPoint(params, t, si / thetaSteps * Math.PI * 2));
            if (projected && withinProjectionMargin(projected)) points.push(projected);
        }
    }
    return points;
}

function projectedRing(params: ImplicitBodyParams, view: FitView, t: number, steps: number): Vec2[] {
    const points: Vec2[] = [];
    for (let i = 0; i < steps; i++) {
        const projected = projectPoint(view.viewProjMat, implicitBodyPoint(params, t, i / steps * Math.PI * 2));
        if (projected && withinProjectionMargin(projected)) points.push(projected);
    }
    return points;
}

function projectedMeridian(params: ImplicitBodyParams, view: FitView, theta: number, steps: number): Vec2[] {
    const points: Vec2[] = [];
    for (let i = 0; i <= steps; i++) {
        const projected = projectPoint(view.viewProjMat, implicitBodyPoint(params, i / steps, theta));
        if (projected && withinProjectionMargin(projected)) points.push(projected);
    }
    return points;
}

function contourTFromStroke(strokeBounds: Bounds, targetBounds: Bounds | null): number {
    if (!targetBounds) return 0.5;
    const height = Math.max(0.001, targetBounds.maxY - targetBounds.minY);
    return Math.max(0.03, Math.min(0.97, (strokeBounds.centerY - targetBounds.minY) / height));
}

function meridianThetaFromStroke(params: ImplicitBodyParams, view: FitView, strokeBounds: Bounds): number {
    let bestTheta = 0;
    let bestD2 = Infinity;
    for (let i = 0; i < 48; i++) {
        const theta = i / 48 * Math.PI * 2;
        const projected = projectPoint(view.viewProjMat, implicitBodyPoint(params, 0.5, theta));
        if (!projected) continue;
        const dx = projected.x - strokeBounds.centerX;
        const dy = projected.y - strokeBounds.centerY;
        const d2 = dx * dx + dy * dy;
        if (d2 < bestD2) {
            bestD2 = d2;
            bestTheta = theta;
        }
    }
    return bestTheta;
}

function strokeSamples(strokes: ContourStroke[]): StrokeSample[] {
    return strokes.flatMap(stroke => {
        const stride = Math.max(1, Math.ceil(stroke.resampledPoints.length / 80));
        const samples: StrokeSample[] = [];
        for (let i = 0; i < stroke.resampledPoints.length; i += stride) {
            samples.push({ point: stroke.resampledPoints[i], weight: stroke.weight });
        }
        return samples;
    }).slice(0, 160);
}

function polylineSegments(points: Vec2[], closed = false): Array<{ a: Vec2; b: Vec2 }> {
    const segments: Array<{ a: Vec2; b: Vec2 }> = [];
    for (let i = 1; i < points.length; i++) {
        segments.push({ a: points[i - 1], b: points[i] });
    }
    if (closed && points.length > 2) {
        segments.push({ a: points[points.length - 1], b: points[0] });
    }
    return segments;
}

function ndcFromPixel(px: number, py: number, width: number, height: number): Vec2 {
    return {
        x: width <= 1 ? 0 : px / (width - 1) * 2 - 1,
        y: height <= 1 ? 0 : -(py / (height - 1) * 2 - 1),
    };
}

function pointSegmentDistance(point: Vec2, a: Vec2, b: Vec2): number {
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const len2 = dx * dx + dy * dy;
    if (len2 <= 1e-8) return Math.hypot(point.x - a.x, point.y - a.y);
    const t = Math.max(0, Math.min(1, ((point.x - a.x) * dx + (point.y - a.y) * dy) / len2));
    return Math.hypot(point.x - (a.x + dx * t), point.y - (a.y + dy * t));
}

function nearestPointDistance2(point: Vec2, points: Vec2[]): number {
    let best = Infinity;
    for (const other of points) {
        const dx = point.x - other.x;
        const dy = point.y - other.y;
        best = Math.min(best, dx * dx + dy * dy);
    }
    return Number.isFinite(best) ? best : 1;
}

function pointBounds(points: Vec2[]): Bounds | null {
    if (points.length === 0) return null;
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

function isFinitePoint(point: Vec2): boolean {
    return Number.isFinite(point.x) && Number.isFinite(point.y);
}

function withinProjectionMargin(point: Vec2): boolean {
    return point.x >= -1.25 && point.x <= 1.25 && point.y >= -1.25 && point.y <= 1.25;
}

function edgeOutlineFromStrokes(strokes: ContourStroke[]): Vec2[] {
    const points = strokes.flatMap(stroke => stroke.resampledPoints);
    if (points.length < 4) return [];
    return sampleClosedPolyline(convexHull(points), 192);
}

function convexHull(points: Vec2[]): Vec2[] {
    const unique = [...new Map(points
        .filter(isFinitePoint)
        .map(point => [`${point.x.toFixed(5)},${point.y.toFixed(5)}`, point])).values()]
        .sort((a, b) => a.x === b.x ? a.y - b.y : a.x - b.x);
    if (unique.length <= 3) return unique;

    const lower: Vec2[] = [];
    for (const point of unique) {
        while (lower.length >= 2 && cross2(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) {
            lower.pop();
        }
        lower.push(point);
    }

    const upper: Vec2[] = [];
    for (let i = unique.length - 1; i >= 0; i--) {
        const point = unique[i];
        while (upper.length >= 2 && cross2(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) {
            upper.pop();
        }
        upper.push(point);
    }

    lower.pop();
    upper.pop();
    return lower.concat(upper);
}

function sampleClosedPolyline(points: Vec2[], count: number): Vec2[] {
    if (points.length < 2) return points.slice();
    const segments: Array<{ a: Vec2; b: Vec2; length: number }> = [];
    let total = 0;
    for (let i = 0; i < points.length; i++) {
        const a = points[i];
        const b = points[(i + 1) % points.length];
        const length = Math.hypot(b.x - a.x, b.y - a.y);
        if (length <= 1e-6) continue;
        segments.push({ a, b, length });
        total += length;
    }
    if (total <= 1e-6) return points.slice();

    const out: Vec2[] = [];
    let segmentIndex = 0;
    let distanceBeforeSegment = 0;
    for (let i = 0; i < count; i++) {
        const target = i / count * total;
        while (
            segmentIndex < segments.length - 1
            && distanceBeforeSegment + segments[segmentIndex].length < target
        ) {
            distanceBeforeSegment += segments[segmentIndex].length;
            segmentIndex += 1;
        }
        const segment = segments[segmentIndex];
        const t = Math.max(0, Math.min(1, (target - distanceBeforeSegment) / segment.length));
        out.push({
            x: segment.a.x + (segment.b.x - segment.a.x) * t,
            y: segment.a.y + (segment.b.y - segment.a.y) * t,
        });
    }
    return out;
}

function cross2(a: Vec2, b: Vec2, c: Vec2): number {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}
