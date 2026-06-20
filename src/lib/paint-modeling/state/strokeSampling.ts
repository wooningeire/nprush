import type { Vec2 } from "../types.ts";

const MIN_STROKE_SPLINE_CONTROL_POINTS = 4;
const STROKE_SPLINE_SAMPLE_SPACING = 0.018;

export function resamplePaintPolyline(points: Vec2[], maxSamples: number): Vec2[] {
    if (points.length <= 1 || maxSamples <= 1) return points.map(point => ({ ...point }));
    const length = polylineLength(points);
    if (length <= 1e-6) return [points[0], points.at(-1)!].map(point => ({ ...point }));
    const samples: Vec2[] = [];
    const step = length / (maxSamples - 1);
    let segmentIndex = 1;
    let segmentStart = points[0];
    let segmentEnd = points[1];
    let segmentLength = distance2d(segmentStart, segmentEnd);
    let distanceBeforeSegment = 0;

    for (let sampleIndex = 0; sampleIndex < maxSamples; sampleIndex++) {
        const target = Math.min(length, sampleIndex * step);
        while (segmentIndex < points.length - 1 && distanceBeforeSegment + segmentLength < target) {
            distanceBeforeSegment += segmentLength;
            segmentIndex += 1;
            segmentStart = points[segmentIndex - 1];
            segmentEnd = points[segmentIndex];
            segmentLength = distance2d(segmentStart, segmentEnd);
        }
        const t = segmentLength <= 1e-6 ? 0 : (target - distanceBeforeSegment) / segmentLength;
        samples.push({
            x: segmentStart.x + (segmentEnd.x - segmentStart.x) * t,
            y: segmentStart.y + (segmentEnd.y - segmentStart.y) * t,
        });
    }
    return samples;
}

export function samplePaintStrokeSpline(points: Vec2[]): Vec2[] {
    const length = polylineLength(points);
    const sampleCount = paintStrokePolylineSampleCount(length);
    if (points.length < MIN_STROKE_SPLINE_CONTROL_POINTS) {
        return resamplePaintPolyline(points, sampleCount);
    }
    return sampleClampedCubicBSpline(points);
}

export function distance2d(a: Vec2, b: Vec2): number {
    return Math.hypot(a.x - b.x, a.y - b.y);
}

function paintStrokePolylineSampleCount(length: number): number {
    return Math.max(2, Math.ceil(length / STROKE_SPLINE_SAMPLE_SPACING) + 1);
}

function sampleClampedCubicBSpline(controls: Vec2[]): Vec2[] {
    if (controls.length < MIN_STROKE_SPLINE_CONTROL_POINTS) {
        return resamplePaintPolyline(controls, paintStrokePolylineSampleCount(polylineLength(controls)));
    }

    const padded = [
        controls[0],
        controls[0],
        ...controls,
        controls.at(-1)!,
        controls.at(-1)!,
    ];
    const samples: Vec2[] = [];
    for (let i = 0; i <= padded.length - 4; i++) {
        const p0 = padded[i];
        const p1 = padded[i + 1];
        const p2 = padded[i + 2];
        const p3 = padded[i + 3];
        const steps = Math.max(
            2,
            Math.ceil(cubicBSplineSpanLength(p0, p1, p2, p3) / STROKE_SPLINE_SAMPLE_SPACING),
        );
        for (let step = 0; step < steps; step++) {
            if (i > 0 && step === 0) continue;
            samples.push(cubicBSplinePoint(p0, p1, p2, p3, step / steps));
        }
    }
    samples.push({ ...controls.at(-1)! });
    return samples;
}

function cubicBSplineSpanLength(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2): number {
    let length = 0;
    let previous = cubicBSplinePoint(p0, p1, p2, p3, 0);
    for (let i = 1; i <= 8; i++) {
        const current = cubicBSplinePoint(p0, p1, p2, p3, i / 8);
        length += distance2d(previous, current);
        previous = current;
    }
    return length;
}

function cubicBSplinePoint(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: number): Vec2 {
    const t2 = t * t;
    const t3 = t2 * t;
    const b0 = (-t3 + 3 * t2 - 3 * t + 1) / 6;
    const b1 = (3 * t3 - 6 * t2 + 4) / 6;
    const b2 = (-3 * t3 + 3 * t2 + 3 * t + 1) / 6;
    const b3 = t3 / 6;

    return {
        x: p0.x * b0 + p1.x * b1 + p2.x * b2 + p3.x * b3,
        y: p0.y * b0 + p1.y * b1 + p2.y * b2 + p3.y * b3,
    };
}

function polylineLength(points: Vec2[]): number {
    let length = 0;
    for (let i = 1; i < points.length; i++) {
        length += distance2d(points[i - 1], points[i]);
    }
    return length;
}