import {
    CONTOUR_STROKE_KIND_WEIGHTS,
    type ContourStroke,
    type ContourStrokeKind,
    type LegacyContourRole,
    type Vec2,
} from "./types.ts";

export function contourStrokeKindWeight(kind: ContourStrokeKind): number {
    return CONTOUR_STROKE_KIND_WEIGHTS[kind];
}

export function normalizeContourStrokeKind(kind: ContourStrokeKind | LegacyContourRole): ContourStrokeKind {
    return kind === "interior" || kind === "contour" ? "contour" : "edge";
}

export function ndcFromClientPoint(
    clientX: number,
    clientY: number,
    rect: Pick<DOMRect, "left" | "top" | "width" | "height">,
): Vec2 {
    return {
        x: ((clientX - rect.left) / rect.width) * 2 - 1,
        y: -(((clientY - rect.top) / rect.height) * 2 - 1),
    };
}

export function clampNdcPoint(p: Vec2): Vec2 {
    return {
        x: Math.max(-1, Math.min(1, p.x)),
        y: Math.max(-1, Math.min(1, p.y)),
    };
}

export function distance2(a: Vec2, b: Vec2): number {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return dx * dx + dy * dy;
}

export function polylineLength(points: Vec2[]): number {
    let length = 0;
    for (let i = 1; i < points.length; i++) {
        length += Math.sqrt(distance2(points[i - 1], points[i]));
    }
    return length;
}

export function resamplePolyline(points: Vec2[], maxSamples = 96): Vec2[] {
    if (points.length <= 2) return points.slice();

    const length = polylineLength(points);
    if (length <= 1e-6) return [points[0]];

    const sampleCount = Math.max(2, Math.min(maxSamples, Math.ceil(length / 0.015)));
    const spacing = length / (sampleCount - 1);
    const out: Vec2[] = [points[0]];

    let segmentStart = points[0];
    let segmentEndIndex = 1;
    let segmentEnd = points[segmentEndIndex];
    let segmentLength = Math.sqrt(distance2(segmentStart, segmentEnd));
    let distanceIntoSegment = 0;

    for (let s = 1; s < sampleCount - 1; s++) {
        const targetDistance = s * spacing;

        while (distanceIntoSegment + segmentLength < targetDistance && segmentEndIndex < points.length - 1) {
            distanceIntoSegment += segmentLength;
            segmentStart = segmentEnd;
            segmentEndIndex += 1;
            segmentEnd = points[segmentEndIndex];
            segmentLength = Math.sqrt(distance2(segmentStart, segmentEnd));
        }

        const local = segmentLength <= 1e-6
            ? 0
            : (targetDistance - distanceIntoSegment) / segmentLength;
        out.push({
            x: segmentStart.x + (segmentEnd.x - segmentStart.x) * local,
            y: segmentStart.y + (segmentEnd.y - segmentStart.y) * local,
        });
    }

    out.push(points[points.length - 1]);
    return out;
}

export function estimateTangents(points: Vec2[]): Vec2[] {
    return points.map((point, i) => {
        const prev = points[Math.max(0, i - 1)];
        const next = points[Math.min(points.length - 1, i + 1)];
        const dx = next.x - prev.x;
        const dy = next.y - prev.y;
        const len = Math.hypot(dx, dy);
        if (len <= 1e-6) return { x: 1, y: 0 };
        return { x: dx / len, y: dy / len };
    });
}

export function estimateNormals(tangents: Vec2[]): Vec2[] {
    return tangents.map(t => ({ x: -t.y, y: t.x }));
}

export function isClosedPolyline(points: Vec2[]): boolean {
    if (points.length < 4) return false;
    const length = polylineLength(points);
    if (length <= 1e-6) return false;
    const endpointGap = Math.sqrt(distance2(points[0], points[points.length - 1]));
    return endpointGap < Math.max(0.045, length * 0.08);
}

export function makeContourStroke({
    id,
    kind,
    role,
    viewId,
    shapeId,
    points,
    depthNdc,
    depthOffset = 0,
    depthLocked = false,
}: {
    id: string;
    kind?: ContourStrokeKind | LegacyContourRole;
    role?: LegacyContourRole;
    viewId: string;
    shapeId: string;
    points: Vec2[];
    depthNdc?: number;
    depthOffset?: number;
    depthLocked?: boolean;
}): ContourStroke {
    const strokeKind = normalizeContourStrokeKind(kind ?? role ?? "edge");
    const resampledPoints = resamplePolyline(points);
    const tangents = estimateTangents(resampledPoints);
    return {
        id,
        kind: strokeKind,
        viewId,
        shapeId,
        points,
        resampledPoints,
        tangents,
        normals: estimateNormals(tangents),
        weight: contourStrokeKindWeight(strokeKind),
        depthNdc,
        depthOffset,
        depthLocked,
    };
}
