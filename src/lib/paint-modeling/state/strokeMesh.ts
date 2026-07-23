import {
    defaultDepthForProjection,
    viewPointToWorldAtDepth,
} from "./projection.ts";
import { distance2d, samplePaintStrokeSpline } from "./strokeSampling.ts";
import {
    distance3,
    sub3,
} from "./vectorMath.ts";
import type {
    PaintRibbon,
    PaintRibbonVertex,
    ProjectionSnapshot,
    Vec2,
    Vec3,
} from "../types.ts";

export type RibbonBuildResult = {
    ribbon: PaintRibbon,
};

export const buildRibbonStrokeGeometry = (
    sourcePoints: Vec2[],
    sourceProjection: ProjectionSnapshot,
    width: number,
): RibbonBuildResult | null => {
    const closed = isClosedSourceStroke(sourcePoints);
    const points = closed ? sourcePoints.slice(0, -1) : sourcePoints;
    if (points.length < 2) return null;

    const depth = defaultDepthForProjection(sourceProjection);
    const centers = points
        .map(point => viewPointToWorldAtDepth(sourceProjection, point, depth))
        .filter((point): point is Vec3 => point !== null);
    if (centers.length < 2) return null;

    const uValues = normalizedPolylineU(centers, closed);
    const vertices: PaintRibbonVertex[] = centers.map((center, index) => ({
        position: center,
        side: sideVectorAt(
            sourceProjection,
            points[index],
            depth,
            ribbonSideOffsetAt(points, index, sourceProjection, width, closed),
            center,
        ),
        u: uValues[index],
    }));

    return {
        ribbon: {
            closed,
            vertices,
        },
    };
};

export const buildRibbonGeometryFromDraft = (
    draftStroke: Vec2[],
    sourceProjection: ProjectionSnapshot,
    width: number,
): RibbonBuildResult | null => buildRibbonStrokeGeometry(
    samplePaintStrokeSpline(draftStroke),
    sourceProjection,
    width,
);

export const ribbonSegmentCount = (ribbon: PaintRibbon): number => {
    if (ribbon.vertices.length < 2) return 0;
    return ribbon.closed ? ribbon.vertices.length : ribbon.vertices.length - 1;
};

const normalizedPolylineU = (points: Vec3[], closed: boolean): number[] => {
    const distances = [0];
    let total = 0;
    for (let index = 1; index < points.length; index++) {
        total += distance3(points[index - 1], points[index]);
        distances.push(total);
    }
    if (closed && points.length > 2) {
        total += distance3(points.at(-1)!, points[0]);
    }
    if (total <= 1e-8) {
        return points.map((_, index) => points.length <= 1 ? 0 : index / (points.length - 1));
    }
    return distances.map(distance => distance / total);
};

const ribbonSideOffsetAt = (
    points: Vec2[],
    index: number,
    projection: ProjectionSnapshot,
    width: number,
    closed: boolean,
): Vec2 => {
    const current = points[index];
    const previous = index > 0
        ? points[index - 1]
        : closed
            ? points.at(-1)!
            : current;
    const next = index < points.length - 1
        ? points[index + 1]
        : closed
            ? points[0]
            : current;
    const dxPx = (next.x - previous.x) * projection.width * 0.5;
    const dyPx = (next.y - previous.y) * projection.height * 0.5;
    const lengthPx = Math.hypot(dxPx, dyPx);
    if (lengthPx <= 1e-6) return { x: 0, y: Math.max(width, 1) / projection.height };

    const halfWidthPx = Math.max(width, 1) * 0.5;
    return {
        x: -dyPx / lengthPx * halfWidthPx * 2 / projection.width,
        y: dxPx / lengthPx * halfWidthPx * 2 / projection.height,
    };
};

const sideVectorAt = (
    sourceProjection: ProjectionSnapshot,
    sourcePoint: Vec2,
    depth: number,
    sideOffset: Vec2,
    center: Vec3,
): Vec3 => {
    const sideWorld = viewPointToWorldAtDepth(
        sourceProjection,
        {
            x: sourcePoint.x + sideOffset.x,
            y: sourcePoint.y + sideOffset.y,
        },
        depth,
    );
    if (!sideWorld) return [0, 0, 0];
    return sub3(sideWorld, center);
};

const isClosedSourceStroke = (points: Vec2[]): boolean => (
    points.length > 3 && distance2d(points[0], points.at(-1)!) <= 0.04
);
