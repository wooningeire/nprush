import { describe, expect, it } from "vitest";
import {
    contourStrokeKindWeight,
    isClosedPolyline,
    makeContourStroke,
    normalizeContourStrokeKind,
    resamplePolyline,
} from "./contourGeometry.ts";

describe("contour geometry", () => {
    it("resamples drawn strokes and preserves endpoints", () => {
        const points = [
            { x: -0.5, y: -0.5 },
            { x: 0, y: 0.5 },
            { x: 0.5, y: -0.5 },
        ];

        const resampled = resamplePolyline(points, 24);

        expect(resampled.length).toBeGreaterThan(points.length);
        expect(resampled[0]).toEqual(points[0]);
        expect(resampled[resampled.length - 1]).toEqual(points[points.length - 1]);
    });

    it("assigns contour kind weights, migrates old roles, and builds tangent metadata", () => {
        const stroke = makeContourStroke({
            id: "stroke-a",
            role: "occluded",
            viewId: "view-a",
            shapeId: "shape-a",
            points: [
                { x: -0.2, y: 0 },
                { x: 0.2, y: 0 },
            ],
        });

        expect(normalizeContourStrokeKind("exterior")).toBe("edge");
        expect(normalizeContourStrokeKind("occluded")).toBe("edge");
        expect(normalizeContourStrokeKind("interior")).toBe("contour");
        expect(contourStrokeKindWeight("edge")).toBe(1);
        expect(contourStrokeKindWeight("contour")).toBe(0.85);
        expect(stroke.kind).toBe("edge");
        expect(stroke.weight).toBe(1);
        expect(stroke.tangents.length).toBe(stroke.resampledPoints.length);
        expect(stroke.normals.length).toBe(stroke.resampledPoints.length);
    });

    it("detects closed contour polylines", () => {
        expect(isClosedPolyline([
            { x: 0.2, y: 0 },
            { x: 0, y: 0.2 },
            { x: -0.2, y: 0 },
            { x: 0, y: -0.2 },
            { x: 0.21, y: 0.01 },
        ])).toBe(true);
        expect(isClosedPolyline([
            { x: -0.2, y: 0 },
            { x: 0.2, y: 0 },
        ])).toBe(false);
    });
});
