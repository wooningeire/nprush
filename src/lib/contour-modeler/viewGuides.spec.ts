import { describe, expect, it } from "vitest";
import { makeContourStroke } from "./contourGeometry.ts";
import { buildCrossViewGuides } from "./viewGuides.ts";
import type { ContourView } from "./types.ts";

const IDENTITY = [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
];

const VIEW: ContourView = {
    id: "view-a",
    name: "View A",
    long: 0,
    lat: 0,
    radius: 1,
    offset: [0, 0, 0],
    width: 800,
    height: 600,
    viewProjMat: IDENTITY,
    viewProjInvMat: IDENTITY,
    viewMat: IDENTITY,
    viewInvMat: IDENTITY,
    createdAt: 0,
};

describe("cross-view guides", () => {
    it("projects saved strokes into the current camera as proxy guides", () => {
        const stroke = makeContourStroke({
            id: "stroke-a",
            kind: "edge",
            viewId: "view-a",
            shapeId: "shape-a",
            points: [
                { x: -0.3, y: 0.2 },
                { x: 0.3, y: -0.2 },
            ],
        });

        const guides = buildCrossViewGuides({
            strokes: [stroke],
            views: [VIEW],
            currentViewProjMat: IDENTITY,
        });

        const proxy = guides.find(guide => guide.style === "proxy");
        expect(proxy?.points.length).toBeGreaterThan(1);
        expect(proxy?.points[0].x).toBeCloseTo(-0.3, 2);
        expect(proxy?.kind).toBe("edge");
    });

    it("moves proxy guides when stroke depth is edited", () => {
        const stroke = makeContourStroke({
            id: "stroke-a",
            kind: "edge",
            viewId: "view-a",
            shapeId: "shape-a",
            depthNdc: 0.2,
            depthLocked: true,
            points: [
                { x: 0.4, y: 0 },
                { x: 0.5, y: 0.1 },
            ],
        });
        const perspectiveCurrentView = [
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 1,
            0, 0, 0, 1,
        ];

        const nearProxy = buildCrossViewGuides({
            strokes: [stroke],
            views: [VIEW],
            currentViewProjMat: perspectiveCurrentView,
        }).find(guide => guide.style === "proxy");
        const farProxy = buildCrossViewGuides({
            strokes: [{ ...stroke, depthNdc: 0.8 }],
            views: [VIEW],
            currentViewProjMat: perspectiveCurrentView,
        }).find(guide => guide.style === "proxy");

        expect(nearProxy?.points[0].x).not.toBeCloseTo(farProxy?.points[0].x ?? 0, 3);
        expect(nearProxy?.strokeId).toBe("stroke-a");
    });

    it("supports per-vertex depth offsets for brushed guide bends", () => {
        const stroke = makeContourStroke({
            id: "stroke-a",
            kind: "edge",
            viewId: "view-a",
            shapeId: "shape-a",
            depthNdc: 0.2,
            depthOffset: 0,
            points: [
                { x: 0.2, y: 0 },
                { x: 0.3, y: 0 },
                { x: 0.4, y: 0 },
            ],
        });
        stroke.depthSamplesOffset = stroke.resampledPoints.map((_, index) => index === stroke.resampledPoints.length - 1 ? 0.55 : 0);
        stroke.depthSamplesLocked = stroke.resampledPoints.map((_, index) => index === stroke.resampledPoints.length - 1);

        const guides = buildCrossViewGuides({
            strokes: [stroke],
            views: [VIEW],
            currentViewProjMat: [
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 1,
                0, 0, 0, 1,
            ],
        });

        const proxy = guides.find(guide => guide.style === "proxy");
        expect(proxy?.vertices?.some(vertex => vertex.depthOffset > 0.5)).toBe(true);
        expect(proxy?.vertices?.some(vertex => vertex.depthOffset < 0.1)).toBe(true);
        expect(proxy?.vertices?.some(vertex => vertex.depthDirection)).toBe(true);
    });
});
