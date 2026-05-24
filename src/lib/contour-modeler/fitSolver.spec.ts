import { describe, expect, it } from "vitest";
import { makeContourStroke } from "./contourGeometry.ts";
import { DEFAULT_IMPLICIT_BODY_PARAMS } from "./implicitBody.ts";
import { estimateInitialParamsFromContours, fitImplicitBody } from "./fitSolver.ts";
import { buildRenderedFitTarget, evaluateRenderedFeatureLoss } from "./renderedFeatureLoss.ts";
import type { ContourStroke, FitView, Vec2 } from "./types.ts";

const IDENTITY_VIEW: FitView = {
    id: "front",
    viewProjMat: [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ],
    width: 512,
    height: 512,
};

describe("fit solver", () => {
    it("reduces loss for synthetic cone-like contour strokes", async () => {
        const strokes = syntheticConeStrokes();
        const target = buildRenderedFitTarget(strokes, [IDENTITY_VIEW]);
        const initialLoss = evaluateRenderedFeatureLoss(DEFAULT_IMPLICIT_BODY_PARAMS, target);

        const result = await fitImplicitBody({
            strokes,
            views: [IDENTITY_VIEW],
            meshResolution: 14,
            iterations: 5,
            candidatesPerIteration: 18,
        });

        expect(result.loss).toBeLessThan(initialLoss);
        expect(result.params.radiusBottom).toBeGreaterThan(result.params.radiusTop);
        expect(result.mesh.indices.length).toBeGreaterThan(0);
    });

    it("uses edited stroke depth when estimating the initial body", () => {
        const stroke = makeContourStroke({
            id: "depth-edge",
            kind: "edge",
            viewId: "front",
            shapeId: "shape",
            depthNdc: 0.8,
            depthLocked: true,
            points: [
                { x: -0.2, y: -0.2 },
                { x: 0.2, y: -0.2 },
                { x: 0.2, y: 0.2 },
                { x: -0.2, y: 0.2 },
            ],
        });

        const params = estimateInitialParamsFromContours([stroke], [{
            ...IDENTITY_VIEW,
            viewProjInvMat: IDENTITY_VIEW.viewProjMat,
        }]);

        expect(params.center[2]).toBeGreaterThan(0.2);
    });
});

function syntheticConeStrokes(): ContourStroke[] {
    return [
        makeContourStroke({
            id: "left-edge",
            kind: "edge",
            viewId: "front",
            shapeId: "shape",
            points: [
                { x: -0.05, y: 0.52 },
                { x: -0.35, y: -0.52 },
            ],
        }),
        makeContourStroke({
            id: "right-edge",
            kind: "edge",
            viewId: "front",
            shapeId: "shape",
            points: [
                { x: 0.05, y: 0.52 },
                { x: 0.35, y: -0.52 },
            ],
        }),
        makeContourStroke({
            id: "base-ring",
            kind: "contour",
            viewId: "front",
            shapeId: "shape",
            points: ellipsePoints(0, -0.44, 0.34, 0.08),
        }),
        makeContourStroke({
            id: "mid-ring",
            kind: "contour",
            viewId: "front",
            shapeId: "shape",
            points: ellipsePoints(0, -0.08, 0.22, 0.06),
        }),
    ];
}

function ellipsePoints(cx: number, cy: number, rx: number, ry: number): Vec2[] {
    const points: Vec2[] = [];
    for (let i = 0; i <= 24; i++) {
        const theta = i / 24 * Math.PI * 2;
        points.push({
            x: cx + Math.cos(theta) * rx,
            y: cy + Math.sin(theta) * ry,
        });
    }
    return points;
}
