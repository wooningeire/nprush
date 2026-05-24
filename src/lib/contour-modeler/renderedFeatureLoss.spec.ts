import { describe, expect, it } from "vitest";
import { makeContourStroke } from "./contourGeometry.ts";
import {
    buildRenderedFitTarget,
    evaluateRenderedFeatureLoss,
    strokeDistanceAt,
} from "./renderedFeatureLoss.ts";
import type { FitView, ImplicitBodyParams, Vec2 } from "./types.ts";

const FRONT_VIEW: FitView = {
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

const TOP_RING_VIEW: FitView = {
    id: "top",
    viewProjMat: [
        1, 0, 0, 0,
        0, 0, 0, 0,
        0, 1, 1, 0,
        0, 0, 0, 1,
    ],
    width: 512,
    height: 512,
};

describe("rendered feature loss", () => {
    it("builds distance fields for edge strokes", () => {
        const stroke = makeContourStroke({
            id: "edge",
            kind: "edge",
            viewId: "front",
            shapeId: "shape",
            points: [
                { x: -0.5, y: 0 },
                { x: 0.5, y: 0 },
            ],
        });

        const target = buildRenderedFitTarget([stroke], [FRONT_VIEW], 64);

        expect(strokeDistanceAt(target.views[0].edgeField, { x: 0, y: 0 })).toBeLessThan(0.02);
        expect(strokeDistanceAt(target.views[0].edgeField, { x: 0, y: 0.6 })).toBeGreaterThan(0.4);
    });

    it("gives a lower edge loss to a candidate with matching silhouette width", () => {
        const strokes = [
            makeContourStroke({
                id: "left",
                kind: "edge",
                viewId: "front",
                shapeId: "shape",
                points: [
                    { x: -0.35, y: -0.5 },
                    { x: -0.05, y: 0.5 },
                ],
            }),
            makeContourStroke({
                id: "right",
                kind: "edge",
                viewId: "front",
                shapeId: "shape",
                points: [
                    { x: 0.35, y: -0.5 },
                    { x: 0.05, y: 0.5 },
                ],
            }),
        ];
        const target = buildRenderedFitTarget(strokes, [FRONT_VIEW], 96);

        expect(evaluateRenderedFeatureLoss(coneParams(0.35, 0.05), target))
            .toBeLessThan(evaluateRenderedFeatureLoss(coneParams(0.12, 0.03), target));
    });

    it("uses the projected outline, including top and bottom edge spans", () => {
        const strokes = [
            makeContourStroke({
                id: "top",
                kind: "edge",
                viewId: "front",
                shapeId: "shape",
                points: [
                    { x: -0.35, y: 0.5 },
                    { x: 0.35, y: 0.5 },
                ],
            }),
            makeContourStroke({
                id: "bottom",
                kind: "edge",
                viewId: "front",
                shapeId: "shape",
                points: [
                    { x: -0.35, y: -0.5 },
                    { x: 0.35, y: -0.5 },
                ],
            }),
        ];
        const target = buildRenderedFitTarget(strokes, [FRONT_VIEW], 96);

        expect(evaluateRenderedFeatureLoss(cylinderParams(0.35, 1, 1, 1), target))
            .toBeLessThan(evaluateRenderedFeatureLoss(cylinderParams(0.35, 1, 1, 0.35), target));
    });

    it("uses closed contour strokes as ring constraints", () => {
        const contour = makeContourStroke({
            id: "ring",
            kind: "contour",
            viewId: "top",
            shapeId: "shape",
            points: ellipsePoints(0, 0, 0.28, 0.18),
        });
        const target = buildRenderedFitTarget([contour], [TOP_RING_VIEW], 96);

        expect(evaluateRenderedFeatureLoss(cylinderParams(0.28, 1, 0.65, 1), target))
            .toBeLessThan(evaluateRenderedFeatureLoss(cylinderParams(0.08, 1, 0.65, 1), target));
    });
});

function coneParams(radiusBottom: number, radiusTop: number): ImplicitBodyParams {
    return {
        center: [0, 0, 0],
        axisX: [1, 0, 0],
        axisY: [0, 1, 0],
        axisZ: [0, 0, 1],
        height: 1,
        radiusBottom,
        radiusTop,
        bulge: 0,
        ovalX: 1,
        ovalZ: 1,
        boxiness: 0,
    };
}

function cylinderParams(radius: number, ovalX: number, ovalZ: number, height: number): ImplicitBodyParams {
    return {
        center: [0, 0, 0],
        axisX: [1, 0, 0],
        axisY: [0, 1, 0],
        axisZ: [0, 0, 1],
        height,
        radiusBottom: radius,
        radiusTop: radius,
        bulge: 0,
        ovalX,
        ovalZ,
        boxiness: 0,
    };
}

function ellipsePoints(cx: number, cy: number, rx: number, ry: number): Vec2[] {
    const points: Vec2[] = [];
    for (let i = 0; i <= 36; i++) {
        const theta = i / 36 * Math.PI * 2;
        points.push({
            x: cx + Math.cos(theta) * rx,
            y: cy + Math.sin(theta) * ry,
        });
    }
    return points;
}
