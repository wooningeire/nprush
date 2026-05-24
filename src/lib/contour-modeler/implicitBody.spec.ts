import { describe, expect, it } from "vitest";
import { implicitBodyNormal, implicitBodyPoint, implicitBodySdf } from "./implicitBody.ts";
import { extractImplicitBodyMesh } from "./marchingTetrahedra.ts";
import type { ImplicitBodyParams } from "./types.ts";

describe("implicit body", () => {
    it("evaluates inside, surface, and outside distances", () => {
        const params: ImplicitBodyParams = {
            center: [0, 0, 0],
            height: 1,
            radiusBottom: 0.4,
            radiusTop: 0.4,
            bulge: 0,
            ovalX: 1,
            ovalZ: 1,
            boxiness: 0,
        };

        expect(implicitBodySdf(params, [0, 0, 0])).toBeLessThan(0);
        expect(Math.abs(implicitBodySdf(params, [0.4, 0, 0]))).toBeLessThan(0.01);
        expect(implicitBodySdf(params, [0.8, 0, 0])).toBeGreaterThan(0);
    });

    it("returns normalized finite normals", () => {
        const normal = implicitBodyNormal({
            center: [0, 0, 0],
            height: 1,
            radiusBottom: 0.5,
            radiusTop: 0.2,
            bulge: 0.03,
            ovalX: 1,
            ovalZ: 1,
            boxiness: 0,
        }, [0.35, -0.2, 0]);

        expect(Math.hypot(normal[0], normal[1], normal[2])).toBeCloseTo(1, 2);
    });

    it("extracts a tapered nonempty mesh", () => {
        const mesh = extractImplicitBodyMesh({
            center: [0, 0, 0],
            height: 1.2,
            radiusBottom: 0.45,
            radiusTop: 0.08,
            bulge: 0,
            ovalX: 1,
            ovalZ: 1,
            boxiness: 0,
        }, { resolution: 20 });

        const topRadius = maxBandRadius(mesh.vertices, 0.3, Infinity);
        const bottomRadius = maxBandRadius(mesh.vertices, -Infinity, -0.3);

        expect(mesh.vertices.length).toBeGreaterThan(0);
        expect(mesh.indices.length).toBeGreaterThan(0);
        expect(bottomRadius).toBeGreaterThan(topRadius * 1.8);
    });

    it("supports boxier cross sections for straight-edged forms", () => {
        const round = implicitBodyPoint({
            center: [0, 0, 0],
            height: 1,
            radiusBottom: 0.4,
            radiusTop: 0.4,
            bulge: 0,
            ovalX: 1,
            ovalZ: 1,
            boxiness: 0,
        }, 0.5, Math.PI * 0.25);
        const boxy = implicitBodyPoint({
            center: [0, 0, 0],
            height: 1,
            radiusBottom: 0.4,
            radiusTop: 0.4,
            bulge: 0,
            ovalX: 1,
            ovalZ: 1,
            boxiness: 0.8,
        }, 0.5, Math.PI * 0.25);

        expect(Math.hypot(boxy[0], boxy[2])).toBeGreaterThan(Math.hypot(round[0], round[2]) * 1.2);
    });
});

function maxBandRadius(vertices: Float32Array, minY: number, maxY: number): number {
    let radius = 0;
    for (let i = 0; i < vertices.length / 12; i++) {
        const x = vertices[i * 12 + 0];
        const y = vertices[i * 12 + 1];
        const z = vertices[i * 12 + 2];
        if (y >= minY && y <= maxY) {
            radius = Math.max(radius, Math.hypot(x, z));
        }
    }
    return radius;
}
