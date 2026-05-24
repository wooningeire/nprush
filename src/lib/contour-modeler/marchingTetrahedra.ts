import type { MeshData } from "$/gpu/file-load/loadGlb";
import { bodyAxes, implicitBodyNormal, implicitBodyPoint } from "./implicitBody.ts";
import type { ImplicitBodyParams } from "./types.ts";

export interface MeshExtractionOptions {
    resolution?: number;
    color?: [number, number, number, number];
}

/**
 * Meshes the current spine/radial implicit body as a stable parametric surface.
 *
 * The first contour modeler version used marching tetrahedra directly against
 * the SDF grid. That was technically generic, but it produced distracting
 * cracks and inconsistent facets for the tapered cone workflow. This body is
 * already represented as rings along a spine, so a lathe-style surface is the
 * correct preview mesh for the artist-facing v1.
 */
export function extractImplicitBodyMesh(
    params: ImplicitBodyParams,
    options: MeshExtractionOptions = {},
): MeshData {
    const resolution = options.resolution ?? 34;
    const rings = Math.max(8, resolution);
    const segments = Math.max(24, Math.ceil(resolution * 1.5));
    const color = options.color ?? [0.74, 0.8, 0.86, 1];
    const vertices: number[] = [];
    const indices: number[] = [];

    for (let r = 0; r <= rings; r++) {
        const t = r / rings;
        for (let s = 0; s < segments; s++) {
            const theta = s / segments * Math.PI * 2;
            addSurfaceVertex(params, t, theta, color, vertices);
        }
    }

    for (let r = 0; r < rings; r++) {
        for (let s = 0; s < segments; s++) {
            const nextS = (s + 1) % segments;
            const a = ringIndex(r, s, segments);
            const b = ringIndex(r, nextS, segments);
            const c = ringIndex(r + 1, s, segments);
            const d = ringIndex(r + 1, nextS, segments);

            // Wound for outward normals in the app's coordinate system.
            indices.push(a, d, b);
            indices.push(a, c, d);
        }
    }

    const bottomCenter = vertices.length / 12;
    addCapVertex(params, 0, [0, -1, 0], color, vertices);
    const topCenter = vertices.length / 12;
    addCapVertex(params, 1, [0, 1, 0], color, vertices);

    for (let s = 0; s < segments; s++) {
        const nextS = (s + 1) % segments;
        indices.push(bottomCenter, ringIndex(0, s, segments), ringIndex(0, nextS, segments));
        indices.push(topCenter, ringIndex(rings, nextS, segments), ringIndex(rings, s, segments));
    }

    return {
        vertices: new Float32Array(vertices),
        indices: new Uint32Array(indices),
        hasUvs: false,
    };
}

function ringIndex(ring: number, segment: number, segments: number): number {
    return ring * segments + segment;
}

function addSurfaceVertex(
    params: ImplicitBodyParams,
    t: number,
    theta: number,
    color: [number, number, number, number],
    vertices: number[],
) {
    const p = implicitBodyPoint(params, t, theta);
    const n = implicitBodyNormal(params, p);
    vertices.push(
        p[0], p[1], p[2],
        n[0], n[1], n[2],
        color[0], color[1], color[2], color[3],
        t, theta / (Math.PI * 2),
    );
}

function addCapVertex(
    params: ImplicitBodyParams,
    t: number,
    normal: [number, number, number],
    color: [number, number, number, number],
    vertices: number[],
) {
    const axes = bodyAxes(params);
    const y = (t - 0.5) * params.height;
    const worldNormal: [number, number, number] = [
        axes.y[0] * normal[1],
        axes.y[1] * normal[1],
        axes.y[2] * normal[1],
    ];
    vertices.push(
        params.center[0] + axes.y[0] * y,
        params.center[1] + axes.y[1] * y,
        params.center[2] + axes.y[2] * y,
        worldNormal[0], worldNormal[1], worldNormal[2],
        color[0], color[1], color[2], color[3],
        0.5, t,
    );
}
