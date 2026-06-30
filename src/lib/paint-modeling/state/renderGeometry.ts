import {
    cross3,
    normalize3,
    sub3,
} from "./vectorMath.ts";
import type {
    PaintRibbonMesh,
    PaintStroke,
    RenderPrimitive,
    Vec3,
    Vec4,
} from "../types.ts";

export const appendStrokeRenderTriangles = (
    segments: RenderPrimitive[],
    stroke: PaintStroke,
    shadeRibbons: boolean,
) => {
    appendRibbonMeshTriangles(
        segments,
        stroke.mesh,
        parseColor(stroke.style.color, stroke.style.opacity),
        shadeRibbons ? 1 : 0,
    );
};

export const appendRibbonMeshTriangles = (
    segments: RenderPrimitive[],
    mesh: PaintRibbonMesh,
    color: Vec4,
    shade: number,
) => {
    for (const face of mesh.faces) {
        const a = mesh.vertices[face[0]].position;
        const b = mesh.vertices[face[1]].position;
        const c = mesh.vertices[face[2]].position;
        const d = mesh.vertices[face[3]].position;
        appendWorldTriangle(segments, a, b, c, color, shade);
        appendWorldTriangle(segments, a, c, d, color, shade);
    }
};

export const appendWorldStrokeRun = (
    segments: RenderPrimitive[],
    points: Array<Vec3 | null>,
    color: Vec4,
    width: number,
) => {
    let run: Vec3[] = [];
    const flushRun = () => {
        if (run.length >= 2) {
            segments.push({
                kind: "stroke",
                points: run,
                color,
                width,
            });
        }
        run = [];
    };

    for (const point of points) {
        if (point) {
            run.push(point);
        } else {
            flushRun();
        }
    }
    flushRun();
};

export const parseColor = (color: string, opacity: number): Vec4 => {
    const value = color.startsWith("#") ? color.slice(1) : color;
    const r = parseInt(value.slice(0, 2), 16) / 255;
    const g = parseInt(value.slice(2, 4), 16) / 255;
    const b = parseInt(value.slice(4, 6), 16) / 255;
    return [r, g, b, opacity];
};

const appendWorldTriangle = (
    segments: RenderPrimitive[],
    a: Vec3,
    b: Vec3,
    c: Vec3,
    color: Vec4,
    shade: number,
) => {
    segments.push({
        kind: "triangle",
        a,
        b,
        c,
        color,
        normal: triangleNormal(a, b, c),
        shade,
    });
};

const triangleNormal = (a: Vec3, b: Vec3, c: Vec3): Vec3 => {
    return normalize3(cross3(sub3(b, a), sub3(c, a)), [0, 0, 1]);
};
