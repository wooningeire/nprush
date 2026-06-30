import type { RenderPrimitive, RenderSegment, RenderStroke, RenderTriangle, Vec3 } from "../types.ts";
import {
    FLOATS_PER_TRIANGLE_VERTEX,
    FLOATS_PER_VERTEX,
    VERTICES_PER_SEGMENT,
    VERTICES_PER_TRIANGLE,
} from "./constants.ts";

const DEFAULT_TRIANGLE_NORMAL: Vec3 = [0, 0, 1];

export const isRenderSegment = (primitive: RenderPrimitive): primitive is RenderSegment => (
    primitive.kind !== "triangle" && primitive.kind !== "stroke"
);

export const isRenderTriangle = (primitive: RenderPrimitive): primitive is RenderTriangle => primitive.kind === "triangle";

export const isRenderStroke = (primitive: RenderPrimitive): primitive is RenderStroke => primitive.kind === "stroke";

export const createSegmentData = (segments: RenderSegment[]): Float32Array => {
    const data = new Float32Array(segments.length * VERTICES_PER_SEGMENT * FLOATS_PER_VERTEX);
    let offset = 0;
    for (const segment of segments) {
        offset = appendSegment(data, offset, segment);
    }
    return data;
};

export const createStrokeData = (strokes: RenderStroke[]): Float32Array => {
    const data = new Float32Array(strokeStripVertexCount(strokes) * FLOATS_PER_VERTEX);
    appendStrokeStrips(data, 0, strokes);
    return data;
};

export const createTriangleData = (triangles: RenderTriangle[]): Float32Array => {
    const data = new Float32Array(triangles.length * VERTICES_PER_TRIANGLE * FLOATS_PER_TRIANGLE_VERTEX);
    let offset = 0;
    for (const triangle of triangles) {
        offset = appendTriangle(data, offset, triangle);
    }
    return data;
};

export const strokeStripVertexCount = (strokes: RenderStroke[]): number => {
    let count = 0;
    let hasPreviousRun = false;
    for (const stroke of strokes) {
        if (stroke.points.length < 2) continue;
        if (hasPreviousRun) count += 2;
        count += stroke.points.length * 2;
        hasPreviousRun = true;
    }
    return count;
};

const appendSegment = (
    out: Float32Array,
    offset: number,
    segment: RenderSegment,
): number => {
    const width = segment.width ?? 1.25;
    offset = appendSegmentVertex(out, offset, segment, 0, -1, width);
    offset = appendSegmentVertex(out, offset, segment, 1, -1, width);
    offset = appendSegmentVertex(out, offset, segment, 1, 1, width);
    offset = appendSegmentVertex(out, offset, segment, 0, -1, width);
    offset = appendSegmentVertex(out, offset, segment, 1, 1, width);
    offset = appendSegmentVertex(out, offset, segment, 0, 1, width);
    return offset;
};

const appendStrokeStrips = (
    out: Float32Array,
    offset: number,
    strokes: RenderStroke[],
): number => {
    let previousStroke: RenderStroke | null = null;
    for (const stroke of strokes) {
        if (stroke.points.length < 2) continue;
        if (previousStroke) {
            offset = appendStrokeVertex(out, offset, previousStroke, previousStroke.points.length - 1, 1);
            offset = appendStrokeVertex(out, offset, stroke, 0, -1);
        }
        for (let i = 0; i < stroke.points.length; i++) {
            offset = appendStrokeVertex(out, offset, stroke, i, -1);
            offset = appendStrokeVertex(out, offset, stroke, i, 1);
        }
        previousStroke = stroke;
    }
    return offset;
};

const appendTriangle = (
    out: Float32Array,
    offset: number,
    triangle: RenderTriangle,
): number => {
    const normal = triangle.normal ?? DEFAULT_TRIANGLE_NORMAL;
    const shade = triangle.shade ?? 0;
    offset = appendTriangleVertex(out, offset, triangle.a, triangle.color, normal, shade);
    offset = appendTriangleVertex(out, offset, triangle.b, triangle.color, normal, shade);
    offset = appendTriangleVertex(out, offset, triangle.c, triangle.color, normal, shade);
    return offset;
};

const appendTriangleVertex = (
    out: Float32Array,
    offset: number,
    position: Vec3,
    color: [number, number, number, number],
    normal: Vec3,
    shade: number,
): number => {
    out[offset++] = position[0];
    out[offset++] = position[1];
    out[offset++] = position[2];
    out[offset++] = color[0];
    out[offset++] = color[1];
    out[offset++] = color[2];
    out[offset++] = color[3];
    out[offset++] = normal[0];
    out[offset++] = normal[1];
    out[offset++] = normal[2];
    out[offset++] = shade;
    return offset;
};

const appendSegmentVertex = (
    out: Float32Array,
    offset: number,
    segment: RenderSegment,
    along: number,
    side: number,
    width: number,
): number => {
    const point = along < 0.5 ? segment.a : segment.b;
    const cap = along < 0.5
        ? segment.capStart === false ? 0 : -1
        : segment.capEnd === false ? 0 : 1;
    return appendJoinVertex(
        out,
        offset,
        segment.a,
        point,
        segment.b,
        segment.color,
        side,
        width,
        cap,
    );
};

const appendStrokeVertex = (
    out: Float32Array,
    offset: number,
    stroke: RenderStroke,
    index: number,
    side: number,
): number => {
    const point = stroke.points[index];
    const cap = index === 0
        ? -1
        : index === stroke.points.length - 1
            ? 1
            : 0;
    return appendJoinVertex(
        out,
        offset,
        stroke.points[index - 1] ?? point,
        point,
        stroke.points[index + 1] ?? point,
        stroke.color,
        side,
        stroke.width,
        cap,
    );
};

const appendJoinVertex = (
    out: Float32Array,
    offset: number,
    joinPrev: Vec3,
    joinPoint: Vec3,
    joinNext: Vec3,
    color: [number, number, number, number],
    side: number,
    width: number,
    cap: number,
): number => {
    out[offset++] = joinPrev[0];
    out[offset++] = joinPrev[1];
    out[offset++] = joinPrev[2];
    out[offset++] = joinPoint[0];
    out[offset++] = joinPoint[1];
    out[offset++] = joinPoint[2];
    out[offset++] = joinNext[0];
    out[offset++] = joinNext[1];
    out[offset++] = joinNext[2];
    out[offset++] = color[0];
    out[offset++] = color[1];
    out[offset++] = color[2];
    out[offset++] = color[3];
    out[offset++] = side;
    out[offset++] = width;
    out[offset++] = cap;
    return offset;
};
