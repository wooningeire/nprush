import type { RenderPrimitive, RenderSegment, RenderStroke, RenderTriangle, Vec3 } from "./types.ts";

const SEGMENT_VERTEX_SHADER = `#version 300 es
layout(location = 0) in vec3 joinPrev;
layout(location = 1) in vec3 joinPoint;
layout(location = 2) in vec3 joinNext;
layout(location = 3) in vec4 color;
layout(location = 4) in float side;
layout(location = 5) in float width;
layout(location = 6) in float cap;

uniform mat4 viewProj;
uniform vec2 viewportSize;

out vec4 vColor;

float safeClipW(vec4 clip) {
    return abs(clip.w) > 0.000001 ? clip.w : 0.000001;
}

vec2 directionPx(vec2 fromNdc, vec2 toNdc, vec2 fallback) {
    vec2 delta = (toNdc - fromNdc) * viewportSize;
    float deltaLength = length(delta);
    return deltaLength > 0.0001 ? delta / deltaLength : fallback;
}

vec2 perpendicular(vec2 direction) {
    return vec2(-direction.y, direction.x);
}

void main() {
    vec4 prevClip = viewProj * vec4(joinPrev, 1.0);
    vec4 pointClip = viewProj * vec4(joinPoint, 1.0);
    vec4 nextClip = viewProj * vec4(joinNext, 1.0);
    float prevW = safeClipW(prevClip);
    float pointW = safeClipW(pointClip);
    float nextW = safeClipW(nextClip);
    vec2 prevNdc = prevClip.xy / prevW;
    vec2 pointNdc = pointClip.xy / pointW;
    vec2 nextNdc = nextClip.xy / nextW;
    vec2 dirIn = directionPx(prevNdc, pointNdc, vec2(0.0, 0.0));
    vec2 dirOut = directionPx(pointNdc, nextNdc, vec2(0.0, 0.0));
    float dirInLength = length(dirIn);
    float dirOutLength = length(dirOut);
    if (dirInLength <= 0.0001 && dirOutLength > 0.0001) {
        dirIn = dirOut;
    } else if (dirOutLength <= 0.0001 && dirInLength > 0.0001) {
        dirOut = dirIn;
    } else if (dirInLength <= 0.0001 && dirOutLength <= 0.0001) {
        dirIn = vec2(1.0, 0.0);
        dirOut = dirIn;
    }
    vec2 tangentSum = dirIn + dirOut;
    float tangentLength = length(tangentSum);
    vec2 tangent = tangentLength > 0.0001 ? tangentSum / tangentLength : dirOut;
    vec2 normalIn = perpendicular(dirIn);
    vec2 miter = perpendicular(tangent);
    float denom = dot(miter, normalIn);
    float miterScale = abs(denom) > 0.15 ? min(abs(1.0 / denom), 2.0) : 1.0;
    float halfWidth = max(width, 1.0) * 0.5;
    vec2 offsetNdc = miter * side * halfWidth * miterScale * 2.0 / viewportSize;
    vec2 capNdc = (cap < 0.0 ? dirOut : dirIn) * cap * halfWidth * 2.0 / viewportSize;
    pointClip.xy += offsetNdc * pointClip.w;
    pointClip.xy += capNdc * pointClip.w;
    gl_Position = pointClip;
    vColor = color;
}
`;

const SEGMENT_FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec4 vColor;
out vec4 outColor;

void main() {
    outColor = vColor;
}
`;

const TRIANGLE_VERTEX_SHADER = `#version 300 es
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;

uniform mat4 viewProj;

out vec4 vColor;

void main() {
    gl_Position = viewProj * vec4(position, 1.0);
    vColor = color;
}
`;

const TRIANGLE_FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec4 vColor;
out vec4 outColor;

void main() {
    outColor = vColor;
}
`;

const GRID_VERTEX_SHADER = `#version 300 es
precision highp float;

const vec2 POSITIONS[3] = vec2[3](
    vec2(-1.0, -1.0),
    vec2(3.0, -1.0),
    vec2(-1.0, 3.0)
);

out vec2 vNdc;

void main() {
    vNdc = POSITIONS[gl_VertexID];
    gl_Position = vec4(vNdc, 0.0, 1.0);
}
`;

const GRID_FRAGMENT_SHADER = `#version 300 es
precision highp float;

uniform mat4 viewProjInv;
uniform float planeZ;

in vec2 vNdc;
out vec4 outColor;

float gridLine(vec2 position, float spacing) {
    vec2 coord = position / spacing;
    vec2 derivative = max(fwidth(coord), vec2(0.00001));
    vec2 grid = abs(fract(coord + 0.5) - 0.5) / derivative;
    return 1.0 - clamp(min(grid.x, grid.y), 0.0, 1.0);
}

float axisLine(float distance) {
    float derivative = max(fwidth(distance), 0.00001);
    return 1.0 - smoothstep(0.0, derivative * 1.35, abs(distance));
}

void main() {
    vec4 nearH = viewProjInv * vec4(vNdc, 0.02, 1.0);
    vec4 farH = viewProjInv * vec4(vNdc, 0.98, 1.0);
    vec3 nearWorld = nearH.xyz / nearH.w;
    vec3 farWorld = farH.xyz / farH.w;
    vec3 ray = farWorld - nearWorld;

    if (abs(ray.z) < 0.000001) discard;

    float t = (planeZ - nearWorld.z) / ray.z;
    if (t <= 0.0) discard;

    vec3 world = nearWorld + ray * t;
    vec3 rayDirection = normalize(ray);
    float rayDistance = length(world - nearWorld);
    float horizonFade = smoothstep(0.015, 0.11, abs(rayDirection.z));
    float distanceFade = 1.0 - smoothstep(24.0, 96.0, rayDistance);
    float fade = horizonFade * distanceFade;
    if (fade <= 0.001) discard;

    float minor = gridLine(world.xy, 0.25);
    float major = gridLine(world.xy, 1.0);
    float xAxis = axisLine(world.y);
    float yAxis = axisLine(world.x);

    vec3 color = vec3(0.34, 0.40, 0.40);
    float alpha = minor * 0.14;
    color = mix(color, vec3(0.46, 0.54, 0.54), major);
    alpha = max(alpha, major * 0.26);

    if (xAxis > alpha) {
        color = mix(color, vec3(0.92, 0.42, 0.38), xAxis);
        alpha = max(alpha, xAxis * 0.72);
    }
    if (yAxis > alpha) {
        color = mix(color, vec3(0.48, 0.82, 0.55), yAxis);
        alpha = max(alpha, yAxis * 0.72);
    }

    outColor = vec4(color, alpha * fade);
}
`;

const GRID_PLANE_Z = -0.02;
const FLOATS_PER_VERTEX = 16;
const VERTICES_PER_SEGMENT = 6;
const FLOATS_PER_TRIANGLE_VERTEX = 7;
const VERTICES_PER_TRIANGLE = 3;

export class PaintModelingRenderer {
    private readonly gl: WebGL2RenderingContext;
    private readonly gridProgram: WebGLProgram;
    private readonly gridVao: WebGLVertexArrayObject;
    private readonly gridViewProjInvLocation: WebGLUniformLocation;
    private readonly gridPlaneZLocation: WebGLUniformLocation;
    private readonly segmentProgram: WebGLProgram;
    private readonly segmentVao: WebGLVertexArrayObject;
    private readonly segmentVertexBuffer: WebGLBuffer;
    private readonly strokeVao: WebGLVertexArrayObject;
    private readonly strokeVertexBuffer: WebGLBuffer;
    private readonly draftSegmentVao: WebGLVertexArrayObject;
    private readonly draftSegmentVertexBuffer: WebGLBuffer;
    private readonly draftStrokeVao: WebGLVertexArrayObject;
    private readonly draftStrokeVertexBuffer: WebGLBuffer;
    private readonly segmentViewProjLocation: WebGLUniformLocation;
    private readonly viewportSizeLocation: WebGLUniformLocation;
    private readonly triangleProgram: WebGLProgram;
    private readonly triangleVao: WebGLVertexArrayObject;
    private readonly triangleVertexBuffer: WebGLBuffer;
    private readonly triangleViewProjLocation: WebGLUniformLocation;
    private segmentCapacityVertices = 0;
    private segmentVertexCount = 0;
    private strokeCapacityVertices = 0;
    private strokeVertexCount = 0;
    private draftSegmentCapacityVertices = 0;
    private draftSegmentVertexCount = 0;
    private draftStrokeCapacityVertices = 0;
    private draftStrokeVertexCount = 0;
    private triangleCapacityVertices = 0;
    private triangleVertexCount = 0;

    static create(canvas: HTMLCanvasElement): PaintModelingRenderer {
        const gl = canvas.getContext("webgl2", {
            antialias: true,
            depth: true,
            alpha: true,
        });
        if (!gl) throw new Error("WebGL2 is unavailable");
        return new PaintModelingRenderer(gl);
    }

    private constructor(gl: WebGL2RenderingContext) {
        this.gl = gl;
        this.gridProgram = createProgram(gl, GRID_VERTEX_SHADER, GRID_FRAGMENT_SHADER);
        const gridViewProjInvLocation = gl.getUniformLocation(this.gridProgram, "viewProjInv");
        if (!gridViewProjInvLocation) throw new Error("Paint renderer grid inverse uniform missing");
        this.gridViewProjInvLocation = gridViewProjInvLocation;
        const gridPlaneZLocation = gl.getUniformLocation(this.gridProgram, "planeZ");
        if (!gridPlaneZLocation) throw new Error("Paint renderer grid plane uniform missing");
        this.gridPlaneZLocation = gridPlaneZLocation;

        this.segmentProgram = createProgram(gl, SEGMENT_VERTEX_SHADER, SEGMENT_FRAGMENT_SHADER);
        const segmentViewProjLocation = gl.getUniformLocation(this.segmentProgram, "viewProj");
        if (!segmentViewProjLocation) throw new Error("Paint renderer segment uniform missing");
        this.segmentViewProjLocation = segmentViewProjLocation;
        const viewportSizeLocation = gl.getUniformLocation(this.segmentProgram, "viewportSize");
        if (!viewportSizeLocation) throw new Error("Paint renderer viewport uniform missing");
        this.viewportSizeLocation = viewportSizeLocation;

        this.triangleProgram = createProgram(gl, TRIANGLE_VERTEX_SHADER, TRIANGLE_FRAGMENT_SHADER);
        const triangleViewProjLocation = gl.getUniformLocation(this.triangleProgram, "viewProj");
        if (!triangleViewProjLocation) throw new Error("Paint renderer triangle uniform missing");
        this.triangleViewProjLocation = triangleViewProjLocation;

        const segmentVao = gl.createVertexArray();
        const segmentVertexBuffer = gl.createBuffer();
        const strokeVao = gl.createVertexArray();
        const strokeVertexBuffer = gl.createBuffer();
        const draftSegmentVao = gl.createVertexArray();
        const draftSegmentVertexBuffer = gl.createBuffer();
        const draftStrokeVao = gl.createVertexArray();
        const draftStrokeVertexBuffer = gl.createBuffer();
        const triangleVao = gl.createVertexArray();
        const triangleVertexBuffer = gl.createBuffer();
        if (
            !segmentVao
            || !segmentVertexBuffer
            || !strokeVao
            || !strokeVertexBuffer
            || !draftSegmentVao
            || !draftSegmentVertexBuffer
            || !draftStrokeVao
            || !draftStrokeVertexBuffer
            || !triangleVao
            || !triangleVertexBuffer
        ) {
            throw new Error("Paint renderer buffers unavailable");
        }
        const gridVao = gl.createVertexArray();
        if (!gridVao) throw new Error("Paint renderer grid buffer unavailable");
        this.gridVao = gridVao;
        this.segmentVao = segmentVao;
        this.segmentVertexBuffer = segmentVertexBuffer;
        this.strokeVao = strokeVao;
        this.strokeVertexBuffer = strokeVertexBuffer;
        this.draftSegmentVao = draftSegmentVao;
        this.draftSegmentVertexBuffer = draftSegmentVertexBuffer;
        this.draftStrokeVao = draftStrokeVao;
        this.draftStrokeVertexBuffer = draftStrokeVertexBuffer;
        this.triangleVao = triangleVao;
        this.triangleVertexBuffer = triangleVertexBuffer;

        configureSegmentVertexArray(gl, this.segmentVao, this.segmentVertexBuffer);
        configureSegmentVertexArray(gl, this.strokeVao, this.strokeVertexBuffer);
        configureSegmentVertexArray(gl, this.draftSegmentVao, this.draftSegmentVertexBuffer);
        configureSegmentVertexArray(gl, this.draftStrokeVao, this.draftStrokeVertexBuffer);

        gl.bindVertexArray(this.triangleVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleVertexBuffer);
        const triangleStride = FLOATS_PER_TRIANGLE_VERTEX * Float32Array.BYTES_PER_ELEMENT;
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, triangleStride, 0);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 4, gl.FLOAT, false, triangleStride, 3 * Float32Array.BYTES_PER_ELEMENT);
        gl.bindVertexArray(null);
    }

    setSegments(segments: RenderPrimitive[]) {
        const gl = this.gl;
        const renderSegments = segments.filter(isRenderSegment);
        const strokes = segments.filter(isRenderStroke);
        const triangles = segments.filter(isRenderTriangle);
        const segmentVertexCount = renderSegments.length * VERTICES_PER_SEGMENT;
        const strokeVertexCount = strokeStripVertexCount(strokes);
        const triangleVertexCount = triangles.length * VERTICES_PER_TRIANGLE;
        const segmentData = new Float32Array(segmentVertexCount * FLOATS_PER_VERTEX);
        const strokeData = new Float32Array(strokeVertexCount * FLOATS_PER_VERTEX);
        const triangleData = new Float32Array(triangleVertexCount * FLOATS_PER_TRIANGLE_VERTEX);
        let segmentOffset = 0;
        let triangleOffset = 0;

        for (const segment of renderSegments) {
            segmentOffset = appendSegment(segmentData, segmentOffset, segment);
        }
        appendStrokeStrips(strokeData, 0, strokes);
        for (const triangle of triangles) {
            triangleOffset = appendTriangle(triangleData, triangleOffset, triangle);
        }

        this.segmentVertexCount = segmentVertexCount;
        this.strokeVertexCount = strokeVertexCount;
        this.triangleVertexCount = triangleVertexCount;

        this.segmentCapacityVertices = uploadVertexData(
            gl,
            this.segmentVao,
            this.segmentVertexBuffer,
            segmentData,
            segmentVertexCount,
            this.segmentCapacityVertices,
        );
        this.strokeCapacityVertices = uploadVertexData(
            gl,
            this.strokeVao,
            this.strokeVertexBuffer,
            strokeData,
            strokeVertexCount,
            this.strokeCapacityVertices,
        );

        gl.bindVertexArray(this.triangleVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.triangleVertexBuffer);
        if (triangleVertexCount > this.triangleCapacityVertices) {
            gl.bufferData(gl.ARRAY_BUFFER, triangleData, gl.DYNAMIC_DRAW);
            this.triangleCapacityVertices = triangleVertexCount;
        } else if (triangleVertexCount > 0) {
            gl.bufferSubData(gl.ARRAY_BUFFER, 0, triangleData);
        }
        gl.bindVertexArray(null);
    }

    setDraftSegments(segments: RenderPrimitive[]) {
        const gl = this.gl;
        const renderSegments = segments.filter(isRenderSegment);
        const strokes = segments.filter(isRenderStroke);
        const segmentVertexCount = renderSegments.length * VERTICES_PER_SEGMENT;
        const strokeVertexCount = strokeStripVertexCount(strokes);
        const segmentData = new Float32Array(segmentVertexCount * FLOATS_PER_VERTEX);
        const strokeData = new Float32Array(strokeVertexCount * FLOATS_PER_VERTEX);
        let segmentOffset = 0;

        for (const segment of renderSegments) {
            segmentOffset = appendSegment(segmentData, segmentOffset, segment);
        }
        appendStrokeStrips(strokeData, 0, strokes);

        this.draftSegmentVertexCount = segmentVertexCount;
        this.draftStrokeVertexCount = strokeVertexCount;
        this.draftSegmentCapacityVertices = uploadVertexData(
            gl,
            this.draftSegmentVao,
            this.draftSegmentVertexBuffer,
            segmentData,
            segmentVertexCount,
            this.draftSegmentCapacityVertices,
        );
        this.draftStrokeCapacityVertices = uploadVertexData(
            gl,
            this.draftStrokeVao,
            this.draftStrokeVertexBuffer,
            strokeData,
            strokeVertexCount,
            this.draftStrokeCapacityVertices,
        );
    }

    render(viewProjMat: number[] | Float32Array, viewProjInvMat: number[] | Float32Array) {
        const gl = this.gl;
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        gl.clearColor(0.035, 0.043, 0.047, 1);
        gl.clearDepth(1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        gl.enable(gl.DEPTH_TEST);
        gl.depthFunc(gl.LEQUAL);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        gl.useProgram(this.gridProgram);
        gl.uniformMatrix4fv(this.gridViewProjInvLocation, false, viewProjInvMat);
        gl.uniform1f(this.gridPlaneZLocation, GRID_PLANE_Z);
        gl.depthMask(false);
        gl.disable(gl.DEPTH_TEST);
        gl.bindVertexArray(this.gridVao);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.bindVertexArray(null);
        gl.enable(gl.DEPTH_TEST);
        gl.depthMask(true);

        if (this.triangleVertexCount > 0) {
            gl.useProgram(this.triangleProgram);
            gl.uniformMatrix4fv(this.triangleViewProjLocation, false, viewProjMat);
            gl.depthMask(true);
            gl.bindVertexArray(this.triangleVao);
            gl.drawArrays(gl.TRIANGLES, 0, this.triangleVertexCount);
            gl.bindVertexArray(null);
        }

        if (
            this.segmentVertexCount === 0
            && this.strokeVertexCount === 0
            && this.draftSegmentVertexCount === 0
            && this.draftStrokeVertexCount === 0
        ) return;

        gl.useProgram(this.segmentProgram);
        gl.uniformMatrix4fv(this.segmentViewProjLocation, false, viewProjMat);
        const canvas = gl.canvas as HTMLCanvasElement;
        gl.uniform2f(
            this.viewportSizeLocation,
            Math.max(1, canvas.clientWidth || canvas.width),
            Math.max(1, canvas.clientHeight || canvas.height),
        );

        gl.depthMask(false);
        drawSegmentBuffer(gl, this.segmentVao, this.segmentVertexCount);
        drawStrokeBuffer(gl, this.strokeVao, this.strokeVertexCount);
        drawSegmentBuffer(gl, this.draftSegmentVao, this.draftSegmentVertexCount);
        drawStrokeBuffer(gl, this.draftStrokeVao, this.draftStrokeVertexCount);
        gl.depthMask(true);
    }

    destroy() {
        const gl = this.gl;
        gl.deleteVertexArray(this.gridVao);
        gl.deleteProgram(this.gridProgram);
        gl.deleteBuffer(this.segmentVertexBuffer);
        gl.deleteVertexArray(this.segmentVao);
        gl.deleteBuffer(this.strokeVertexBuffer);
        gl.deleteVertexArray(this.strokeVao);
        gl.deleteProgram(this.segmentProgram);
        gl.deleteBuffer(this.draftSegmentVertexBuffer);
        gl.deleteVertexArray(this.draftSegmentVao);
        gl.deleteBuffer(this.draftStrokeVertexBuffer);
        gl.deleteVertexArray(this.draftStrokeVao);
        gl.deleteBuffer(this.triangleVertexBuffer);
        gl.deleteVertexArray(this.triangleVao);
        gl.deleteProgram(this.triangleProgram);
    }
}

function configureSegmentVertexArray(
    gl: WebGL2RenderingContext,
    vao: WebGLVertexArrayObject,
    buffer: WebGLBuffer,
) {
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    const stride = FLOATS_PER_VERTEX * Float32Array.BYTES_PER_ELEMENT;
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, stride, 3 * Float32Array.BYTES_PER_ELEMENT);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 3, gl.FLOAT, false, stride, 6 * Float32Array.BYTES_PER_ELEMENT);
    gl.enableVertexAttribArray(3);
    gl.vertexAttribPointer(3, 4, gl.FLOAT, false, stride, 9 * Float32Array.BYTES_PER_ELEMENT);
    gl.enableVertexAttribArray(4);
    gl.vertexAttribPointer(4, 1, gl.FLOAT, false, stride, 13 * Float32Array.BYTES_PER_ELEMENT);
    gl.enableVertexAttribArray(5);
    gl.vertexAttribPointer(5, 1, gl.FLOAT, false, stride, 14 * Float32Array.BYTES_PER_ELEMENT);
    gl.enableVertexAttribArray(6);
    gl.vertexAttribPointer(6, 1, gl.FLOAT, false, stride, 15 * Float32Array.BYTES_PER_ELEMENT);
    gl.bindVertexArray(null);
}

function uploadVertexData(
    gl: WebGL2RenderingContext,
    vao: WebGLVertexArrayObject,
    buffer: WebGLBuffer,
    data: Float32Array,
    vertexCount: number,
    capacityVertices: number,
): number {
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    if (vertexCount > capacityVertices) {
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
        capacityVertices = vertexCount;
    } else if (vertexCount > 0) {
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
    }
    gl.bindVertexArray(null);
    return capacityVertices;
}

function drawSegmentBuffer(gl: WebGL2RenderingContext, vao: WebGLVertexArrayObject, vertexCount: number) {
    if (vertexCount === 0) return;
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.TRIANGLES, 0, vertexCount);
    gl.bindVertexArray(null);
}

function drawStrokeBuffer(gl: WebGL2RenderingContext, vao: WebGLVertexArrayObject, vertexCount: number) {
    if (vertexCount === 0) return;
    gl.bindVertexArray(vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, vertexCount);
    gl.bindVertexArray(null);
}

function isRenderSegment(primitive: RenderPrimitive): primitive is RenderSegment {
    return primitive.kind !== "triangle" && primitive.kind !== "stroke";
}

function isRenderTriangle(primitive: RenderPrimitive): primitive is RenderTriangle {
    return primitive.kind === "triangle";
}

function isRenderStroke(primitive: RenderPrimitive): primitive is RenderStroke {
    return primitive.kind === "stroke";
}

function appendSegment(
    out: Float32Array,
    offset: number,
    segment: RenderSegment,
): number {
    const width = segment.width ?? 1.25;
    offset = appendSegmentVertex(out, offset, segment, 0, -1, width);
    offset = appendSegmentVertex(out, offset, segment, 1, -1, width);
    offset = appendSegmentVertex(out, offset, segment, 1, 1, width);
    offset = appendSegmentVertex(out, offset, segment, 0, -1, width);
    offset = appendSegmentVertex(out, offset, segment, 1, 1, width);
    offset = appendSegmentVertex(out, offset, segment, 0, 1, width);
    return offset;
}

function strokeStripVertexCount(strokes: RenderStroke[]): number {
    let count = 0;
    let hasPreviousRun = false;
    for (const stroke of strokes) {
        if (stroke.points.length < 2) continue;
        if (hasPreviousRun) count += 2;
        count += stroke.points.length * 2;
        hasPreviousRun = true;
    }
    return count;
}

function appendStrokeStrips(
    out: Float32Array,
    offset: number,
    strokes: RenderStroke[],
): number {
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
}

function appendTriangle(
    out: Float32Array,
    offset: number,
    triangle: RenderTriangle,
): number {
    offset = appendTriangleVertex(out, offset, triangle.a, triangle.color);
    offset = appendTriangleVertex(out, offset, triangle.b, triangle.color);
    offset = appendTriangleVertex(out, offset, triangle.c, triangle.color);
    return offset;
}

function appendTriangleVertex(
    out: Float32Array,
    offset: number,
    position: [number, number, number],
    color: [number, number, number, number],
): number {
    out[offset++] = position[0];
    out[offset++] = position[1];
    out[offset++] = position[2];
    out[offset++] = color[0];
    out[offset++] = color[1];
    out[offset++] = color[2];
    out[offset++] = color[3];
    return offset;
}

function appendSegmentVertex(
    out: Float32Array,
    offset: number,
    segment: RenderSegment,
    along: number,
    side: number,
    width: number,
): number {
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
}

function appendStrokeVertex(
    out: Float32Array,
    offset: number,
    stroke: RenderStroke,
    index: number,
    side: number,
): number {
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
}

function appendJoinVertex(
    out: Float32Array,
    offset: number,
    joinPrev: Vec3,
    joinPoint: Vec3,
    joinNext: Vec3,
    color: [number, number, number, number],
    side: number,
    width: number,
    cap: number,
): number {
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
}

function createProgram(gl: WebGL2RenderingContext, vertexSource: string, fragmentSource: string): WebGLProgram {
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
    const program = gl.createProgram();
    if (!program) throw new Error("Unable to create paint renderer program");
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const log = gl.getProgramInfoLog(program) ?? "unknown link error";
        gl.deleteProgram(program);
        throw new Error(`Paint renderer link failed: ${log}`);
    }

    return program;
}

function createShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
    const shader = gl.createShader(type);
    if (!shader) throw new Error("Unable to create paint renderer shader");
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(shader) ?? "unknown compile error";
        gl.deleteShader(shader);
        throw new Error(`Paint renderer shader failed: ${log}`);
    }
    return shader;
}
