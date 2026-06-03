import type { RenderPrimitive, RenderSegment, RenderTriangle } from "./types.ts";

const SEGMENT_VERTEX_SHADER = `#version 300 es
layout(location = 0) in vec3 segmentStart;
layout(location = 1) in vec3 segmentEnd;
layout(location = 2) in vec4 color;
layout(location = 3) in float along;
layout(location = 4) in float side;
layout(location = 5) in float width;

uniform mat4 viewProj;
uniform vec2 viewportSize;

out vec4 vColor;

void main() {
    vec4 startClip = viewProj * vec4(segmentStart, 1.0);
    vec4 endClip = viewProj * vec4(segmentEnd, 1.0);
    float startW = abs(startClip.w) > 0.000001 ? startClip.w : 0.000001;
    float endW = abs(endClip.w) > 0.000001 ? endClip.w : 0.000001;
    vec2 startNdc = startClip.xy / startW;
    vec2 endNdc = endClip.xy / endW;
    vec2 directionPx = (endNdc - startNdc) * viewportSize;
    float directionLength = length(directionPx);
    vec2 normal = directionLength > 0.0001
        ? vec2(-directionPx.y, directionPx.x) / directionLength
        : vec2(0.0, 1.0);
    vec2 tangent = directionLength > 0.0001
        ? directionPx / directionLength
        : vec2(1.0, 0.0);
    vec4 clip = mix(startClip, endClip, along);
    vec2 offsetNdc = normal * side * max(width, 1.0) * 0.5 * 2.0 / viewportSize;
    vec2 capNdc = tangent * (along < 0.5 ? -1.0 : 1.0) * max(width, 1.0) * 0.5 * 2.0 / viewportSize;
    clip.xy += offsetNdc * clip.w;
    clip.xy += capNdc * clip.w;
    gl_Position = clip;
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

const FLOATS_PER_VERTEX = 13;
const VERTICES_PER_SEGMENT = 6;
const FLOATS_PER_TRIANGLE_VERTEX = 7;
const VERTICES_PER_TRIANGLE = 3;

export class PaintModelingRenderer {
    private readonly gl: WebGL2RenderingContext;
    private readonly segmentProgram: WebGLProgram;
    private readonly segmentVao: WebGLVertexArrayObject;
    private readonly segmentVertexBuffer: WebGLBuffer;
    private readonly segmentViewProjLocation: WebGLUniformLocation;
    private readonly viewportSizeLocation: WebGLUniformLocation;
    private readonly triangleProgram: WebGLProgram;
    private readonly triangleVao: WebGLVertexArrayObject;
    private readonly triangleVertexBuffer: WebGLBuffer;
    private readonly triangleViewProjLocation: WebGLUniformLocation;
    private segmentCapacityVertices = 0;
    private segmentVertexCount = 0;
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
        const triangleVao = gl.createVertexArray();
        const triangleVertexBuffer = gl.createBuffer();
        if (!segmentVao || !segmentVertexBuffer || !triangleVao || !triangleVertexBuffer) {
            throw new Error("Paint renderer buffers unavailable");
        }
        this.segmentVao = segmentVao;
        this.segmentVertexBuffer = segmentVertexBuffer;
        this.triangleVao = triangleVao;
        this.triangleVertexBuffer = triangleVertexBuffer;

        gl.bindVertexArray(this.segmentVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.segmentVertexBuffer);
        const stride = FLOATS_PER_VERTEX * Float32Array.BYTES_PER_ELEMENT;
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, 0);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 3, gl.FLOAT, false, stride, 3 * Float32Array.BYTES_PER_ELEMENT);
        gl.enableVertexAttribArray(2);
        gl.vertexAttribPointer(2, 4, gl.FLOAT, false, stride, 6 * Float32Array.BYTES_PER_ELEMENT);
        gl.enableVertexAttribArray(3);
        gl.vertexAttribPointer(3, 1, gl.FLOAT, false, stride, 10 * Float32Array.BYTES_PER_ELEMENT);
        gl.enableVertexAttribArray(4);
        gl.vertexAttribPointer(4, 1, gl.FLOAT, false, stride, 11 * Float32Array.BYTES_PER_ELEMENT);
        gl.enableVertexAttribArray(5);
        gl.vertexAttribPointer(5, 1, gl.FLOAT, false, stride, 12 * Float32Array.BYTES_PER_ELEMENT);
        gl.bindVertexArray(null);

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
        const triangles = segments.filter(isRenderTriangle);
        const segmentVertexCount = renderSegments.length * VERTICES_PER_SEGMENT;
        const triangleVertexCount = triangles.length * VERTICES_PER_TRIANGLE;
        const segmentData = new Float32Array(segmentVertexCount * FLOATS_PER_VERTEX);
        const triangleData = new Float32Array(triangleVertexCount * FLOATS_PER_TRIANGLE_VERTEX);
        let segmentOffset = 0;
        let triangleOffset = 0;

        for (const segment of renderSegments) {
            segmentOffset = appendSegment(segmentData, segmentOffset, segment);
        }
        for (const triangle of triangles) {
            triangleOffset = appendTriangle(triangleData, triangleOffset, triangle);
        }

        this.segmentVertexCount = segmentVertexCount;
        this.triangleVertexCount = triangleVertexCount;

        gl.bindVertexArray(this.segmentVao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.segmentVertexBuffer);
        if (segmentVertexCount > this.segmentCapacityVertices) {
            gl.bufferData(gl.ARRAY_BUFFER, segmentData, gl.DYNAMIC_DRAW);
            this.segmentCapacityVertices = segmentVertexCount;
        } else if (segmentVertexCount > 0) {
            gl.bufferSubData(gl.ARRAY_BUFFER, 0, segmentData);
        }
        gl.bindVertexArray(null);

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

    render(viewProjMat: number[] | Float32Array) {
        const gl = this.gl;
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        gl.clearColor(0.035, 0.043, 0.047, 1);
        gl.clearDepth(1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        if (this.triangleVertexCount === 0 && this.segmentVertexCount === 0) return;

        gl.enable(gl.DEPTH_TEST);
        gl.depthFunc(gl.LEQUAL);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

        if (this.triangleVertexCount > 0) {
            gl.useProgram(this.triangleProgram);
            gl.uniformMatrix4fv(this.triangleViewProjLocation, false, viewProjMat);
            gl.depthMask(true);
            gl.bindVertexArray(this.triangleVao);
            gl.drawArrays(gl.TRIANGLES, 0, this.triangleVertexCount);
            gl.bindVertexArray(null);
        }

        if (this.segmentVertexCount === 0) return;

        gl.useProgram(this.segmentProgram);
        gl.uniformMatrix4fv(this.segmentViewProjLocation, false, viewProjMat);
        const canvas = gl.canvas as HTMLCanvasElement;
        gl.uniform2f(
            this.viewportSizeLocation,
            Math.max(1, canvas.clientWidth || canvas.width),
            Math.max(1, canvas.clientHeight || canvas.height),
        );

        gl.bindVertexArray(this.segmentVao);
        gl.depthMask(false);
        gl.drawArrays(gl.TRIANGLES, 0, this.segmentVertexCount);
        gl.depthMask(true);
        gl.bindVertexArray(null);
    }

    destroy() {
        const gl = this.gl;
        gl.deleteBuffer(this.segmentVertexBuffer);
        gl.deleteVertexArray(this.segmentVao);
        gl.deleteProgram(this.segmentProgram);
        gl.deleteBuffer(this.triangleVertexBuffer);
        gl.deleteVertexArray(this.triangleVao);
        gl.deleteProgram(this.triangleProgram);
    }
}

function isRenderSegment(primitive: RenderPrimitive): primitive is RenderSegment {
    return primitive.kind !== "triangle";
}

function isRenderTriangle(primitive: RenderPrimitive): primitive is RenderTriangle {
    return primitive.kind === "triangle";
}

function appendSegment(
    out: Float32Array,
    offset: number,
    segment: RenderSegment,
): number {
    const width = segment.width ?? 1.25;
    offset = appendVertex(out, offset, segment, 0, -1, width);
    offset = appendVertex(out, offset, segment, 1, -1, width);
    offset = appendVertex(out, offset, segment, 1, 1, width);
    offset = appendVertex(out, offset, segment, 0, -1, width);
    offset = appendVertex(out, offset, segment, 1, 1, width);
    offset = appendVertex(out, offset, segment, 0, 1, width);
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

function appendVertex(
    out: Float32Array,
    offset: number,
    segment: RenderSegment,
    along: number,
    side: number,
    width: number,
): number {
    out[offset++] = segment.a[0];
    out[offset++] = segment.a[1];
    out[offset++] = segment.a[2];
    out[offset++] = segment.b[0];
    out[offset++] = segment.b[1];
    out[offset++] = segment.b[2];
    out[offset++] = segment.color[0];
    out[offset++] = segment.color[1];
    out[offset++] = segment.color[2];
    out[offset++] = segment.color[3];
    out[offset++] = along;
    out[offset++] = side;
    out[offset++] = width;
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
