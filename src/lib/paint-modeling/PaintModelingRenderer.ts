import { requestGpu } from "$/gpu/setup/requestGpu";
import type { RenderPrimitive, RenderSegment, RenderStroke, RenderTriangle, Vec3 } from "./types.ts";

const PAINT_MODELING_SHADER = /* wgsl */`
struct GridUniforms {
    view_proj_inv: mat4x4f,
    plane_z: f32,
};

struct SegmentUniforms {
    view_proj: mat4x4f,
    viewport_size: vec2f,
};

struct TriangleUniforms {
    view_proj: mat4x4f,
};

struct GridVertexOut {
    @builtin(position) position: vec4f,
    @location(0) ndc: vec2f,
};

struct SegmentVertexOut {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
};

struct TriangleVertexOut {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
};

@group(0) @binding(0) var<uniform> grid_uniforms: GridUniforms;
@group(0) @binding(0) var<uniform> segment_uniforms: SegmentUniforms;
@group(0) @binding(0) var<uniform> triangle_uniforms: TriangleUniforms;

fn fwidth_vec2(value: vec2f) -> vec2f {
    return abs(dpdx(value)) + abs(dpdy(value));
}

fn fwidth_f32(value: f32) -> f32 {
    return abs(dpdx(value)) + abs(dpdy(value));
}

fn grid_line(position: vec2f, spacing: f32) -> f32 {
    let coord = position / vec2f(spacing);
    let derivative = max(fwidth_vec2(coord), vec2f(0.00001));
    let grid = abs(fract(coord + vec2f(0.5)) - vec2f(0.5)) / derivative;
    return 1.0 - clamp(min(grid.x, grid.y), 0.0, 1.0);
}

fn axis_line(distance: f32) -> f32 {
    let derivative = max(fwidth_f32(distance), 0.00001);
    return 1.0 - smoothstep(0.0, derivative * 1.35, abs(distance));
}

@vertex
fn grid_vertex(@builtin(vertex_index) vertex_index: u32) -> GridVertexOut {
    let positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0),
    );
    let ndc = positions[vertex_index];
    var out: GridVertexOut;
    out.ndc = ndc;
    out.position = vec4f(ndc, 0.0, 1.0);
    return out;
}

@fragment
fn grid_fragment(in: GridVertexOut) -> @location(0) vec4f {
    let near_h = grid_uniforms.view_proj_inv * vec4f(in.ndc, 0.02, 1.0);
    let far_h = grid_uniforms.view_proj_inv * vec4f(in.ndc, 0.98, 1.0);
    let near_world = near_h.xyz / near_h.w;
    let far_world = far_h.xyz / far_h.w;
    let ray = far_world - near_world;

    if (abs(ray.z) < 0.000001) {
        discard;
    }

    let t = (grid_uniforms.plane_z - near_world.z) / ray.z;
    if (t <= 0.0) {
        discard;
    }

    let world = near_world + ray * t;
    let ray_direction = normalize(ray);
    let ray_distance = length(world - near_world);
    let horizon_fade = smoothstep(0.015, 0.11, abs(ray_direction.z));
    let distance_fade = 1.0 - smoothstep(24.0, 96.0, ray_distance);
    let fade = horizon_fade * distance_fade;
    if (fade <= 0.001) {
        discard;
    }

    let minor = grid_line(world.xy, 0.25);
    let major = grid_line(world.xy, 1.0);
    let x_axis = axis_line(world.y);
    let y_axis = axis_line(world.x);

    var color = vec3f(0.34, 0.40, 0.40);
    var alpha = minor * 0.14;
    color = mix(color, vec3f(0.46, 0.54, 0.54), major);
    alpha = max(alpha, major * 0.26);

    if (x_axis > alpha) {
        color = mix(color, vec3f(0.92, 0.42, 0.38), x_axis);
        alpha = max(alpha, x_axis * 0.72);
    }
    if (y_axis > alpha) {
        color = mix(color, vec3f(0.48, 0.82, 0.55), y_axis);
        alpha = max(alpha, y_axis * 0.72);
    }

    return vec4f(color, alpha * fade);
}

fn safe_clip_w(clip: vec4f) -> f32 {
    return select(0.000001, clip.w, abs(clip.w) > 0.000001);
}

fn direction_px(from_ndc: vec2f, to_ndc: vec2f, fallback: vec2f) -> vec2f {
    let delta = (to_ndc - from_ndc) * segment_uniforms.viewport_size;
    let delta_length = length(delta);
    return select(fallback, delta / delta_length, delta_length > 0.0001);
}

fn perpendicular(direction: vec2f) -> vec2f {
    return vec2f(-direction.y, direction.x);
}

@vertex
fn segment_vertex(
    @location(0) join_prev: vec3f,
    @location(1) join_point: vec3f,
    @location(2) join_next: vec3f,
    @location(3) color: vec4f,
    @location(4) side: f32,
    @location(5) width: f32,
    @location(6) cap: f32,
) -> SegmentVertexOut {
    let prev_clip = segment_uniforms.view_proj * vec4f(join_prev, 1.0);
    var point_clip = segment_uniforms.view_proj * vec4f(join_point, 1.0);
    let next_clip = segment_uniforms.view_proj * vec4f(join_next, 1.0);
    let prev_w = safe_clip_w(prev_clip);
    let point_w = safe_clip_w(point_clip);
    let next_w = safe_clip_w(next_clip);
    let prev_ndc = prev_clip.xy / prev_w;
    let point_ndc = point_clip.xy / point_w;
    let next_ndc = next_clip.xy / next_w;
    var dir_in = direction_px(prev_ndc, point_ndc, vec2f(0.0));
    var dir_out = direction_px(point_ndc, next_ndc, vec2f(0.0));
    let dir_in_length = length(dir_in);
    let dir_out_length = length(dir_out);

    if (dir_in_length <= 0.0001 && dir_out_length > 0.0001) {
        dir_in = dir_out;
    } else if (dir_out_length <= 0.0001 && dir_in_length > 0.0001) {
        dir_out = dir_in;
    } else if (dir_in_length <= 0.0001 && dir_out_length <= 0.0001) {
        dir_in = vec2f(1.0, 0.0);
        dir_out = dir_in;
    }

    let tangent_sum = dir_in + dir_out;
    let tangent_length = length(tangent_sum);
    let tangent = select(dir_out, tangent_sum / tangent_length, tangent_length > 0.0001);
    let normal_in = perpendicular(dir_in);
    let miter = perpendicular(tangent);
    let denom = dot(miter, normal_in);
    let miter_scale = select(1.0, min(abs(1.0 / denom), 2.0), abs(denom) > 0.15);
    let half_width = max(width, 1.0) * 0.5;
    let offset_ndc = miter * side * half_width * miter_scale * 2.0 / segment_uniforms.viewport_size;
    let cap_direction = select(dir_in, dir_out, cap < 0.0);
    let cap_ndc = cap_direction * cap * half_width * 2.0 / segment_uniforms.viewport_size;
    point_clip.x += offset_ndc.x * point_clip.w + cap_ndc.x * point_clip.w;
    point_clip.y += offset_ndc.y * point_clip.w + cap_ndc.y * point_clip.w;

    var out: SegmentVertexOut;
    out.position = point_clip;
    out.color = color;
    return out;
}

@fragment
fn segment_fragment(in: SegmentVertexOut) -> @location(0) vec4f {
    return in.color;
}

@vertex
fn triangle_vertex(
    @location(0) position: vec3f,
    @location(1) color: vec4f,
) -> TriangleVertexOut {
    var out: TriangleVertexOut;
    out.position = triangle_uniforms.view_proj * vec4f(position, 1.0);
    out.color = color;
    return out;
}

@fragment
fn triangle_fragment(in: TriangleVertexOut) -> @location(0) vec4f {
    return in.color;
}
`;

const DEPTH_FORMAT: GPUTextureFormat = "depth24plus";
const GRID_PLANE_Z = -0.02;
const FLOATS_PER_VERTEX = 16;
const VERTICES_PER_SEGMENT = 6;
const FLOATS_PER_TRIANGLE_VERTEX = 7;
const VERTICES_PER_TRIANGLE = 3;
const MATRIX_FLOATS = 16;
const GRID_UNIFORM_FLOATS = 20;
const SEGMENT_UNIFORM_FLOATS = 20;

type VertexBufferState = {
    buffer: GPUBuffer | null,
    capacityVertices: number,
};

export class PaintModelingRenderer {
    private readonly device: GPUDevice;
    private readonly context: GPUCanvasContext;
    private readonly gridPipeline: GPURenderPipeline;
    private readonly segmentPipeline: GPURenderPipeline;
    private readonly strokePipeline: GPURenderPipeline;
    private readonly trianglePipeline: GPURenderPipeline;
    private readonly gridUniformBuffer: GPUBuffer;
    private readonly segmentUniformBuffer: GPUBuffer;
    private readonly triangleUniformBuffer: GPUBuffer;
    private readonly gridBindGroup: GPUBindGroup;
    private readonly segmentBindGroup: GPUBindGroup;
    private readonly triangleBindGroup: GPUBindGroup;
    private readonly segmentBuffer: VertexBufferState = createVertexBufferState();
    private readonly strokeBuffer: VertexBufferState = createVertexBufferState();
    private readonly draftSegmentBuffer: VertexBufferState = createVertexBufferState();
    private readonly draftStrokeBuffer: VertexBufferState = createVertexBufferState();
    private readonly triangleBuffer: VertexBufferState = createVertexBufferState();
    private depthTexture: GPUTexture | null = null;
    private depthWidth = 0;
    private depthHeight = 0;
    private segmentVertexCount = 0;
    private strokeVertexCount = 0;
    private draftSegmentVertexCount = 0;
    private draftStrokeVertexCount = 0;
    private triangleVertexCount = 0;

    static async create(canvas: HTMLCanvasElement): Promise<PaintModelingRenderer> {
        const gpu = await requestGpu({});
        if (!gpu) throw new Error("WebGPU unavailable");

        const context = canvas.getContext("webgpu");
        if (!context) throw new Error("Could not create WebGPU canvas context");

        context.configure({
            device: gpu.device,
            format: gpu.format,
            alphaMode: "opaque",
        });

        return new PaintModelingRenderer(gpu.device, context, gpu.format);
    }

    private constructor(
        device: GPUDevice,
        context: GPUCanvasContext,
        format: GPUTextureFormat,
    ) {
        this.device = device;
        this.context = context;

        const shaderModule = device.createShaderModule({
            label: "paint modeler renderer shader",
            code: PAINT_MODELING_SHADER,
        });
        void shaderModule.getCompilationInfo().then(info => {
            for (const message of info.messages) {
                console.warn(`[paint_modeler] ${message.type}: ${message.message} (line ${message.lineNum})`);
            }
        });

        const uniformBindGroupLayout = device.createBindGroupLayout({
            label: "paint modeler uniform bind group layout",
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: { type: "uniform" },
            }],
        });
        const pipelineLayout = device.createPipelineLayout({
            label: "paint modeler pipeline layout",
            bindGroupLayouts: [uniformBindGroupLayout],
        });

        this.gridPipeline = createGridPipeline(device, pipelineLayout, shaderModule, format);
        this.segmentPipeline = createSegmentPipeline(device, pipelineLayout, shaderModule, format, "triangle-list");
        this.strokePipeline = createSegmentPipeline(device, pipelineLayout, shaderModule, format, "triangle-strip");
        this.trianglePipeline = createTrianglePipeline(device, pipelineLayout, shaderModule, format);

        this.gridUniformBuffer = createUniformBuffer(device, GRID_UNIFORM_FLOATS, "paint modeler grid uniforms");
        this.segmentUniformBuffer = createUniformBuffer(device, SEGMENT_UNIFORM_FLOATS, "paint modeler segment uniforms");
        this.triangleUniformBuffer = createUniformBuffer(device, MATRIX_FLOATS, "paint modeler triangle uniforms");
        this.gridBindGroup = createUniformBindGroup(
            device,
            uniformBindGroupLayout,
            this.gridUniformBuffer,
            "paint modeler grid bind group",
        );
        this.segmentBindGroup = createUniformBindGroup(
            device,
            uniformBindGroupLayout,
            this.segmentUniformBuffer,
            "paint modeler segment bind group",
        );
        this.triangleBindGroup = createUniformBindGroup(
            device,
            uniformBindGroupLayout,
            this.triangleUniformBuffer,
            "paint modeler triangle bind group",
        );
    }

    setSegments(segments: RenderPrimitive[]) {
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

        uploadVertexData(this.device, this.segmentBuffer, segmentData, segmentVertexCount, "paint modeler segments");
        uploadVertexData(this.device, this.strokeBuffer, strokeData, strokeVertexCount, "paint modeler strokes");
        uploadVertexData(this.device, this.triangleBuffer, triangleData, triangleVertexCount, "paint modeler triangles");
    }

    setDraftSegments(segments: RenderPrimitive[]) {
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
        uploadVertexData(
            this.device,
            this.draftSegmentBuffer,
            segmentData,
            segmentVertexCount,
            "paint modeler draft segments",
        );
        uploadVertexData(
            this.device,
            this.draftStrokeBuffer,
            strokeData,
            strokeVertexCount,
            "paint modeler draft strokes",
        );
    }

    render(viewProjMat: number[] | Float32Array, viewProjInvMat: number[] | Float32Array) {
        const width = Math.floor(this.context.canvas.width);
        const height = Math.floor(this.context.canvas.height);
        if (width <= 0 || height <= 0) return;

        this.ensureDepthTexture(width, height);
        this.writeUniforms(viewProjMat, viewProjInvMat);

        const encoder = this.device.createCommandEncoder({ label: "paint modeler render encoder" });
        const pass = encoder.beginRenderPass({
            label: "paint modeler render pass",
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0.035, g: 0.043, b: 0.047, a: 1 },
                loadOp: "clear",
                storeOp: "store",
            }],
            depthStencilAttachment: {
                view: this.depthTexture!.createView(),
                depthClearValue: 1,
                depthLoadOp: "clear",
                depthStoreOp: "store",
            },
        });

        pass.setPipeline(this.gridPipeline);
        pass.setBindGroup(0, this.gridBindGroup);
        pass.draw(3);

        if (this.triangleVertexCount > 0 && this.triangleBuffer.buffer) {
            pass.setPipeline(this.trianglePipeline);
            pass.setBindGroup(0, this.triangleBindGroup);
            pass.setVertexBuffer(0, this.triangleBuffer.buffer);
            pass.draw(this.triangleVertexCount);
        }

        this.drawSegmentBuffer(pass, this.segmentBuffer, this.segmentVertexCount, this.segmentPipeline);
        this.drawSegmentBuffer(pass, this.strokeBuffer, this.strokeVertexCount, this.strokePipeline);
        this.drawSegmentBuffer(pass, this.draftSegmentBuffer, this.draftSegmentVertexCount, this.segmentPipeline);
        this.drawSegmentBuffer(pass, this.draftStrokeBuffer, this.draftStrokeVertexCount, this.strokePipeline);

        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }

    destroy() {
        destroyVertexBuffer(this.segmentBuffer);
        destroyVertexBuffer(this.strokeBuffer);
        destroyVertexBuffer(this.draftSegmentBuffer);
        destroyVertexBuffer(this.draftStrokeBuffer);
        destroyVertexBuffer(this.triangleBuffer);
        this.depthTexture?.destroy();
        this.gridUniformBuffer.destroy();
        this.segmentUniformBuffer.destroy();
        this.triangleUniformBuffer.destroy();
        this.device.destroy();
    }

    private writeUniforms(viewProjMat: number[] | Float32Array, viewProjInvMat: number[] | Float32Array) {
        const canvas = this.context.canvas as HTMLCanvasElement;
        const gridUniformData = new Float32Array(GRID_UNIFORM_FLOATS);
        gridUniformData.set(viewProjInvMat, 0);
        gridUniformData[MATRIX_FLOATS] = GRID_PLANE_Z;
        this.device.queue.writeBuffer(this.gridUniformBuffer, 0, gridUniformData);

        const segmentUniformData = new Float32Array(SEGMENT_UNIFORM_FLOATS);
        segmentUniformData.set(viewProjMat, 0);
        segmentUniformData[MATRIX_FLOATS] = Math.max(
            1,
            canvas.clientWidth || canvas.width,
        );
        segmentUniformData[MATRIX_FLOATS + 1] = Math.max(
            1,
            canvas.clientHeight || canvas.height,
        );
        this.device.queue.writeBuffer(this.segmentUniformBuffer, 0, segmentUniformData);

        const triangleUniformData = new Float32Array(MATRIX_FLOATS);
        triangleUniformData.set(viewProjMat, 0);
        this.device.queue.writeBuffer(this.triangleUniformBuffer, 0, triangleUniformData);
    }

    private drawSegmentBuffer(
        pass: GPURenderPassEncoder,
        buffer: VertexBufferState,
        vertexCount: number,
        pipeline: GPURenderPipeline,
    ) {
        if (vertexCount === 0 || !buffer.buffer) return;
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.segmentBindGroup);
        pass.setVertexBuffer(0, buffer.buffer);
        pass.draw(vertexCount);
    }

    private ensureDepthTexture(width: number, height: number) {
        if (this.depthTexture && this.depthWidth === width && this.depthHeight === height) return;
        this.depthTexture?.destroy();
        this.depthTexture = this.device.createTexture({
            label: "paint modeler depth",
            size: [width, height],
            format: DEPTH_FORMAT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.depthWidth = width;
        this.depthHeight = height;
    }
}

const createVertexBufferState = (): VertexBufferState => ({
    buffer: null,
    capacityVertices: 0,
});

const createUniformBuffer = (device: GPUDevice, floatCount: number, label: string): GPUBuffer => device.createBuffer({
    label,
    size: floatCount * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const createUniformBindGroup = (
    device: GPUDevice,
    layout: GPUBindGroupLayout,
    buffer: GPUBuffer,
    label: string,
): GPUBindGroup => device.createBindGroup({
    label,
    layout,
    entries: [{ binding: 0, resource: { buffer } }],
});

const createColorTarget = (format: GPUTextureFormat): GPUColorTargetState => ({
    format,
    blend: {
        color: {
            operation: "add",
            srcFactor: "src-alpha",
            dstFactor: "one-minus-src-alpha",
        },
        alpha: {
            operation: "add",
            srcFactor: "src-alpha",
            dstFactor: "one-minus-src-alpha",
        },
    },
});

const createGridPipeline = (
    device: GPUDevice,
    layout: GPUPipelineLayout,
    module: GPUShaderModule,
    format: GPUTextureFormat,
): GPURenderPipeline => device.createRenderPipeline({
    label: "paint modeler grid pipeline",
    layout,
    vertex: {
        module,
        entryPoint: "grid_vertex",
    },
    fragment: {
        module,
        entryPoint: "grid_fragment",
        targets: [createColorTarget(format)],
    },
    primitive: {
        topology: "triangle-list",
        cullMode: "none",
    },
    depthStencil: {
        format: DEPTH_FORMAT,
        depthCompare: "always",
        depthWriteEnabled: false,
    },
});

const createSegmentPipeline = (
    device: GPUDevice,
    layout: GPUPipelineLayout,
    module: GPUShaderModule,
    format: GPUTextureFormat,
    topology: GPUPrimitiveTopology,
): GPURenderPipeline => device.createRenderPipeline({
    label: `paint modeler segment ${topology} pipeline`,
    layout,
    vertex: {
        module,
        entryPoint: "segment_vertex",
        buffers: [{
            arrayStride: FLOATS_PER_VERTEX * Float32Array.BYTES_PER_ELEMENT,
            attributes: [
                { shaderLocation: 0, offset: 0, format: "float32x3" },
                { shaderLocation: 1, offset: 3 * Float32Array.BYTES_PER_ELEMENT, format: "float32x3" },
                { shaderLocation: 2, offset: 6 * Float32Array.BYTES_PER_ELEMENT, format: "float32x3" },
                { shaderLocation: 3, offset: 9 * Float32Array.BYTES_PER_ELEMENT, format: "float32x4" },
                { shaderLocation: 4, offset: 13 * Float32Array.BYTES_PER_ELEMENT, format: "float32" },
                { shaderLocation: 5, offset: 14 * Float32Array.BYTES_PER_ELEMENT, format: "float32" },
                { shaderLocation: 6, offset: 15 * Float32Array.BYTES_PER_ELEMENT, format: "float32" },
            ],
        }],
    },
    fragment: {
        module,
        entryPoint: "segment_fragment",
        targets: [createColorTarget(format)],
    },
    primitive: {
        topology,
        cullMode: "none",
    },
    depthStencil: {
        format: DEPTH_FORMAT,
        depthCompare: "less-equal",
        depthWriteEnabled: false,
    },
});

const createTrianglePipeline = (
    device: GPUDevice,
    layout: GPUPipelineLayout,
    module: GPUShaderModule,
    format: GPUTextureFormat,
): GPURenderPipeline => device.createRenderPipeline({
    label: "paint modeler triangle pipeline",
    layout,
    vertex: {
        module,
        entryPoint: "triangle_vertex",
        buffers: [{
            arrayStride: FLOATS_PER_TRIANGLE_VERTEX * Float32Array.BYTES_PER_ELEMENT,
            attributes: [
                { shaderLocation: 0, offset: 0, format: "float32x3" },
                { shaderLocation: 1, offset: 3 * Float32Array.BYTES_PER_ELEMENT, format: "float32x4" },
            ],
        }],
    },
    fragment: {
        module,
        entryPoint: "triangle_fragment",
        targets: [createColorTarget(format)],
    },
    primitive: {
        topology: "triangle-list",
        cullMode: "none",
    },
    depthStencil: {
        format: DEPTH_FORMAT,
        depthCompare: "less-equal",
        depthWriteEnabled: true,
    },
});

const uploadVertexData = (
    device: GPUDevice,
    state: VertexBufferState,
    data: Float32Array,
    vertexCount: number,
    label: string,
) => {
    if (vertexCount === 0) return;
    if (vertexCount > state.capacityVertices || !state.buffer) {
        state.buffer?.destroy();
        state.buffer = device.createBuffer({
            label,
            size: data.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        state.capacityVertices = vertexCount;
    }
    device.queue.writeBuffer(state.buffer, 0, data);
};

const destroyVertexBuffer = (state: VertexBufferState) => {
    state.buffer?.destroy();
    state.buffer = null;
    state.capacityVertices = 0;
};

const isRenderSegment = (primitive: RenderPrimitive): primitive is RenderSegment => (
    primitive.kind !== "triangle" && primitive.kind !== "stroke"
);

const isRenderTriangle = (primitive: RenderPrimitive): primitive is RenderTriangle => primitive.kind === "triangle";

const isRenderStroke = (primitive: RenderPrimitive): primitive is RenderStroke => primitive.kind === "stroke";

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

const strokeStripVertexCount = (strokes: RenderStroke[]): number => {
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
    offset = appendTriangleVertex(out, offset, triangle.a, triangle.color);
    offset = appendTriangleVertex(out, offset, triangle.b, triangle.color);
    offset = appendTriangleVertex(out, offset, triangle.c, triangle.color);
    return offset;
};

const appendTriangleVertex = (
    out: Float32Array,
    offset: number,
    position: [number, number, number],
    color: [number, number, number, number],
): number => {
    out[offset++] = position[0];
    out[offset++] = position[1];
    out[offset++] = position[2];
    out[offset++] = color[0];
    out[offset++] = color[1];
    out[offset++] = color[2];
    out[offset++] = color[3];
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
