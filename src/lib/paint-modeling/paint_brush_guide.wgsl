const GUIDE_CIRCLE_SEGMENTS: u32 = 48u;
const GUIDE_CIRCLE_VERTEX_COUNT: u32 = GUIDE_CIRCLE_SEGMENTS * 2u;
const GUIDE_BRUSH_VERTEX_COUNT: u32 = GUIDE_CIRCLE_VERTEX_COUNT + 2u;
const GUIDE_GRID_HALF_LINES: u32 = 8u;
const GUIDE_GRID_LINE_COUNT: u32 = GUIDE_GRID_HALF_LINES * 2u + 1u;
const GUIDE_GRID_VERTEX_COUNT: u32 = GUIDE_GRID_LINE_COUNT * 4u;
const GUIDE_GRID_VERTEX_START: u32 = GUIDE_BRUSH_VERTEX_COUNT;
const GUIDE_PLANE_NORMAL_START: u32 = GUIDE_GRID_VERTEX_START + GUIDE_GRID_VERTEX_COUNT;
const GUIDE_VERTEX_COUNT: u32 = GUIDE_PLANE_NORMAL_START + 2u;
const GUIDE_DEPTH_BIAS: f32 = 0.0008;
const GUIDE_NORMAL_LENGTH_SCALE: f32 = 1.0;
const PI: f32 = 3.141592653589793;

const PLACEMENT_KIND_VIEW: f32 = 0.0;
const PLACEMENT_KIND_SURFACE: f32 = 1.0;
const PLACEMENT_KIND_BRIDGE: f32 = 2.0;
const PLACEMENT_KIND_START_DEPTH: f32 = 3.0;
const PLACEMENT_KIND_START_PLANE: f32 = 4.0;
const PLACEMENT_KIND_CONSTRUCTION_PLANE: f32 = 5.0;

struct PlacementUniforms {
    view_proj: mat4x4f,
    view_proj_inv: mat4x4f,
    view_inv: mat4x4f,
    pointer: vec4f,
    viewport_counts: vec4f,
    placement: vec4f,
    construction_origin: vec4f,
    construction_normal: vec4f,
    start_point: vec4f,
};

struct PlacementResult {
    center: vec4f,
    normal: vec4f,
    tangent: vec4f,
    bitangent: vec4f,
};

struct GuideVertexOut {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) style: f32,
};

@group(0) @binding(0) var<uniform> placement_uniforms: PlacementUniforms;
@group(0) @binding(4) var<storage, read> placement_results: array<PlacementResult>;

fn hidden_vertex() -> GuideVertexOut {
    var out: GuideVertexOut;
    out.position = vec4f(2.0, 2.0, 0.0, 1.0);
    out.color = vec4f(0.0);
    out.style = 0.0;
    return out;
}

fn circle_position(placement: PlacementResult, segment: u32, endpoint: u32) -> vec3f {
    let angle = (f32(segment + endpoint) / f32(GUIDE_CIRCLE_SEGMENTS)) * PI * 2.0;
    return placement.center.xyz
        + placement.tangent.xyz * cos(angle)
        + placement.bitangent.xyz * sin(angle);
}

fn guide_radius(placement: PlacementResult) -> f32 {
    return max(length(placement.tangent.xyz), length(placement.bitangent.xyz));
}

fn guide_clip_position(position: vec3f) -> vec4f {
    var clip = placement_uniforms.view_proj * vec4f(position, 1.0);
    clip.z -= GUIDE_DEPTH_BIAS * clip.w;
    return clip;
}

fn placement_color(kind: f32) -> vec4f {
    if (kind < PLACEMENT_KIND_SURFACE) {
        return vec4f(1.0, 0.76, 0.34, 0.9);
    }
    if (kind < PLACEMENT_KIND_BRIDGE) {
        return vec4f(0.48, 0.96, 0.76, 0.96);
    }
    if (kind < PLACEMENT_KIND_START_DEPTH) {
        return vec4f(1.0, 0.63, 0.24, 0.96);
    }
    if (kind < PLACEMENT_KIND_START_PLANE) {
        return vec4f(0.97, 0.86, 0.36, 0.96);
    }
    if (kind < PLACEMENT_KIND_CONSTRUCTION_PLANE) {
        return vec4f(0.35, 0.9, 0.84, 0.96);
    }
    return vec4f(0.42, 0.72, 1.0, 0.98);
}

fn normal_plane_axis(preferred: vec3f, normal: vec3f, fallback: vec3f) -> vec3f {
    let preferred_axis = preferred - normal * dot(preferred, normal);
    let preferred_length = length(preferred_axis);
    if (preferred_length > 0.000001) {
        return preferred_axis / preferred_length;
    }

    let fallback_axis = fallback - normal * dot(fallback, normal);
    let fallback_length = length(fallback_axis);
    if (fallback_length > 0.000001) {
        return fallback_axis / fallback_length;
    }

    let world_x = vec3f(1.0, 0.0, 0.0) - normal * normal.x;
    let world_x_length = length(world_x);
    if (world_x_length > 0.000001) {
        return world_x / world_x_length;
    }

    return vec3f(0.0, 1.0, 0.0);
}

fn construction_axes(normal: vec3f) -> mat2x3f {
    let tangent = normal_plane_axis(
        placement_uniforms.view_inv[0].xyz,
        normal,
        placement_uniforms.view_inv[1].xyz,
    );
    return mat2x3f(tangent, normalize(cross(normal, tangent)));
}

fn grid_color(offset: i32, second_axis: bool) -> vec4f {
    if (offset == 0) {
        if (second_axis) {
            return vec4f(0.94, 0.42, 0.38, 0.84);
        }
        return vec4f(0.46, 0.86, 0.55, 0.84);
    }

    let major = offset % 4 == 0;
    return select(
        vec4f(0.42, 0.69, 0.74, 0.24),
        vec4f(0.55, 0.82, 0.86, 0.48),
        major,
    );
}

@vertex
fn guide_vertex(@builtin(vertex_index) vertex_index: u32) -> GuideVertexOut {
    if (vertex_index >= GUIDE_VERTEX_COUNT) {
        return hidden_vertex();
    }

    let placement = placement_results[0];
    var position = vec3f(0.0);
    var color = vec4f(0.0);
    var style = 0.0;

    if (vertex_index < GUIDE_CIRCLE_VERTEX_COUNT) {
        if (placement_uniforms.pointer.w <= 0.5 || placement.center.w <= 0.5) {
            return hidden_vertex();
        }

        let segment = vertex_index / 2u;
        let endpoint = vertex_index - segment * 2u;
        position = circle_position(placement, segment, endpoint);
        color = placement_color(placement.normal.w);
        if (
            placement.normal.w == PLACEMENT_KIND_BRIDGE
            && segment % 2u != 0u
        ) {
            color.a = 0.0;
        }
    } else if (vertex_index < GUIDE_BRUSH_VERTEX_COUNT) {
        if (placement_uniforms.pointer.w <= 0.5 || placement.center.w <= 0.5) {
            return hidden_vertex();
        }

        let endpoint = vertex_index - GUIDE_CIRCLE_VERTEX_COUNT;
        position = placement.center.xyz
            + placement.normal.xyz
            * guide_radius(placement)
            * GUIDE_NORMAL_LENGTH_SCALE
            * f32(endpoint);
        color = vec4f(0.54, 0.78, 1.0, 0.95);
        style = 1.0;
    } else if (vertex_index < GUIDE_PLANE_NORMAL_START) {
        if (placement_uniforms.placement.y <= 0.5) {
            return hidden_vertex();
        }

        let local_index = vertex_index - GUIDE_GRID_VERTEX_START;
        let line_vertex = local_index / 2u;
        let endpoint = local_index % 2u;
        let second_axis = line_vertex >= GUIDE_GRID_LINE_COUNT;
        let line_index = line_vertex % GUIDE_GRID_LINE_COUNT;
        let offset = i32(line_index) - i32(GUIDE_GRID_HALF_LINES);
        let origin = placement_uniforms.construction_origin.xyz;
        let radius = max(placement_uniforms.construction_origin.w, 0.01);
        let normal = normalize(placement_uniforms.construction_normal.xyz);
        let axes = construction_axes(normal);
        let across = mix(-radius, radius, f32(endpoint));
        let spacing = radius / f32(GUIDE_GRID_HALF_LINES);

        if (second_axis) {
            position = origin + axes[0] * across + axes[1] * f32(offset) * spacing;
        } else {
            position = origin + axes[0] * f32(offset) * spacing + axes[1] * across;
        }
        color = grid_color(offset, second_axis);
        style = 2.0;
    } else {
        if (placement_uniforms.placement.y <= 0.5) {
            return hidden_vertex();
        }

        let endpoint = vertex_index - GUIDE_PLANE_NORMAL_START;
        let radius = max(placement_uniforms.construction_origin.w, 0.01);
        position = placement_uniforms.construction_origin.xyz
            + normalize(placement_uniforms.construction_normal.xyz)
            * radius
            * 0.35
            * f32(endpoint);
        color = vec4f(0.54, 0.78, 1.0, 0.98);
        style = 3.0;
    }

    var out: GuideVertexOut;
    out.position = guide_clip_position(position);
    out.color = color;
    out.style = style;
    return out;
}

@fragment
fn guide_fragment(in: GuideVertexOut) -> @location(0) vec4f {
    return in.color;
}

@fragment
fn guide_fragment_xray(in: GuideVertexOut) -> @location(0) vec4f {
    if (in.style < 1.5) {
        discard;
    }
    return vec4f(in.color.rgb, in.color.a * 0.16);
}