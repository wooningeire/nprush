const GUIDE_CIRCLE_SEGMENTS: u32 = 48u;
const GUIDE_VERTEX_COUNT: u32 = GUIDE_CIRCLE_SEGMENTS * 2u + 2u;
const GUIDE_DEPTH_BIAS: f32 = 0.0008;
const GUIDE_NORMAL_LENGTH_SCALE: f32 = 1.0;
const PI: f32 = 3.141592653589793;

struct PlacementUniforms {
    view_proj: mat4x4f,
    view_proj_inv: mat4x4f,
    view_inv: mat4x4f,
    pointer: vec4f,
    viewport_counts: vec4f,
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
};

@group(0) @binding(0) var<uniform> placement_uniforms: PlacementUniforms;
@group(0) @binding(4) var<storage, read> placement_results: array<PlacementResult>;

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

@vertex
fn guide_vertex(@builtin(vertex_index) vertex_index: u32) -> GuideVertexOut {
    let placement = placement_results[0];
    var position = vec3f(0.0);
    var color = vec4f(0.0);

    if (placement.center.w <= 0.5 || vertex_index >= GUIDE_VERTEX_COUNT) {
        var out: GuideVertexOut;
        out.position = vec4f(2.0, 2.0, 0.0, 1.0);
        out.color = vec4f(0.0);
        return out;
    }

    if (vertex_index < GUIDE_CIRCLE_SEGMENTS * 2u) {
        let segment = vertex_index / 2u;
        let endpoint = vertex_index - segment * 2u;
        position = circle_position(placement, segment, endpoint);
        color = mix(
            vec4f(1.0, 0.76, 0.34, 0.86),
            vec4f(0.48, 0.96, 0.76, 0.94),
            clamp(placement.normal.w, 0.0, 1.0),
        );
    } else {
        let endpoint = vertex_index - GUIDE_CIRCLE_SEGMENTS * 2u;
        let radius = guide_radius(placement);
        position = placement.center.xyz
            + placement.normal.xyz * radius * GUIDE_NORMAL_LENGTH_SCALE * f32(endpoint);
        color = vec4f(0.54, 0.78, 1.0, 0.95);
    }

    var out: GuideVertexOut;
    out.position = guide_clip_position(position);
    out.color = color;
    return out;
}

@fragment
fn guide_fragment(in: GuideVertexOut) -> @location(0) vec4f {
    return in.color;
}
