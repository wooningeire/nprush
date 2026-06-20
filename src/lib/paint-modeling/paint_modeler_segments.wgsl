struct SegmentUniforms {
    view_proj: mat4x4f,
    viewport_size: vec2f,
};

struct SegmentVertexOut {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
};

@group(0) @binding(0) var<uniform> segment_uniforms: SegmentUniforms;

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