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
