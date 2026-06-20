struct GridUniforms {
    view_proj_inv: mat4x4f,
    plane_z: f32,
};

struct GridVertexOut {
    @builtin(position) position: vec4f,
    @location(0) ndc: vec2f,
};

@group(0) @binding(0) var<uniform> grid_uniforms: GridUniforms;

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