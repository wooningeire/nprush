struct ChartSample {
    point: vec2f,
    depth: f32,
    _pad: f32,
};

@group(0) @binding(0) var<storage, read_write> fields: array<vec2f>;
@group(0) @binding(1) var<storage, read> samples: array<ChartSample>;
@group(0) @binding(2) var<storage, read> params: array<f32>;

fn chart_width() -> u32 {
    return u32(params[0]);
}

fn chart_height() -> u32 {
    return u32(params[1]);
}

fn sample_count() -> u32 {
    return u32(params[2]);
}

fn require_coverage() -> bool {
    return params[3] > 0.5;
}

fn replace_depth() -> bool {
    return params[4] > 0.5;
}

fn paint_radius() -> f32 {
    return params[5];
}

fn coverage_epsilon() -> f32 {
    return params[6];
}

fn min_depth() -> f32 {
    return params[7];
}

fn grid_uv(index: u32) -> vec2f {
    let width = chart_width();
    let x = index % width;
    let y = index / width;
    let u = select(0.0, f32(x) / f32(width - 1u) * 2.0 - 1.0, width > 1u);
    let v = select(0.0, f32(y) / f32(chart_height() - 1u) * 2.0 - 1.0, chart_height() > 1u);
    return vec2f(u, v);
}

fn nearest_point_on_segment(point: vec2f, a: vec2f, b: vec2f) -> vec2f {
    let delta = b - a;
    let length_squared = dot(delta, delta);
    let t = select(0.0, clamp(dot(point - a, delta) / length_squared, 0.0, 1.0), length_squared > 0.0000000001);
    return vec2f(t, length(point - (a + delta * t)));
}

fn nearest_sample_depth(point: vec2f) -> vec2f {
    let count = sample_count();
    if (count == 1u) {
        return vec2f(length(point - samples[0].point), samples[0].depth);
    }

    var best_distance = 340282346638528859811704183484516925440.0;
    var best_depth = samples[0].depth;
    for (var index = 1u; index < count; index += 1u) {
        let previous = samples[index - 1u];
        let current = samples[index];
        let nearest = nearest_point_on_segment(point, previous.point, current.point);
        if (nearest.y < best_distance) {
            best_distance = nearest.y;
            best_depth = mix(previous.depth, current.depth, nearest.x);
        }
    }
    return vec2f(best_distance, best_depth);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= chart_width() * chart_height() || sample_count() == 0u) {
        return;
    }

    let old_field = fields[index];
    if (require_coverage() && old_field.y <= coverage_epsilon()) {
        return;
    }

    let nearest = nearest_sample_depth(grid_uv(index));
    if (nearest.x > paint_radius()) {
        return;
    }

    let t = nearest.x / max(paint_radius(), 0.00001);
    let influence = pow(1.0 - t * t, 2.0);
    var next_field = old_field;

    let depth = max(min_depth(), nearest.y);
    next_field.x = select(mix(old_field.x, depth, influence), depth, replace_depth());
    next_field.y = max(old_field.y, influence);
    fields[index] = next_field;
}