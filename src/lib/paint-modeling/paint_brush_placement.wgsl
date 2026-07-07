const MIN_DEPTH: f32 = 0.06;

struct PlacementUniforms {
    view_proj: mat4x4f,
    view_proj_inv: mat4x4f,
    view_inv: mat4x4f,
    pointer: vec4f,
    viewport_counts: vec4f,
};

struct RibbonTarget {
    start: u32,
    rows: u32,
    closed: u32,
    _pad: u32,
};

struct RibbonVertex {
    position_u: vec4f,
    side: vec4f,
};

struct PlacementResult {
    center: vec4f,
    normal: vec4f,
    tangent: vec4f,
    bitangent: vec4f,
};

struct Ray {
    origin: vec3f,
    direction: vec3f,
};

struct TriangleHit {
    hit: f32,
    distance: f32,
    normal: vec3f,
};

@group(0) @binding(0) var<uniform> placement_uniforms: PlacementUniforms;
@group(0) @binding(1) var<storage, read> target_vertices: array<RibbonVertex>;
@group(0) @binding(2) var<storage, read> target_infos: array<RibbonTarget>;
@group(0) @binding(3) var<storage, read> source_points: array<vec4f>;
@group(0) @binding(4) var<storage, read_write> placement_results: array<PlacementResult>;
@group(0) @binding(5) var<storage, read_write> placed_vertices: array<RibbonVertex>;
@group(0) @binding(6) var<storage, read_write> placed_meta: array<vec4u>;

fn unproject_ndc(point: vec3f) -> vec3f {
    let world = placement_uniforms.view_proj_inv * vec4f(point, 1.0);
    return world.xyz / max(abs(world.w), 0.000001) * sign(world.w);
}

fn make_ray(point: vec2f) -> Ray {
    let near = unproject_ndc(vec3f(point, -1.0));
    let far = unproject_ndc(vec3f(point, 1.0));
    return Ray(
        placement_uniforms.view_inv[3].xyz,
        normalize(far - near),
    );
}

fn view_forward() -> vec3f {
    return make_ray(vec2f(0.0)).direction;
}

fn default_center(ray: Ray) -> vec3f {
    let forward = view_forward();
    let depth = max(MIN_DEPTH, dot(-ray.origin, forward));
    let denominator = dot(ray.direction, forward);
    if (abs(denominator) <= 0.000001) {
        return ray.origin + forward * depth;
    }
    return ray.origin + ray.direction * (depth / denominator);
}

fn ribbon_row(ribbon_info: RibbonTarget, row: u32) -> RibbonVertex {
    return target_vertices[ribbon_info.start + row];
}

fn next_target_row(row: u32, rows: u32) -> u32 {
    if (row + 1u >= rows) {
        return 0u;
    }
    return row + 1u;
}

fn intersect_triangle(ray: Ray, a: vec3f, b: vec3f, c: vec3f) -> TriangleHit {
    let edge1 = b - a;
    let edge2 = c - a;
    let h = cross(ray.direction, edge2);
    let determinant = dot(edge1, h);
    if (abs(determinant) <= 0.000001) {
        return TriangleHit(0.0, 0.0, vec3f(0.0, 0.0, 1.0));
    }

    let inverse_determinant = 1.0 / determinant;
    let s = ray.origin - a;
    let u = inverse_determinant * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        return TriangleHit(0.0, 0.0, vec3f(0.0, 0.0, 1.0));
    }

    let q = cross(s, edge1);
    let v = inverse_determinant * dot(ray.direction, q);
    if (v < 0.0 || u + v > 1.0) {
        return TriangleHit(0.0, 0.0, vec3f(0.0, 0.0, 1.0));
    }

    let distance = inverse_determinant * dot(edge2, q);
    if (distance <= 0.000001) {
        return TriangleHit(0.0, 0.0, vec3f(0.0, 0.0, 1.0));
    }

    var normal = normalize(cross(edge1, edge2));
    if (dot(normal, ray.direction) > 0.0) {
        normal = -normal;
    }
    return TriangleHit(1.0, distance, normal);
}

fn target_segment_hit(ray: Ray, ribbon_info: RibbonTarget, row: u32, best_distance: f32) -> TriangleHit {
    let next_row = next_target_row(row, ribbon_info.rows);
    let a_vertex = ribbon_row(ribbon_info, row);
    let b_vertex = ribbon_row(ribbon_info, next_row);
    let a0 = a_vertex.position_u.xyz - a_vertex.side.xyz;
    let a1 = a_vertex.position_u.xyz + a_vertex.side.xyz;
    let b0 = b_vertex.position_u.xyz - b_vertex.side.xyz;
    let b1 = b_vertex.position_u.xyz + b_vertex.side.xyz;
    let first = intersect_triangle(ray, a0, b0, b1);
    let second = intersect_triangle(ray, a0, b1, a1);

    var best = TriangleHit(0.0, best_distance, vec3f(0.0, 0.0, 1.0));
    if (first.hit > 0.5 && first.distance < best.distance) {
        best = first;
    }
    if (second.hit > 0.5 && second.distance < best.distance) {
        best = second;
    }
    return best;
}

fn snapped_hit(ray: Ray) -> TriangleHit {
    let target_count = u32(max(placement_uniforms.viewport_counts.z, 0.0));
    var best = TriangleHit(0.0, 1000000000.0, vec3f(0.0, 0.0, 1.0));
    for (var target_index = 0u; target_index < target_count; target_index++) {
        let ribbon_info = target_infos[target_index];
        if (ribbon_info.rows < 2u) {
            continue;
        }

        let segment_count = select(ribbon_info.rows - 1u, ribbon_info.rows, ribbon_info.closed != 0u);
        for (var row = 0u; row < segment_count; row++) {
            let hit = target_segment_hit(ray, ribbon_info, row, best.distance);
            if (hit.hit > 0.5 && hit.distance < best.distance) {
                best = hit;
            }
        }
    }
    return best;
}

fn plane_point(ray: Ray, center: vec3f, normal: vec3f) -> vec3f {
    let denominator = dot(ray.direction, normal);
    if (abs(denominator) <= 0.000001) {
        return center;
    }

    let distance = dot(center - ray.origin, normal) / denominator;
    if (distance <= 0.000001) {
        return center;
    }
    return ray.origin + ray.direction * distance;
}

fn half_width_axis(center: vec3f, normal: vec3f, point: vec2f, offset: vec2f) -> vec3f {
    let ray = make_ray(point + offset);
    return plane_point(ray, center, normal) - center;
}

fn resolve_placement(point: vec2f) -> PlacementResult {
    let ray = make_ray(point);
    var center = default_center(ray);
    var normal = -view_forward();
    var snapped = 0.0;

    let hit = snapped_hit(ray);
    if (hit.hit > 0.5) {
        center = ray.origin + ray.direction * hit.distance;
        normal = hit.normal;
        snapped = 1.0;
    }

    let viewport = max(placement_uniforms.viewport_counts.xy, vec2f(1.0));
    let brush_width = max(placement_uniforms.pointer.z, 1.0);
    let tangent = half_width_axis(center, normal, point, vec2f(brush_width / viewport.x, 0.0));
    let bitangent = half_width_axis(center, normal, point, vec2f(0.0, brush_width / viewport.y));
    let depth = dot(center - ray.origin, view_forward());

    return PlacementResult(
        vec4f(center, placement_uniforms.pointer.w),
        vec4f(normal, snapped),
        vec4f(tangent, depth),
        vec4f(bitangent, 0.0),
    );
}

fn source_count() -> u32 {
    return u32(max(placement_uniforms.viewport_counts.w, 0.0));
}

fn stroke_closed(count: u32) -> bool {
    if (count <= 3u) {
        return false;
    }

    let first = source_points[0].xy;
    let last = source_points[count - 1u].xy;
    return distance(first, last) <= 0.04;
}

fn row_count(count: u32, closed: bool) -> u32 {
    return select(count, count - 1u, closed);
}

fn source_point(index: u32, rows: u32, closed: bool) -> vec2f {
    if (index < rows) {
        return source_points[index].xy;
    }
    if (closed) {
        return source_points[0].xy;
    }
    return source_points[rows - 1u].xy;
}

fn stroke_side_offset(index: u32, rows: u32, closed: bool) -> vec2f {
    let point = source_point(index, rows, closed);
    let previous_index = select(index, index - 1u, index > 0u);
    let next_index = select(index, index + 1u, index + 1u < rows);
    let previous = select(
        point,
        source_point(rows - 1u, rows, closed),
        index == 0u && closed,
    );
    let next = select(
        source_point(next_index, rows, closed),
        source_point(0u, rows, closed),
        index + 1u >= rows && closed,
    );
    let stable_previous = select(previous, source_point(previous_index, rows, closed), !(index == 0u && closed));
    let stable_next = select(next, source_point(next_index, rows, closed), !(index + 1u >= rows && closed));
    let viewport = max(placement_uniforms.viewport_counts.xy, vec2f(1.0));
    let delta_px = (stable_next - stable_previous) * viewport * 0.5;
    let length_px = length(delta_px);
    let brush_width = max(placement_uniforms.pointer.z, 1.0);

    if (length_px <= 0.000001) {
        return vec2f(0.0, brush_width / viewport.y);
    }

    return vec2f(
        -delta_px.y / length_px * brush_width / viewport.x,
        delta_px.x / length_px * brush_width / viewport.y,
    );
}

@compute @workgroup_size(1)
fn compute_hover() {
    if (placement_uniforms.pointer.w <= 0.5) {
        placement_results[0] = PlacementResult(
            vec4f(0.0),
            vec4f(0.0),
            vec4f(0.0),
            vec4f(0.0),
        );
        return;
    }

    placement_results[0] = resolve_placement(placement_uniforms.pointer.xy);
}

@compute @workgroup_size(64)
fn compute_stroke(@builtin(global_invocation_id) global_id: vec3u) {
    let count = source_count();
    if (count < 2u) {
        if (global_id.x == 0u) {
            placed_meta[0] = vec4u(0u);
        }
        return;
    }

    let closed = stroke_closed(count);
    let rows = row_count(count, closed);
    if (global_id.x == 0u) {
        placed_meta[0] = vec4u(rows, select(0u, 1u, closed), 0u, 0u);
    }
    if (global_id.x >= rows) {
        return;
    }

    let index = global_id.x;
    let point = source_point(index, rows, closed);
    let placement = resolve_placement(point);
    let side_ray = make_ray(point + stroke_side_offset(index, rows, closed));
    let side_point = plane_point(side_ray, placement.center.xyz, placement.normal.xyz);
    let u = select(
        f32(index) / max(f32(rows - 1u), 1.0),
        f32(index) / max(f32(rows), 1.0),
        closed,
    );

    placed_vertices[index] = RibbonVertex(
        vec4f(placement.center.xyz, u),
        vec4f(side_point - placement.center.xyz, 0.0),
    );
}
