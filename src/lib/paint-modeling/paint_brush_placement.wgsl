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

fn direct_hit_at(index: u32) -> bool {
    return placement_results[index].normal.w == PLACEMENT_KIND_SURFACE;
}

fn source_segment_distance_px(a: vec2f, b: vec2f) -> f32 {
    let viewport = max(placement_uniforms.viewport_counts.xy, vec2f(1.0));
    return max(length((b - a) * viewport * 0.5), 0.000001);
}

fn next_source_index(index: u32, rows: u32, closed: bool) -> u32 {
    if (index + 1u < rows) {
        return index + 1u;
    }
    if (closed) {
        return 0u;
    }
    return index;
}

fn previous_source_index(index: u32, rows: u32, closed: bool) -> u32 {
    if (index > 0u) {
        return index - 1u;
    }
    if (closed) {
        return rows - 1u;
    }
    return index;
}

fn find_hit_neighbors(index: u32, rows: u32, closed: bool) -> HitNeighbors {
    var previous = index;
    var next = index;
    var has_previous = 0u;
    var has_next = 0u;

    if (closed) {
        for (var step = 1u; step < rows; step++) {
            let candidate = (index + rows - step) % rows;
            if (direct_hit_at(candidate)) {
                previous = candidate;
                has_previous = 1u;
                break;
            }
        }
        for (var step = 1u; step < rows; step++) {
            let candidate = (index + step) % rows;
            if (direct_hit_at(candidate)) {
                next = candidate;
                has_next = 1u;
                break;
            }
        }
    } else {
        var candidate = index;
        loop {
            if (candidate == 0u) {
                break;
            }
            candidate -= 1u;
            if (direct_hit_at(candidate)) {
                previous = candidate;
                has_previous = 1u;
                break;
            }
        }

        candidate = index;
        loop {
            if (candidate + 1u >= rows) {
                break;
            }
            candidate += 1u;
            if (direct_hit_at(candidate)) {
                next = candidate;
                has_next = 1u;
                break;
            }
        }
    }

    return HitNeighbors(previous, next, has_previous, has_next);
}

fn forward_arc_distance(start_index: u32, end_index: u32, rows: u32, closed: bool) -> f32 {
    if (start_index == end_index) {
        return 0.0;
    }

    var total = 0.0;
    var current = start_index;
    for (var step = 0u; step < rows; step++) {
        let next = next_source_index(current, rows, closed);
        if (next == current) {
            break;
        }
        total += source_segment_distance_px(
            source_point(current, rows, closed),
            source_point(next, rows, closed),
        );
        current = next;
        if (current == end_index) {
            break;
        }
    }
    return total;
}

fn plane_derivative_before(index: u32, rows: u32, closed: bool) -> PlaneDerivative {
    let previous = previous_source_index(index, rows, closed);
    if (previous == index || !direct_hit_at(previous)) {
        return PlaneDerivative(vec3f(0.0), vec3f(0.0));
    }

    let current_plane = placement_results[index];
    let previous_plane = placement_results[previous];
    let distance_px = source_segment_distance_px(
        source_point(previous, rows, closed),
        source_point(index, rows, closed),
    );
    let origin_derivative = (current_plane.center.xyz - previous_plane.center.xyz) / distance_px;
    var previous_normal = previous_plane.normal.xyz;
    if (dot(previous_normal, current_plane.normal.xyz) < 0.0) {
        previous_normal = -previous_normal;
    }
    var normal_derivative = (current_plane.normal.xyz - previous_normal) / distance_px;
    normal_derivative -= current_plane.normal.xyz * dot(normal_derivative, current_plane.normal.xyz);
    return PlaneDerivative(origin_derivative, normal_derivative);
}

fn plane_derivative_after(index: u32, rows: u32, closed: bool) -> PlaneDerivative {
    let next = next_source_index(index, rows, closed);
    if (next == index || !direct_hit_at(next)) {
        return PlaneDerivative(vec3f(0.0), vec3f(0.0));
    }

    let current_plane = placement_results[index];
    let next_plane = placement_results[next];
    let distance_px = source_segment_distance_px(
        source_point(index, rows, closed),
        source_point(next, rows, closed),
    );
    let origin_derivative = (next_plane.center.xyz - current_plane.center.xyz) / distance_px;
    var next_normal = next_plane.normal.xyz;
    if (dot(next_normal, current_plane.normal.xyz) < 0.0) {
        next_normal = -next_normal;
    }
    var normal_derivative = (next_normal - current_plane.normal.xyz) / distance_px;
    normal_derivative -= current_plane.normal.xyz * dot(normal_derivative, current_plane.normal.xyz);
    return PlaneDerivative(origin_derivative, normal_derivative);
}

fn clamp_vector_length(value: vec3f, maximum: f32) -> vec3f {
    let value_length = length(value);
    if (value_length <= maximum || value_length <= 0.000001) {
        return value;
    }
    return value * (maximum / value_length);
}

fn hermite_vec3(
    a: vec3f,
    tangent_a: vec3f,
    b: vec3f,
    tangent_b: vec3f,
    u: f32,
) -> vec3f {
    let u2 = u * u;
    let u3 = u2 * u;
    let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
    let h10 = u3 - 2.0 * u2 + u;
    let h01 = -2.0 * u3 + 3.0 * u2;
    let h11 = u3 - u2;
    return h00 * a + h10 * tangent_a + h01 * b + h11 * tangent_b;
}

fn resolve_surface_stroke_placement(
    point: vec2f,
    index: u32,
    rows: u32,
    closed: bool,
) -> PlacementResult {
    if (direct_hit_at(index)) {
        return placement_results[index];
    }

    let neighbors = find_hit_neighbors(index, rows, closed);
    if (neighbors.has_previous != 0u && neighbors.has_next != 0u) {
        let a = placement_results[neighbors.previous];
        let b = placement_results[neighbors.next];
        if (neighbors.previous == neighbors.next) {
            return resolve_plane_placement(point, a.center.xyz, a.normal.xyz, PLACEMENT_KIND_BRIDGE);
        }

        let distance_from_a = forward_arc_distance(neighbors.previous, index, rows, closed);
        let distance_to_b = forward_arc_distance(index, neighbors.next, rows, closed);
        let gap_distance = max(distance_from_a + distance_to_b, 0.000001);
        let u = clamp(distance_from_a / gap_distance, 0.0, 1.0);
        let derivative_a = plane_derivative_before(neighbors.previous, rows, closed);
        var derivative_b = plane_derivative_after(neighbors.next, rows, closed);
        var normal_b = b.normal.xyz;
        if (dot(a.normal.xyz, normal_b) < 0.0) {
            normal_b = -normal_b;
            derivative_b.normal = -derivative_b.normal;
        }

        let chord_length = length(b.center.xyz - a.center.xyz);
        let tangent_limit = max(chord_length * 1.5, 0.001);
        let origin = hermite_vec3(
            a.center.xyz,
            clamp_vector_length(derivative_a.origin * gap_distance, tangent_limit),
            b.center.xyz,
            clamp_vector_length(derivative_b.origin * gap_distance, tangent_limit),
            u,
        );
        let normal_candidate = hermite_vec3(
            a.normal.xyz,
            clamp_vector_length(derivative_a.normal * gap_distance, 1.25),
            normal_b,
            clamp_vector_length(derivative_b.normal * gap_distance, 1.25),
            u,
        );
        var normal = normalize(mix(a.normal.xyz, normal_b, u));
        if (length(normal_candidate) > 0.000001) {
            normal = normalize(normal_candidate);
        }
        return resolve_plane_placement(point, origin, normal, PLACEMENT_KIND_BRIDGE);
    }

    if (neighbors.has_previous != 0u) {
        let plane = placement_results[neighbors.previous];
        return resolve_plane_placement(point, plane.center.xyz, plane.normal.xyz, PLACEMENT_KIND_BRIDGE);
    }
    if (neighbors.has_next != 0u) {
        let plane = placement_results[neighbors.next];
        return resolve_plane_placement(point, plane.center.xyz, plane.normal.xyz, PLACEMENT_KIND_BRIDGE);
    }
    return resolve_view_placement(point);
}

fn resolve_stroke_placement(
    point: vec2f,
    index: u32,
    rows: u32,
    closed: bool,
) -> PlacementResult {
    let mode = placement_mode();
    if (mode < PLACEMENT_MODE_START_DEPTH) {
        return resolve_view_placement(point);
    }
    if (mode < PLACEMENT_MODE_START_PLANE) {
        let anchor = placement_results[0];
        if (!direct_hit_at(0u)) {
            return resolve_view_placement(point);
        }
        return resolve_plane_placement(
            point,
            anchor.center.xyz,
            -view_forward(),
            PLACEMENT_KIND_START_DEPTH,
        );
    }
    if (mode < PLACEMENT_MODE_SURFACE) {
        let anchor = placement_results[0];
        if (!direct_hit_at(0u)) {
            return resolve_view_placement(point);
        }
        return resolve_plane_placement(
            point,
            anchor.center.xyz,
            anchor.normal.xyz,
            PLACEMENT_KIND_START_PLANE,
        );
    }
    if (mode < PLACEMENT_MODE_CONSTRUCTION_PLANE) {
        return resolve_surface_stroke_placement(point, index, rows, closed);
    }
    return resolve_plane_placement(
        point,
        placement_uniforms.construction_origin.xyz,
        placement_uniforms.construction_normal.xyz,
        PLACEMENT_KIND_CONSTRUCTION_PLANE,
    );
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
        placement_results[0] = zero_placement();
        return;
    }

    placement_results[0] = resolve_hover_placement(placement_uniforms.pointer.xy);
}

@compute @workgroup_size(64)
fn compute_direct_stroke(@builtin(global_invocation_id) global_id: vec3u) {
    let count = source_count();
    if (count < 2u) {
        return;
    }

    let closed = stroke_closed(count);
    let rows = row_count(count, closed);
    if (global_id.x >= rows) {
        return;
    }

    let mode = placement_mode();
    let index = global_id.x;
    if (
        mode < PLACEMENT_MODE_START_DEPTH
        || mode >= PLACEMENT_MODE_CONSTRUCTION_PLANE
        || (mode < PLACEMENT_MODE_SURFACE && index != 0u)
    ) {
        placement_results[index] = zero_placement();
        return;
    }

    placement_results[index] = resolve_direct_surface_placement(
        source_point(index, rows, closed),
    );
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
    let placement = resolve_stroke_placement(point, index, rows, closed);
    let side_ray = make_ray(point + stroke_side_offset(index, rows, closed));
    let side_point = plane_point(side_ray, placement.center.xyz, placement.normal.xyz);
    let u = select(
        f32(index) / max(f32(rows - 1u), 1.0),
        f32(index) / max(f32(rows), 1.0),
        closed,
    );

    placed_vertices[index] = RibbonVertex(
        vec4f(placement.center.xyz, u),
        vec4f(side_point - placement.center.xyz, placement.normal.w),
    );
}