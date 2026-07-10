const MIN_DEPTH: f32 = 0.06;
const PLACEMENT_MODE_VIEW: f32 = 0.0;
const PLACEMENT_MODE_START_DEPTH: f32 = 1.0;
const PLACEMENT_MODE_START_PLANE: f32 = 2.0;
const PLACEMENT_MODE_SURFACE: f32 = 3.0;
const PLACEMENT_MODE_CONSTRUCTION_PLANE: f32 = 4.0;

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

struct PlaneProjection {
    point: vec3f,
    valid: f32,
};

struct HitNeighbors {
    previous: u32,
    next: u32,
    has_previous: u32,
    has_next: u32,
};

struct PlaneDerivative {
    origin: vec3f,
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

fn surface_hit(ray: Ray) -> TriangleHit {
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

fn project_plane(ray: Ray, origin: vec3f, normal: vec3f) -> PlaneProjection {
    let denominator = dot(ray.direction, normal);
    if (abs(denominator) <= 0.000001) {
        return PlaneProjection(origin, 0.0);
    }

    let distance = dot(origin - ray.origin, normal) / denominator;
    if (distance <= 0.000001) {
        return PlaneProjection(origin, 0.0);
    }
    return PlaneProjection(ray.origin + ray.direction * distance, 1.0);
}

fn plane_point(ray: Ray, origin: vec3f, normal: vec3f) -> vec3f {
    let projection = project_plane(ray, origin, normal);
    return select(origin, projection.point, projection.valid > 0.5);
}

fn half_width_axis(center: vec3f, normal: vec3f, point: vec2f, offset: vec2f) -> vec3f {
    let ray = make_ray(point + offset);
    return plane_point(ray, center, normal) - center;
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

    let view_right = placement_uniforms.view_inv[0].xyz;
    let view_axis = view_right - normal * dot(view_right, normal);
    let view_length = length(view_axis);
    if (view_length > 0.000001) {
        return view_axis / view_length;
    }

    let world_x = vec3f(1.0, 0.0, 0.0) - normal * normal.x;
    let world_x_length = length(world_x);
    if (world_x_length > 0.000001) {
        return world_x / world_x_length;
    }

    let world_z = vec3f(0.0, 0.0, 1.0) - normal * normal.z;
    let world_z_length = length(world_z);
    if (world_z_length > 0.000001) {
        return world_z / world_z_length;
    }

    return vec3f(0.0, 1.0, 0.0);
}

fn placement_mode() -> f32 {
    return placement_uniforms.placement.x;
}

fn zero_placement() -> PlacementResult {
    return PlacementResult(
        vec4f(0.0),
        vec4f(0.0),
        vec4f(0.0),
        vec4f(0.0),
    );
}

fn build_placement_result(
    ray: Ray,
    point: vec2f,
    center: vec3f,
    normal_value: vec3f,
    kind: f32,
) -> PlacementResult {
    let normal = normalize(normal_value);
    let viewport = max(placement_uniforms.viewport_counts.xy, vec2f(1.0));
    let brush_width = max(placement_uniforms.pointer.z, 1.0);
    let guide_plane_normal = -view_forward();
    let screen_tangent = half_width_axis(
        center,
        guide_plane_normal,
        point,
        vec2f(brush_width / viewport.x, 0.0),
    );
    let screen_bitangent = half_width_axis(
        center,
        guide_plane_normal,
        point,
        vec2f(0.0, brush_width / viewport.y),
    );
    let guide_radius = max(max(length(screen_tangent), length(screen_bitangent)), 0.000001);
    let tangent_direction = normal_plane_axis(screen_tangent, normal, screen_bitangent);
    let bitangent_direction = normalize(cross(normal, tangent_direction));
    let tangent = tangent_direction * guide_radius;
    let bitangent = bitangent_direction * guide_radius;
    let depth = dot(center - ray.origin, view_forward());

    return PlacementResult(
        vec4f(center, placement_uniforms.pointer.w),
        vec4f(normal, kind),
        vec4f(tangent, depth),
        vec4f(bitangent, 0.0),
    );
}

fn resolve_view_placement(point: vec2f) -> PlacementResult {
    let ray = make_ray(point);
    return build_placement_result(
        ray,
        point,
        default_center(ray),
        -view_forward(),
        PLACEMENT_KIND_VIEW,
    );
}

fn resolve_direct_surface_placement(point: vec2f) -> PlacementResult {
    let ray = make_ray(point);
    let hit = surface_hit(ray);
    if (hit.hit <= 0.5) {
        return resolve_view_placement(point);
    }

    return build_placement_result(
        ray,
        point,
        ray.origin + ray.direction * hit.distance,
        hit.normal,
        PLACEMENT_KIND_SURFACE,
    );
}

fn resolve_plane_placement(
    point: vec2f,
    origin: vec3f,
    normal: vec3f,
    kind: f32,
) -> PlacementResult {
    let ray = make_ray(point);
    let projection = project_plane(ray, origin, normalize(normal));
    if (projection.valid <= 0.5) {
        return resolve_view_placement(point);
    }
    return build_placement_result(ray, point, projection.point, normal, kind);
}

fn resolve_start_placement(
    point: vec2f,
    anchor_point: vec2f,
    use_surface_normal: bool,
) -> PlacementResult {
    let anchor = resolve_direct_surface_placement(anchor_point);
    if (anchor.normal.w != PLACEMENT_KIND_SURFACE) {
        return resolve_view_placement(point);
    }

    let normal = select(-view_forward(), anchor.normal.xyz, use_surface_normal);
    let kind = select(PLACEMENT_KIND_START_DEPTH, PLACEMENT_KIND_START_PLANE, use_surface_normal);
    return resolve_plane_placement(point, anchor.center.xyz, normal, kind);
}

fn resolve_hover_placement(point: vec2f) -> PlacementResult {
    let mode = placement_mode();
    if (mode < PLACEMENT_MODE_START_DEPTH) {
        return resolve_view_placement(point);
    }
    if (mode < PLACEMENT_MODE_START_PLANE) {
        let anchor_point = select(point, placement_uniforms.start_point.xy, placement_uniforms.start_point.z > 0.5);
        return resolve_start_placement(point, anchor_point, false);
    }
    if (mode < PLACEMENT_MODE_SURFACE) {
        let anchor_point = select(point, placement_uniforms.start_point.xy, placement_uniforms.start_point.z > 0.5);
        return resolve_start_placement(point, anchor_point, true);
    }
    if (mode < PLACEMENT_MODE_CONSTRUCTION_PLANE) {
        return resolve_direct_surface_placement(point);
    }
    return resolve_plane_placement(
        point,
        placement_uniforms.construction_origin.xyz,
        placement_uniforms.construction_normal.xyz,
        PLACEMENT_KIND_CONSTRUCTION_PLANE,
    );
}
