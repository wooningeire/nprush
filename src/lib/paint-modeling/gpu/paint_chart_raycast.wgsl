struct RaycastParams {
    view_proj_inv: mat4x4f,
    source_view_proj_inv: mat4x4f,
    view_camera_projection: vec4f,
    source_camera_coverage: vec4f,
    source_forward_padding: vec4f,
    dims_chart: vec4u,
};

struct Ray {
    origin: vec3f,
    direction: vec3f,
};

struct WorldPoint {
    world: vec3f,
    valid: bool,
};

struct TriangleHit {
    t: f32,
    u: f32,
    v: f32,
    hit: bool,
};

struct RaycastResult {
    depth_bits: atomic<u32>,
    chart_index: u32,
    uv: vec2f,
    world: vec4f,
};

struct TriangleCorners {
    a: vec2u,
    b: vec2u,
    c: vec2u,
};

@group(0) @binding(0) var<storage, read> fields: array<vec2f>;
@group(0) @binding(1) var<storage, read> points: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> results: array<RaycastResult>;
@group(0) @binding(3) var<uniform> params: RaycastParams;

fn chart_width() -> u32 {
    return params.dims_chart.x;
}

fn chart_height() -> u32 {
    return params.dims_chart.y;
}

fn point_count() -> u32 {
    return params.dims_chart.z;
}

fn chart_index() -> u32 {
    return params.dims_chart.w;
}

fn coverage_epsilon() -> f32 {
    return params.source_camera_coverage.w;
}

fn grid_uv(x: u32, y: u32) -> vec2f {
    let width = chart_width();
    let height = chart_height();
    let u = select(0.0, f32(x) / f32(width - 1u) * 2.0 - 1.0, width > 1u);
    let v = select(0.0, f32(y) / f32(height - 1u) * 2.0 - 1.0, height > 1u);
    return vec2f(u, v);
}

fn grid_index(x: u32, y: u32) -> u32 {
    return y * chart_width() + x;
}

fn covered(index: u32) -> bool {
    return fields[index].y > coverage_epsilon();
}

fn unproject(matrix: mat4x4f, ndc: vec3f) -> vec3f {
    let p = matrix * vec4f(ndc, 1.0);
    return p.xyz / p.w;
}

fn point_ray(point_index: u32) -> Ray {
    let point = points[point_index];
    let near_point = unproject(params.view_proj_inv, vec3f(point, -1.0));
    let far_point = unproject(params.view_proj_inv, vec3f(point, 1.0));
    return Ray(
        params.view_camera_projection.xyz,
        normalize(far_point - near_point),
    );
}

fn chart_world_at(x: u32, y: u32) -> WorldPoint {
    let uv = grid_uv(x, y);
    let near_point = unproject(params.source_view_proj_inv, vec3f(uv, -1.0));
    let far_point = unproject(params.source_view_proj_inv, vec3f(uv, 1.0));
    let ray_direction = normalize(far_point - near_point);
    let depth = fields[grid_index(x, y)].x;
    let source_camera = params.source_camera_coverage.xyz;
    let projection_mode = params.view_camera_projection.w;

    if (projection_mode > 0.5) {
        return WorldPoint(source_camera + ray_direction * depth, true);
    }

    let denominator = dot(ray_direction, params.source_forward_padding.xyz);
    if (abs(denominator) <= 0.000001) {
        return WorldPoint(source_camera, false);
    }

    let ray_distance = depth / denominator;
    return WorldPoint(source_camera + ray_direction * ray_distance, ray_distance > 0.000001);
}

fn triangle_corners(cell_x: u32, cell_y: u32, triangle_in_cell: u32) -> TriangleCorners {
    if (triangle_in_cell == 0u) {
        return TriangleCorners(
            vec2u(cell_x, cell_y),
            vec2u(cell_x + 1u, cell_y),
            vec2u(cell_x + 1u, cell_y + 1u),
        );
    }
    return TriangleCorners(
        vec2u(cell_x, cell_y),
        vec2u(cell_x + 1u, cell_y + 1u),
        vec2u(cell_x, cell_y + 1u),
    );
}

fn triangle_is_covered(corners: TriangleCorners) -> bool {
    return covered(grid_index(corners.a.x, corners.a.y))
        && covered(grid_index(corners.b.x, corners.b.y))
        && covered(grid_index(corners.c.x, corners.c.y));
}

fn intersect_ray_triangle(ray: Ray, p0: vec3f, p1: vec3f, p2: vec3f) -> TriangleHit {
    let epsilon = 0.0000001;
    let edge1 = p1 - p0;
    let edge2 = p2 - p0;
    let h = cross(ray.direction, edge2);
    let a = dot(edge1, h);
    if (abs(a) < epsilon) {
        return TriangleHit(0.0, 0.0, 0.0, false);
    }

    let f = 1.0 / a;
    let s = ray.origin - p0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) {
        return TriangleHit(0.0, 0.0, 0.0, false);
    }

    let q = cross(s, edge1);
    let v = f * dot(ray.direction, q);
    if (v < 0.0 || u + v > 1.0) {
        return TriangleHit(0.0, 0.0, 0.0, false);
    }

    let t = f * dot(edge2, q);
    return TriangleHit(t, u, v, t > epsilon);
}

@compute @workgroup_size(64)
fn chart_raycast(@builtin(global_invocation_id) id: vec3u) {
    let cells_wide = chart_width() - 1u;
    let cells_high = chart_height() - 1u;
    let triangle_count = cells_wide * cells_high * 2u;
    let task_index = id.x;
    if (triangle_count == 0u || task_index >= triangle_count * point_count()) {
        return;
    }

    let point_index = task_index / triangle_count;
    let triangle_index = task_index % triangle_count;
    let cell_index = triangle_index / 2u;
    let triangle_in_cell = triangle_index % 2u;
    let cell_x = cell_index % cells_wide;
    let cell_y = cell_index / cells_wide;
    let corners = triangle_corners(cell_x, cell_y, triangle_in_cell);
    if (!triangle_is_covered(corners)) {
        return;
    }

    let p0 = chart_world_at(corners.a.x, corners.a.y);
    let p1 = chart_world_at(corners.b.x, corners.b.y);
    let p2 = chart_world_at(corners.c.x, corners.c.y);
    if (!p0.valid || !p1.valid || !p2.valid) {
        return;
    }

    let ray = point_ray(point_index);
    let hit = intersect_ray_triangle(ray, p0.world, p1.world, p2.world);
    if (!hit.hit) {
        return;
    }

    let depth_bits = bitcast<u32>(hit.t);
    let old_depth_bits = atomicMin(&results[point_index].depth_bits, depth_bits);
    if (depth_bits >= old_depth_bits) {
        return;
    }

    let w0 = 1.0 - hit.u - hit.v;
    let uv0 = grid_uv(corners.a.x, corners.a.y);
    let uv1 = grid_uv(corners.b.x, corners.b.y);
    let uv2 = grid_uv(corners.c.x, corners.c.y);
    results[point_index].chart_index = chart_index();
    results[point_index].uv = uv0 * w0 + uv1 * hit.u + uv2 * hit.v;
    results[point_index].world = vec4f(ray.origin + ray.direction * hit.t, 1.0);
}