struct ChartParams {
    view_proj: mat4x4f,
    source_view_proj_inv: mat4x4f,
    camera_center_projection: vec4f,
    source_forward_coverage: vec4f,
    dims_stride: vec4f,
    normal_settings: vec4f,
    line_color: vec4f,
    fill_color: vec4f,
    normal_color: vec4f,
};

struct ChartVertexOut {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
};

struct WorldPoint {
    world: vec3f,
    valid: bool,
};

@group(0) @binding(0) var<storage, read> fields: array<vec2f>;
@group(0) @binding(1) var<uniform> params: ChartParams;

fn chart_width() -> u32 {
    return u32(params.dims_stride.x);
}

fn chart_height() -> u32 {
    return u32(params.dims_stride.y);
}

fn line_stride() -> u32 {
    return max(1u, u32(params.dims_stride.z));
}

fn field_stride() -> u32 {
    return max(1u, u32(params.dims_stride.w));
}

fn normal_length_scale() -> f32 {
    return params.normal_settings.x;
}

fn coverage_epsilon() -> f32 {
    return params.source_forward_coverage.w;
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

fn unproject(ndc: vec3f) -> vec3f {
    let p = params.source_view_proj_inv * vec4f(ndc, 1.0);
    return p.xyz / p.w;
}

fn chart_world_at(x: u32, y: u32) -> WorldPoint {
    let uv = grid_uv(x, y);
    let near_point = unproject(vec3f(uv, -1.0));
    let far_point = unproject(vec3f(uv, 1.0));
    let ray_direction = normalize(far_point - near_point);
    let depth = fields[grid_index(x, y)].x;
    let camera_center = params.camera_center_projection.xyz;
    let projection_mode = params.camera_center_projection.w;

    if (projection_mode > 0.5) {
        return WorldPoint(camera_center + ray_direction * depth, true);
    }

    let denominator = dot(ray_direction, params.source_forward_coverage.xyz);
    if (abs(denominator) <= 0.000001) {
        return WorldPoint(camera_center, false);
    }

    let ray_distance = depth / denominator;
    return WorldPoint(camera_center + ray_direction * ray_distance, ray_distance > 0.000001);
}

fn chart_normal_endpoint(x: u32, y: u32, world: vec3f) -> WorldPoint {
    let x0 = select(x - 1u, 0u, x == 0u);
    let x1 = min(chart_width() - 1u, x + 1u);
    let y0 = select(y - 1u, 0u, y == 0u);
    let y1 = min(chart_height() - 1u, y + 1u);
    if (x0 == x1 || y0 == y1) {
        return WorldPoint(world, false);
    }

    let left = chart_world_at(x0, y);
    let right = chart_world_at(x1, y);
    let bottom = chart_world_at(x, y0);
    let top = chart_world_at(x, y1);
    if (!left.valid || !right.valid || !bottom.valid || !top.valid) {
        return WorldPoint(world, false);
    }

    let raw_normal = cross(right.world - left.world, top.world - bottom.world);
    let raw_length = length(raw_normal);
    if (raw_length <= 0.000001) {
        return WorldPoint(world, false);
    }

    var normal = raw_normal / raw_length;
    let toward_camera = params.camera_center_projection.xyz - world;
    if (dot(normal, toward_camera) < 0.0) {
        normal = -normal;
    }

    let normal_length = clamp(
        distance(params.camera_center_projection.xyz, world) * normal_length_scale(),
        0.035,
        0.18,
    );
    return WorldPoint(world + normal * normal_length, true);
}

fn triangle_corner(cell_x: u32, cell_y: u32, vertex_in_cell: u32) -> vec2u {
    switch vertex_in_cell {
        case 0u: { return vec2u(cell_x, cell_y); }
        case 1u: { return vec2u(cell_x + 1u, cell_y); }
        case 2u: { return vec2u(cell_x + 1u, cell_y + 1u); }
        case 3u: { return vec2u(cell_x, cell_y); }
        case 4u: { return vec2u(cell_x + 1u, cell_y + 1u); }
        default: { return vec2u(cell_x, cell_y + 1u); }
    }
}

fn triangle_covered(cell_x: u32, cell_y: u32, vertex_in_cell: u32) -> bool {
    let i00 = grid_index(cell_x, cell_y);
    let i10 = grid_index(cell_x + 1u, cell_y);
    let i01 = grid_index(cell_x, cell_y + 1u);
    let i11 = grid_index(cell_x + 1u, cell_y + 1u);
    if (vertex_in_cell < 3u) {
        return covered(i00) && covered(i10) && covered(i11);
    }
    return covered(i00) && covered(i11) && covered(i01);
}

@vertex
fn chart_fill_vertex(@builtin(vertex_index) vertex_index: u32) -> ChartVertexOut {
    let cells_wide = chart_width() - 1u;
    let cell_index = vertex_index / 6u;
    let vertex_in_cell = vertex_index % 6u;
    let cell_x = cell_index % cells_wide;
    let cell_y = cell_index / cells_wide;
    let corner = triangle_corner(cell_x, cell_y, vertex_in_cell);
    let world = chart_world_at(corner.x, corner.y);
    let valid = world.valid && triangle_covered(cell_x, cell_y, vertex_in_cell);

    var out: ChartVertexOut;
    out.position = params.view_proj * vec4f(world.world, 1.0);
    out.color = select(vec4f(0.0), params.fill_color, valid);
    return out;
}

@fragment
fn chart_fragment(in: ChartVertexOut) -> @location(0) vec4f {
    return in.color;
}

fn horizontal_line_count() -> u32 {
    let row_count = ((chart_height() - 1u) / line_stride()) + 1u;
    return row_count * (chart_width() - 1u);
}

fn line_endpoint(line_index: u32, endpoint: u32) -> vec2u {
    let horizontal_count = horizontal_line_count();
    if (line_index < horizontal_count) {
        let x = line_index % (chart_width() - 1u);
        let row = line_index / (chart_width() - 1u);
        return vec2u(x + endpoint, row * line_stride());
    }

    let local_line = line_index - horizontal_count;
    let y = local_line % (chart_height() - 1u);
    let column = local_line / (chart_height() - 1u);
    return vec2u(column * line_stride(), y + endpoint);
}

fn line_is_covered(line_index: u32) -> bool {
    let a = line_endpoint(line_index, 0u);
    let b = line_endpoint(line_index, 1u);
    return covered(grid_index(a.x, a.y)) && covered(grid_index(b.x, b.y));
}

@vertex
fn chart_wire_vertex(@builtin(vertex_index) vertex_index: u32) -> ChartVertexOut {
    let line_index = vertex_index / 2u;
    let endpoint = vertex_index % 2u;
    let point = line_endpoint(line_index, endpoint);
    let world = chart_world_at(point.x, point.y);
    let valid = world.valid && line_is_covered(line_index);

    var out: ChartVertexOut;
    out.position = params.view_proj * vec4f(world.world, 1.0);
    out.color = select(vec4f(0.0), params.line_color, valid);
    return out;
}

fn field_axis_count(size: u32) -> u32 {
    return ((size - 1u) / field_stride()) + 1u;
}

fn field_point(line_index: u32) -> vec2u {
    let columns = field_axis_count(chart_width());
    let x = min(chart_width() - 1u, (line_index % columns) * field_stride());
    let y = min(chart_height() - 1u, (line_index / columns) * field_stride());
    return vec2u(x, y);
}

@vertex
fn chart_field_vertex(@builtin(vertex_index) vertex_index: u32) -> ChartVertexOut {
    let line_index = vertex_index / 2u;
    let endpoint = vertex_index % 2u;
    let point = field_point(line_index);
    let base = chart_world_at(point.x, point.y);
    let normal_endpoint = chart_normal_endpoint(point.x, point.y, base.world);
    var world = base.world;
    if (endpoint == 1u) {
        world = normal_endpoint.world;
    }
    let valid = base.valid
        && normal_endpoint.valid
        && covered(grid_index(point.x, point.y));

    var out: ChartVertexOut;
    out.position = params.view_proj * vec4f(world, 1.0);
    out.color = select(vec4f(0.0), params.normal_color, valid);
    return out;
}