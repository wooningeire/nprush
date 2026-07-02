struct RibbonUniforms {
    view_proj: mat4x4f,
    view: mat4x4f,
    color: vec4f,
    params: vec4f,
};

struct RibbonVertex {
    position_u: vec4f,
    side: vec4f,
};

struct RibbonVertexOut {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) normal_view: vec3f,
    @location(2) shade: f32,
};

@group(0) @binding(0) var<uniform> ribbon_uniforms: RibbonUniforms;
@group(0) @binding(1) var<storage, read> ribbon_vertices: array<RibbonVertex>;

fn ribbon_column_v(column: u32, columns: u32) -> f32 {
    return -1.0 + f32(column) * 2.0 / f32(max(columns - 1u, 1u));
}

fn ribbon_position(row: u32, v: f32) -> vec3f {
    let vertex = ribbon_vertices[row];
    return vertex.position_u.xyz + vertex.side.xyz * v;
}

fn safe_normal(a: vec3f, b: vec3f, c: vec3f) -> vec3f {
    let normal = cross(b - a, c - a);
    let normal_length = length(normal);
    if (normal_length <= 0.0001) {
        return vec3f(0.0, 0.0, 1.0);
    }
    return normal / normal_length;
}

@vertex
fn ribbon_vertex(@builtin(vertex_index) vertex_index: u32) -> RibbonVertexOut {
    let rows = max(u32(ribbon_uniforms.params.x), 2u);
    let columns = max(u32(ribbon_uniforms.params.w), 2u);
    let columns_per_row = columns - 1u;
    let quad_index = vertex_index / 6u;
    let corner = vertex_index - quad_index * 6u;
    let row_index = quad_index / columns_per_row;
    let column_index = quad_index - row_index * columns_per_row;
    let next_row = select(row_index + 1u, 0u, row_index + 1u >= rows);

    let v0 = ribbon_column_v(column_index, columns);
    let v1 = ribbon_column_v(column_index + 1u, columns);
    let a = ribbon_position(row_index, v0);
    let b = ribbon_position(next_row, v0);
    let c = ribbon_position(next_row, v1);
    let d = ribbon_position(row_index, v1);

    var position = a;
    if (corner == 1u) {
        position = b;
    } else if (corner == 2u) {
        position = c;
    } else if (corner == 4u) {
        position = c;
    } else if (corner == 5u) {
        position = d;
    }

    var out: RibbonVertexOut;
    out.position = ribbon_uniforms.view_proj * vec4f(position, 1.0);
    out.color = ribbon_uniforms.color;
    out.normal_view = (ribbon_uniforms.view * vec4f(safe_normal(a, b, c), 0.0)).xyz;
    out.shade = ribbon_uniforms.params.z;
    return out;
}

fn ribbon_matcap(normal_view_raw: vec3f) -> f32 {
    let normal_length = length(normal_view_raw);
    if (normal_length <= 0.0001) {
        return 1.0;
    }

    var normal_view = normal_view_raw / normal_length;
    if (normal_view.z < 0.0) {
        normal_view = -normal_view;
    }

    let cap = normal_view.xy * 0.5 + vec2f(0.5);
    let key = (1.0 - smoothstep(0.0, 0.46, distance(cap, vec2f(0.36, 0.68)))) * 0.34;
    let fill = 0.78 + normal_view.y * 0.14;
    let rim = pow(clamp(1.0 - normal_view.z, 0.0, 1.0), 2.0) * 0.18;
    return clamp(fill + key + rim, 0.58, 1.22);
}

@fragment
fn ribbon_fragment(in: RibbonVertexOut) -> @location(0) vec4f {
    let shade = clamp(in.shade, 0.0, 1.0);
    let matcap = ribbon_matcap(in.normal_view);
    return vec4f(mix(in.color.rgb, in.color.rgb * matcap, shade), in.color.a);
}
