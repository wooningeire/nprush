struct TriangleUniforms {
    view_proj: mat4x4f,
    view: mat4x4f,
};

struct TriangleVertexOut {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
    @location(1) normal_view: vec3f,
    @location(2) shade: f32,
};

@group(0) @binding(0) var<uniform> triangle_uniforms: TriangleUniforms;

@vertex
fn triangle_vertex(
    @location(0) position: vec3f,
    @location(1) color: vec4f,
    @location(2) normal: vec3f,
    @location(3) shade: f32,
) -> TriangleVertexOut {
    var out: TriangleVertexOut;
    out.position = triangle_uniforms.view_proj * vec4f(position, 1.0);
    out.color = color;
    out.normal_view = (triangle_uniforms.view * vec4f(normal, 0.0)).xyz;
    out.shade = shade;
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
fn triangle_fragment(in: TriangleVertexOut) -> @location(0) vec4f {
    let shade = clamp(in.shade, 0.0, 1.0);
    let matcap = ribbon_matcap(in.normal_view);
    return vec4f(mix(in.color.rgb, in.color.rgb * matcap, shade), in.color.a);
}
