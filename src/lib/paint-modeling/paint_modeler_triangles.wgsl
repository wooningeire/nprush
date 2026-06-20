struct TriangleUniforms {
    view_proj: mat4x4f,
};

struct TriangleVertexOut {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
};

@group(0) @binding(0) var<uniform> triangle_uniforms: TriangleUniforms;

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