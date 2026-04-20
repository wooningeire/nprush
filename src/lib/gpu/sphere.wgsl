struct Uniforms {
    viewProjMat: mat4x4f,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) viewDepth: f32,
}

struct FragOutput {
    @location(0) color: vec4f,
    @location(1) depth: vec4f,
}

@vertex
fn vert(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.viewProjMat * vec4f(in.position, 1);
    out.normal = in.normal;
    // Store linear depth (NDC z mapped to 0..1)
    out.viewDepth = out.position.z / out.position.w * 0.5 + 0.5;
    return out;
}

@fragment
fn frag(in: VertexOutput) -> FragOutput {
    let color = (normalize(in.normal) + 1) * 0.5;
    var out: FragOutput;
    out.color = vec4f(color, 1);
    out.depth = vec4f(in.viewDepth, in.viewDepth, in.viewDepth, 1.0);
    return out;
}
