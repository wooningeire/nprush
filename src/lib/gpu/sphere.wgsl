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
}

@vertex
fn vert(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.viewProjMat * vec4f(in.position, 1);
    out.normal = in.normal;
    return out;
}

@fragment
fn frag(in: VertexOutput) -> @location(0) vec4f {
    let color = (normalize(in.normal) + 1) * 0.5;
    return vec4f(color, 1);
}
