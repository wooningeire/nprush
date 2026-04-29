struct Uniforms {
    viewProjMat: mat4x4f,
    viewMat: mat4x4f,
    shadingMode: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var matcapTex: texture_2d<f32>;
@group(0) @binding(2) var matcapSampler: sampler;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) viewDepth: f32,
    @location(2) viewNormal: vec3f,
}

struct FragOutput {
    @location(0) color: vec4f,
    @location(1) depth: vec4f,
}

const DEPTH_VIZ_FAR: f32 = 10.0;

@vertex
fn vert(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.viewProjMat * vec4f(in.position, 1);
    out.normal = in.normal;
    out.viewNormal = (uniforms.viewMat * vec4f(in.normal, 0)).xyz;
    out.viewDepth = clamp(out.position.w / DEPTH_VIZ_FAR, 0.0, 1.0);
    return out;
}

@fragment
fn frag(in: VertexOutput) -> FragOutput {
    var out: FragOutput;
    
    let normals_color = (normalize(in.normal) + 1.0) * 0.5;
    let vn = normalize(in.viewNormal);
    let uv = vn.xy * 0.5 + 0.5;
    let matcap_color = textureSample(matcapTex, matcapSampler, vec2f(uv.x, 1.0 - uv.y)).rgb;
    out.color = vec4f(select(matcap_color, normals_color, uniforms.shadingMode < 0.5), 1.0);
    
    out.depth = vec4f(in.viewDepth, in.viewDepth, in.viewDepth, 1.0);
    return out;
}
