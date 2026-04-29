struct Uniforms {
    viewProjMat: mat4x4f,
    viewMat: mat4x4f,
    shadingMode: f32,
}
fn reinhard(c: vec3f) -> vec3f { return c / (c + 1.0); }

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var matcapTex: texture_2d<f32>;
@group(0) @binding(2) var matcapSampler: sampler;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) color: vec4f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) viewDepth: f32,
    @location(2) viewNormal: vec3f,
    @location(3) color: vec4f,
}

struct FragOutput {
    @location(0) color: vec4f,
    @location(1) depth: vec4f,
}

// Near/far for depth remapping. Objects closer than DEPTH_NEAR get full
// precision; the reciprocal mapping 1 - DEPTH_NEAR/depth compresses the
// far range and expands the near range, giving much better 8-bit precision
// for nearby geometry.
const DEPTH_NEAR: f32 = 0.1;

@vertex
fn vert(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.viewProjMat * vec4f(in.position, 1);
    out.normal = in.normal;
    out.viewNormal = (uniforms.viewMat * vec4f(in.normal, 0)).xyz;
    // Reciprocal depth: maps [DEPTH_NEAR, inf) -> [0, 1), giving high
    // precision for nearby objects and gracefully compressing far ones.
    let linear_depth = max(out.position.w, DEPTH_NEAR);
    out.viewDepth = clamp(1.0 - DEPTH_NEAR / linear_depth, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn frag(in: VertexOutput) -> FragOutput {
    var out: FragOutput;

    let normals_color = (normalize(in.normal) + 1.0) * 0.5;
    let vn = normalize(in.viewNormal);
    let uv = vn.xy * 0.5 + 0.5;
    let matcap_color = textureSample(matcapTex, matcapSampler, vec2f(uv.x, 1.0 - uv.y)).rgb;

    // Tint the matcap by the material base color (linear multiply).
    let tinted_matcap = matcap_color * in.color.rgb;

    let final_color = select(tinted_matcap, normals_color, uniforms.shadingMode < 0.5);
    out.color = vec4f(reinhard(final_color * 4.0), 1.0);
    out.depth = vec4f(in.viewDepth, in.viewDepth, in.viewDepth, 1.0);
    return out;
}
