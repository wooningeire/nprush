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

// Range used to remap linear view-space depth into [0, 1] for the depth target.
// Tuned so that scenes around the default camera radius (~3) map to the lower half
// of the range, leaving plenty of contrast against the cleared (far) background.
const DEPTH_VIZ_FAR: f32 = 10.0;

@vertex
fn vert(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = uniforms.viewProjMat * vec4f(in.position, 1);
    out.normal = in.normal;
    // Store linear view-space depth (clip-w == view-space distance for a perspective
    // projection) normalized into a useful display range. This gives Sobel something
    // meaningful at the silhouette and keeps depth values away from both 0 and 1.
    out.viewDepth = clamp(out.position.w / DEPTH_VIZ_FAR, 0.0, 1.0);
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
