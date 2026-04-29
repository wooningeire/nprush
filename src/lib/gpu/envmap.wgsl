// Fullscreen equirectangular environment map background.
// Reconstructs a world-space ray direction for each fragment using the
// inverse view-projection matrix, then samples the panorama texture.

struct Uniforms {
    viewProjMat:    mat4x4f,  // offset   0, size 64
    viewMat:        mat4x4f,  // offset  64, size 64
    shadingMode:    f32,      // offset 128, size  4
    _pad0:          f32,      // offset 132
    _pad1:          f32,      // offset 136
    _pad2:          f32,      // offset 140
    invViewProjMat: mat4x4f,  // offset 144, size 64
    // total: 208 bytes
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var envTex:     texture_2d<f32>;
@group(0) @binding(2) var envSampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) ndc: vec2f,
}

@vertex
fn vert(@builtin(vertex_index) vi: u32) -> VsOut {
    // Two triangles covering the full screen.
    let xs = array<f32, 6>(-1.0,  1.0, -1.0, -1.0,  1.0,  1.0);
    let ys = array<f32, 6>(-1.0, -1.0,  1.0,  1.0, -1.0,  1.0);
    let x = xs[vi];
    let y = ys[vi];
    var out: VsOut;
    // Write at max depth so the mesh always renders on top.
    out.pos = vec4f(x, y, 1.0, 1.0);
    out.ndc = vec2f(x, y);
    return out;
}

const PI: f32 = 3.14159265358979;
fn reinhard(c: vec3f) -> vec3f { return c / (c + 1.0); }

@fragment
fn frag(in: VsOut) -> @location(0) vec4f {
    // Reconstruct world-space ray direction from NDC position.
    // Unproject two points at different depths and take the direction.
    let near_h = uniforms.invViewProjMat * vec4f(in.ndc, 0.0, 1.0);
    let far_h  = uniforms.invViewProjMat * vec4f(in.ndc, 1.0, 1.0);
    let near_w = near_h.xyz / near_h.w;
    let far_w  = far_h.xyz  / far_h.w;
    let dir = normalize(far_w - near_w);

    // Equirectangular mapping: longitude [-π, π] → u [0,1], latitude [-π/2, π/2] → v [0,1]
    let lon = atan2(dir.y, dir.x);          // [-π, π]
    let lat = asin(clamp(dir.z, -1.0, 1.0)); // [-π/2, π/2]
    let u = lon / (2.0 * PI) + 0.5;
    let v = 0.5 - lat / PI;

    let color = textureSample(envTex, envSampler, vec2f(u, v)).rgb;
    return vec4f(reinhard(color * 4.0), 1.0);
}
