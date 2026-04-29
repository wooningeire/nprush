// Resolves the path trace accumulation buffer into an rgba8unorm texture.
// Divides by sample count and applies a simple Reinhard tone map.

@group(0) @binding(0) var<storage, read> accum:   array<f32>;
@group(0) @binding(1) var               out_tex:  texture_storage_2d<rgba8unorm, write>;

struct ResolveUniforms {
    out_w: u32,
    out_h: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(2) var<uniform> ru: ResolveUniforms;

fn reinhard(c: vec3f) -> vec3f {
    return c / (c + vec3f(1.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= ru.out_w || gid.y >= ru.out_h) { return; }
    let pixel_idx = gid.y * ru.out_w + gid.x;
    let base = pixel_idx * 4u;
    let r = accum[base + 0u];
    let g = accum[base + 1u];
    let b = accum[base + 2u];
    let n = max(accum[base + 3u], 1.0);
    let color = reinhard(vec3f(r, g, b) / n);
    textureStore(out_tex, vec2i(gid.xy), vec4f(color, 1.0));
}
