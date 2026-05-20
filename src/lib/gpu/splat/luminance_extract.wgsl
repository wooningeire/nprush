@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(src);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let rgb = textureLoad(src, gid.xy, 0).rgb;
    let luma = dot(rgb, vec3f(0.2126, 0.7152, 0.0722));
    textureStore(dst, gid.xy, vec4f(luma, luma, luma, 1.0));
}
