@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;

struct Params {
    direction: vec2i,
    radius: i32,
    sigma: f32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(src);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    var sum = vec3f(0.0);
    var weight_sum = 0.0;

    for (var i = -params.radius; i <= params.radius; i++) {
        let coord = vec2i(id.xy) + params.direction * i;
        let clamped = clamp(coord, vec2i(0), vec2i(dims) - 1);
        let col = textureLoad(src, clamped, 0).rgb;
        let w = exp(-f32(i * i) / (2.0 * params.sigma * params.sigma));
        sum += col * w;
        weight_sum += w;
    }

    textureStore(dst, id.xy, vec4f(sum / weight_sum, 1.0));
}
