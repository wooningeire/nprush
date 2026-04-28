@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;

struct Params {
    direction: vec2i,
    radius: i32,
    flags: u32, // bit 0: input is sRGB, bit 1: output is sRGB
    sigma: f32,
    _pad: vec3f,
}
@group(0) @binding(2) var<uniform> params: Params;

fn srgb_to_linear(c: vec3f) -> vec3f {
    return select(c / 12.92, pow((c + 0.055) / 1.055, vec3f(2.4)), c > vec3f(0.04045));
}

fn linear_to_srgb(c: vec3f) -> vec3f {
    let safe_c = max(c, vec3f(0.0));
    return select(safe_c * 12.92, 1.055 * pow(safe_c, vec3f(1.0 / 2.4)) - 0.055, safe_c > vec3f(0.0031308));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(src);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    var sum = vec3f(0.0);
    var weight_sum = 0.0;

    let input_is_srgb = (params.flags & 1u) != 0u;
    let output_is_srgb = (params.flags & 2u) != 0u;

    for (var i = -params.radius; i <= params.radius; i++) {
        let coord = vec2i(id.xy) + params.direction * i;
        let clamped = clamp(coord, vec2i(0), vec2i(dims) - 1);
        var col = textureLoad(src, clamped, 0).rgb;
        
        if (input_is_srgb) {
            col = srgb_to_linear(col);
        }
        
        let w = exp(-f32(i * i) / (2.0 * params.sigma * params.sigma));
        sum += col * w;
        weight_sum += w;
    }

    var result = sum / weight_sum;
    if (output_is_srgb) {
        result = linear_to_srgb(result);
    }

    textureStore(dst, id.xy, vec4f(result, 1.0));
}
