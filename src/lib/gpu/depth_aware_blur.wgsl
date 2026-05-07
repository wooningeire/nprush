@group(0) @binding(0) var srcColor: texture_2d<f32>;
@group(0) @binding(1) var srcDepth: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var srcNormal: texture_2d<f32>;

struct Params {
    radius: i32,
    _pad: vec3i,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(srcColor);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let px = vec2i(id.xy);
    let center_col = textureLoad(srcColor, px, 0).rgb;
    let center_depth = textureLoad(srcDepth, px, 0).r;
    let center_normal = textureLoad(srcNormal, px, 0).rgb * 2.0 - 1.0;
    
    var sum_col = vec3f(0.0);
    var sum_w = 0.0;
    
    const sigma_s: f32 = {@BLUR_SIGMA_S}; // Spatial sigma
    const sigma_c: f32 = {@BLUR_SIGMA_C}; // Color sigma
    const sigma_d: f32 = {@BLUR_SIGMA_D}; // Depth sigma
    const sigma_n: f32 = {@BLUR_SIGMA_N}; // Normal sigma (creases)
    
    let radius = params.radius;
    
    for (var dy = -radius; dy <= radius; dy++) {
        for (var dx = -radius; dx <= radius; dx++) {
            let n_px = px + vec2i(dx, dy);
            let in_bounds = n_px.x >= 0 && n_px.x < i32(dims.x) && n_px.y >= 0 && n_px.y < i32(dims.y);
            let s_px = clamp(n_px, vec2i(0), vec2i(dims) - 1);
            
            let n_col = textureLoad(srcColor, s_px, 0).rgb;
            let n_depth = textureLoad(srcDepth, s_px, 0).r;
            let n_normal = textureLoad(srcNormal, s_px, 0).rgb * 2.0 - 1.0;
            
            let d2 = f32(dx*dx + dy*dy);
            let dc2 = dot(n_col - center_col, n_col - center_col);
            let dd2 = (n_depth - center_depth) * (n_depth - center_depth);
            
            // Normal weight: 1 - dot(n1, n2) is 0 if same, 2 if opposite.
            // Using exp(-dist / sigma)
            let dn2 = 1.0 - dot(n_normal, center_normal);
            
            let w_raw = exp(-d2 / (2.0 * sigma_s * sigma_s)) * 
                    exp(-dc2 / (2.0 * sigma_c * sigma_c)) *
                    exp(-dd2 / (2.0 * sigma_d * sigma_d)) *
                    exp(-dn2 / (2.0 * sigma_n * sigma_n));
            let w = select(0.0, w_raw, in_bounds);
                     
            sum_col += n_col * w;
            sum_w += w;
        }
    }
    let result = sum_col / max(sum_w, 1e-5);
    textureStore(dst, id.xy, vec4f(result, 1.0));
}
