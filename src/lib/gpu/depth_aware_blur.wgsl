@group(0) @binding(0) var srcColor: texture_2d<f32>;
@group(0) @binding(1) var srcDepth: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba8unorm, write>;

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
    
    var sum_col = vec3f(0.0);
    var sum_w = 0.0;
    
    let sigma_s = 10.0; // Spatial sigma
    let sigma_c = 0.5; // Color sigma
    let sigma_d = 0.5; // Depth sigma
    
    let radius = params.radius;
    
    for (var dy = -radius; dy <= radius; dy++) {
        for (var dx = -radius; dx <= radius; dx++) {
            let n_px = px + vec2i(dx, dy);
            if (n_px.x < 0 || n_px.x >= i32(dims.x) || n_px.y < 0 || n_px.y >= i32(dims.y)) { continue; }
            
            let n_col = textureLoad(srcColor, n_px, 0).rgb;
            let n_depth = textureLoad(srcDepth, n_px, 0).r;
            
            let d2 = f32(dx*dx + dy*dy);
            let dc2 = dot(n_col - center_col, n_col - center_col);
            let dd2 = (n_depth - center_depth) * (n_depth - center_depth);
            
            let w = exp(-d2 / (2.0 * sigma_s * sigma_s)) * 
                    exp(-dc2 / (2.0 * sigma_c * sigma_c)) *
                    exp(-dd2 / (2.0 * sigma_d * sigma_d));
                     
            sum_col += n_col * w;
            sum_w += w;
        }
    }
    let result = sum_col / max(sum_w, 1e-5);
    textureStore(dst, id.xy, vec4f(result, 1.0));
}
