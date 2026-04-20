// Sobel edge detection on a depth texture
// Reads from a depth texture and writes edge intensity to a storage texture

@group(0) @binding(0) var depthTex: texture_2d<f32>;
@group(0) @binding(1) var edgeTex: texture_storage_2d<rgba8unorm, write>;

fn sampleDepth(coord: vec2i, dims: vec2u) -> f32 {
    let c = clamp(coord, vec2i(0), vec2i(dims) - vec2i(1));
    return textureLoad(depthTex, c, 0).r;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let dims = textureDimensions(depthTex);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }
    
    let p = vec2i(global_id.xy);
    
    // Sobel kernel
    let tl = sampleDepth(p + vec2i(-1, -1), dims);
    let tc = sampleDepth(p + vec2i( 0, -1), dims);
    let tr = sampleDepth(p + vec2i( 1, -1), dims);
    let ml = sampleDepth(p + vec2i(-1,  0), dims);
    let mr = sampleDepth(p + vec2i( 1,  0), dims);
    let bl = sampleDepth(p + vec2i(-1,  1), dims);
    let bc = sampleDepth(p + vec2i( 0,  1), dims);
    let br = sampleDepth(p + vec2i( 1,  1), dims);
    
    let gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
    let gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
    
    let edge = sqrt(gx * gx + gy * gy);
    let edgeClamped = clamp(edge * 8.0, 0.0, 1.0);
    
    textureStore(edgeTex, p, vec4f(edgeClamped, edgeClamped, edgeClamped, 1.0));
}
