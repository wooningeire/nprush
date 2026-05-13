@group(0) @binding(0) var depthTex: texture_2d<f32>;
@group(0) @binding(1) var edgeTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var normalTex: texture_2d<f32>;

fn sampleDepth(coord: vec2i, dims: vec2u) -> f32 {
    let c = clamp(coord, vec2i(0), vec2i(dims) - vec2i(1));
    return textureLoad(depthTex, c, 0).r;
}

fn sampleNormal(coord: vec2i, dims: vec2u) -> vec3f {
    let c = clamp(coord, vec2i(0), vec2i(dims) - vec2i(1));
    return textureLoad(normalTex, c, 0).rgb * 2.0 - 1.0;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let dims = textureDimensions(depthTex);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    let p = vec2i(global_id.xy);

    // Depth Sobel — silhouettes and large depth discontinuities
    let d_tl = sampleDepth(p + vec2i(-1, -1), dims);
    let d_tc = sampleDepth(p + vec2i( 0, -1), dims);
    let d_tr = sampleDepth(p + vec2i( 1, -1), dims);
    let d_ml = sampleDepth(p + vec2i(-1,  0), dims);
    let d_mr = sampleDepth(p + vec2i( 1,  0), dims);
    let d_bl = sampleDepth(p + vec2i(-1,  1), dims);
    let d_bc = sampleDepth(p + vec2i( 0,  1), dims);
    let d_br = sampleDepth(p + vec2i( 1,  1), dims);

    let d_gx = -d_tl - 2.0*d_ml - d_bl + d_tr + 2.0*d_mr + d_br;
    let d_gy = -d_tl - 2.0*d_tc - d_tr + d_bl + 2.0*d_bc + d_br;
    let depth_edge = sqrt(d_gx*d_gx + d_gy*d_gy);

    // Normal Sobel — surface creases and ridges
    let n_tl = sampleNormal(p + vec2i(-1, -1), dims);
    let n_tc = sampleNormal(p + vec2i( 0, -1), dims);
    let n_tr = sampleNormal(p + vec2i( 1, -1), dims);
    let n_ml = sampleNormal(p + vec2i(-1,  0), dims);
    let n_mr = sampleNormal(p + vec2i( 1,  0), dims);
    let n_bl = sampleNormal(p + vec2i(-1,  1), dims);
    let n_bc = sampleNormal(p + vec2i( 0,  1), dims);
    let n_br = sampleNormal(p + vec2i( 1,  1), dims);

    let n_gx = -n_tl - 2.0*n_ml - n_bl + n_tr + 2.0*n_mr + n_br;
    let n_gy = -n_tl - 2.0*n_tc - n_tr + n_bl + 2.0*n_bc + n_br;
    let normal_edge = sqrt(dot(n_gx, n_gx) + dot(n_gy, n_gy));

    let depth_thresh  = smoothstep({@SPLAT_EDGE_THRESHOLD_MIN}, {@SPLAT_EDGE_THRESHOLD_MAX}, depth_edge);
    // Normal edges use a separate (lower) threshold since normal gradients are smaller in magnitude
    let normal_thresh = smoothstep({@SPLAT_EDGE_NORMAL_THRESHOLD_MIN}, {@SPLAT_EDGE_NORMAL_THRESHOLD_MAX}, normal_edge);

    let edge = max(depth_thresh, normal_thresh);

    textureStore(edgeTex, p, vec4f(edge, edge, edge, 1.0));
}
