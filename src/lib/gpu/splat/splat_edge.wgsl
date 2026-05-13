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

    // Depth Laplacian — fires at true discontinuities (step H → value ≈ H).
    // Zero on linear gradients; suppresses smooth depth variation unlike Sobel.
    let d_c = sampleDepth(p,                dims);
    let d_l = sampleDepth(p + vec2i(-1, 0), dims);
    let d_r = sampleDepth(p + vec2i( 1, 0), dims);
    let d_u = sampleDepth(p + vec2i( 0,-1), dims);
    let d_d = sampleDepth(p + vec2i( 0, 1), dims);
    let depth_edge = abs(d_l + d_r + d_u + d_d - 4.0 * d_c);

    // Normal Laplacian — fires at creases (|N_B - N_A|: ~1.41 at 90°, ~0.52 at 30°).
    // Zero on linearly-varying normals, so smooth curvature and glancing angles don't trigger.
    let n_c = sampleNormal(p,                dims);
    let n_l = sampleNormal(p + vec2i(-1, 0), dims);
    let n_r = sampleNormal(p + vec2i( 1, 0), dims);
    let n_u = sampleNormal(p + vec2i( 0,-1), dims);
    let n_d = sampleNormal(p + vec2i( 0, 1), dims);
    let normal_edge = length(n_l + n_r + n_u + n_d - 4.0 * n_c);

    let depth_thresh  = step({@SPLAT_EDGE_THRESHOLD}, depth_edge);
    let normal_thresh = step({@SPLAT_EDGE_NORMAL_THRESHOLD}, normal_edge);

    let edge = max(depth_thresh, normal_thresh);

    textureStore(edgeTex, p, vec4f(edge, edge, edge, 1.0));
}
