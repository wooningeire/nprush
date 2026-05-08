// Bitonic merge-sort of splat indices by camera depth (far-to-near for
// correct back-to-front alpha blending).
//
// The sort operates on a buffer of (depth, index) pairs.  A first pass
// computes depth from the VP matrix, then log2(N) rounds of bitonic
// merge are dispatched.

struct Splat {
    pos_sx: vec4f,
    color: vec4f,
    quat: vec4f,
    sy_shape: vec4f,
}

struct SplatArray {
    splats: array<Splat, {@NUM_SPLATS}u>,
}

struct SortUniforms {
    vp: mat4x4f,
    block_k: u32,      // outer loop (block size = 2^block_k)
    sub_k: u32,         // inner loop (stride = 2^sub_k)
    _pad: vec2u,
}

@group(0) @binding(0) var<storage, read> splats: SplatArray;
@group(0) @binding(1) var<storage, read_write> sort_keys: array<f32, {@NUM_SPLATS}u>;
@group(0) @binding(2) var<storage, read_write> sort_indices: array<u32, {@NUM_SPLATS}u>;
@group(0) @binding(3) var<uniform> sort_uniforms: SortUniforms;

// ----- Pass 0: compute depth keys and init indices -----
@compute @workgroup_size(256)
fn init_keys(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    let n = {@SORT_N}u;
    if (i >= n) { return; }

    if (i < {@NUM_SPLATS}u) {
        let s = splats.splats[i];
        let clip = sort_uniforms.vp * vec4f(s.pos_sx.xyz, 1.0);
        // Use w (view-space depth).  Dead splats (opacity ≈ 0) get pushed to
        // the very back so they don't interfere with blending.
        var depth = clip.w;
        if (s.color.a < 0.005 || depth < 0.0) {
            depth = 1e10;
        }
        sort_keys[i] = depth;
        sort_indices[i] = i;
    } else {
        // Padded entries for power-of-two sort: push to near distance (end of descending sort)
        sort_keys[i] = -1e10;
        sort_indices[i] = 0u; // Use a safe index
    }
}

// ----- Pass 1+: bitonic merge step -----
// Each dispatch performs one (block_k, sub_k) step of the bitonic sort.
// The host iterates over all (block_k, sub_k) pairs and dispatches once per pair.
@compute @workgroup_size(256)
fn sort_step(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let n = {@SORT_N}u;
    if (idx >= n / 2u) { return; }

    let block_k = sort_uniforms.block_k;
    let sub_k   = sort_uniforms.sub_k;

    let block_size = 1u << block_k;     // 2^block_k
    let stride     = 1u << sub_k;       // 2^sub_k

    // Identify the pair
    let pair_block = idx / stride;
    let pair_local = idx % stride;
    let i = pair_block * stride * 2u + pair_local;
    let j = i + stride;

    if (j >= n) { return; }

    // Determine sort direction: ascending within even blocks, descending in odd
    let block_id = i / block_size;
    let ascending = (block_id & 1u) == 0u;

    let key_i = sort_keys[i];
    let key_j = sort_keys[j];

    // We want back-to-front (descending depth) overall, so:
    // - "ascending" blocks sort descending (large depth first)
    // - "descending" blocks sort ascending
    let should_swap = (ascending && (key_i < key_j)) || (!ascending && (key_i > key_j));

    if (should_swap) {
        sort_keys[i]    = key_j;
        sort_keys[j]    = key_i;
        let tmp         = sort_indices[i];
        sort_indices[i] = sort_indices[j];
        sort_indices[j] = tmp;
    }
}
