import { interpolateWgslTemplate } from "../wgsl-templates/interpolateWgslTemplate.ts";
import { Splat } from "./Splat.wgsl.ts";

export default interpolateWgslTemplate`
// 4-pass 8-bit radix sort of splat indices by camera depth (far-to-near).
//
// Each pass processes one 8-bit digit of the 32-bit key. Keys are bit-inverted
// floats so that an ascending radix sort produces back-to-front (descending depth) order.
//
// Per pass: count → scan → scatter. The sort alternates between buffer A and B;
// init_keys writes to A, so after 4 passes the result is in A.


struct SplatArray {
    splats: array<${Splat}, {@NUM_SPLATS}u>,
}

struct SortUniforms {
    vp: mat4x4f,
    shift: u32,
    _pad: vec3u,
}

@group(0) @binding(0) var<storage, read> splats: SplatArray;
@group(0) @binding(1) var<storage, read> in_keys: array<u32, {@NUM_SPLATS}u>;
@group(0) @binding(2) var<storage, read> in_indices: array<u32, {@NUM_SPLATS}u>;
@group(0) @binding(3) var<storage, read_write> out_keys: array<u32, {@NUM_SPLATS}u>;
@group(0) @binding(4) var<storage, read_write> out_indices: array<u32, {@NUM_SPLATS}u>;
@group(0) @binding(5) var<uniform> sort_uniforms: SortUniforms;
@group(0) @binding(6) var<storage, read_write> hist: array<atomic<u32>>;

const WG_SIZE = 256u;

// Pass 0: compute bit-inverted depth keys and initialize indices.
// Writes to out_keys/out_indices only (in_keys/in_indices are unused).
@compute @workgroup_size(256)
fn init_keys(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= {@NUM_SPLATS}u) { return; }

    let s = splats.splats[i];
    let clip = sort_uniforms.vp * vec4f(s.pos_sx.xyz, 1.0);
    var depth = clip.w;
    const DEPTH_NEAR_CULL = 0.1;
    if (s.color.a < 0.005 || depth < DEPTH_NEAR_CULL) {
        depth = 1e10;
    }
    // Bit-invert so ascending radix sort = descending depth (back-to-front).
    out_keys[i] = ~bitcast<u32>(depth);
    out_indices[i] = i;
}

// Pass 1: count elements per 8-bit digit bucket, per workgroup, into hist.
// hist layout: hist[digit * W + wg_id], where W = number of workgroups.
@compute @workgroup_size(256)
fn count(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(workgroup_id) wid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let W = ({@NUM_SPLATS}u + WG_SIZE - 1u) / WG_SIZE;
    // Each thread clears its own column slot (bucket=lid.x, wg=wid.x).
    atomicStore(&hist[lid.x * W + wid.x], 0u);
    workgroupBarrier();

    let idx = gid.x;
    if (idx < {@NUM_SPLATS}u) {
        let digit = (in_keys[idx] >> sort_uniforms.shift) & 255u;
        atomicAdd(&hist[digit * W + wid.x], 1u);
    }
}

// Workgroup-scope scratch arrays for scan and scatter (must be module-scope in WGSL).
var<workgroup> wg_bucket_totals: array<u32, 256>;
var<workgroup> wg_digits: array<u32, 256>;

// Pass 2: convert per-workgroup counts to exclusive prefix-sum scatter offsets.
// Single workgroup of 256 threads; thread i handles bucket i.
@compute @workgroup_size(256)
fn scan(@builtin(local_invocation_id) lid: vec3u) {
    let W = ({@NUM_SPLATS}u + WG_SIZE - 1u) / WG_SIZE;
    let bucket = lid.x;

    // Exclusive prefix-sum within this bucket across W workgroup columns.
    var accum = 0u;
    for (var w = 0u; w < W; w++) {
        let val = atomicLoad(&hist[bucket * W + w]);
        atomicStore(&hist[bucket * W + w], accum);
        accum += val;
    }
    wg_bucket_totals[bucket] = accum; // total count for this bucket
    workgroupBarrier();

    // Exclusive prefix-sum across buckets to find each bucket's global start.
    var base = 0u;
    for (var b = 0u; b < bucket; b++) {
        base += wg_bucket_totals[b];
    }

    // Add global base to all workgroup column entries for this bucket.
    for (var w = 0u; w < W; w++) {
        let val = atomicLoad(&hist[bucket * W + w]);
        atomicStore(&hist[bucket * W + w], val + base);
    }
}

// Pass 3: scatter elements to their sorted output positions.
@compute @workgroup_size(256)
fn scatter(
    @builtin(global_invocation_id) gid: vec3u,
    @builtin(workgroup_id) wid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
) {
    let W = ({@NUM_SPLATS}u + WG_SIZE - 1u) / WG_SIZE;
    let idx = gid.x;

    var digit = 256u; // sentinel for out-of-bounds threads
    var key = 0u;
    var val = 0u;

    if (idx < {@NUM_SPLATS}u) {
        key = in_keys[idx];
        val = in_indices[idx];
        digit = (key >> sort_uniforms.shift) & 255u;
    }

    wg_digits[lid.x] = digit;
    workgroupBarrier();

    if (idx < {@NUM_SPLATS}u) {
        // Local rank: count earlier threads in this workgroup with the same digit.
        var local_rank = 0u;
        for (var i = 0u; i < lid.x; i++) {
            if (wg_digits[i] == digit) {
                local_rank++;
            }
        }
        let dst = atomicLoad(&hist[digit * W + wid.x]) + local_rank;
        out_keys[dst] = key;
        out_indices[dst] = val;
    }
}
`;