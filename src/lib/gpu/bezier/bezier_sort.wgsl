// 4-pass 8-bit Radix Sort of bezier indices by camera depth

struct Bezier {
    p0: vec4f,
    p1: vec4f,
    p2: vec4f,
    p3: vec4f,
    color: vec4f,
    sh1_r: vec4f,
    sh1_g: vec4f,
    sh1_b: vec4f,
    sh1_a: vec4f,
}

struct BezierArray {
    items: array<Bezier, {@NUM_BEZIERS}u>,
}

struct SortUniforms {
    vp: mat4x4f,
    shift: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> beziers: BezierArray;
@group(0) @binding(1) var<storage, read> in_keys: array<u32, {@NUM_BEZIERS}u>;
@group(0) @binding(2) var<storage, read> in_indices: array<u32, {@NUM_BEZIERS}u>;
@group(0) @binding(3) var<storage, read_write> out_keys: array<u32, {@NUM_BEZIERS}u>;
@group(0) @binding(4) var<storage, read_write> out_indices: array<u32, {@NUM_BEZIERS}u>;
@group(0) @binding(5) var<uniform> sort_uniforms: SortUniforms;
@group(0) @binding(6) var<storage, read_write> hist: array<atomic<u32>>;

const WG_SIZE = 256u;

@compute @workgroup_size(256)
fn init_keys(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= {@NUM_BEZIERS}u) { return; }

    let b = beziers.items[i];
    
    // Approximate the 3D center of the bezier curve (t = 0.5)
    let p_center = 0.125 * b.p0.xyz + 0.375 * b.p1.xyz + 0.375 * b.p2.xyz + 0.125 * b.p3.xyz;
    
    let clip = sort_uniforms.vp * vec4f(p_center, 1.0);
    var depth = clip.w;
    
    const DEPTH_NEAR_CULL = 0.1;
    if (b.color.a < 0.005 || depth < DEPTH_NEAR_CULL) {
        depth = 1e10;
    }
    
    // Invert float bits to get a descending sort via ascending radix sort.
    out_keys[i] = ~bitcast<u32>(depth);
    out_indices[i] = i;
}

@compute @workgroup_size(256)
fn count(@builtin(global_invocation_id) gid: vec3u, @builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let W = ({@NUM_BEZIERS}u + WG_SIZE - 1u) / WG_SIZE;
    
    atomicStore(&hist[lid.x * W + wid.x], 0u);
    workgroupBarrier();

    let idx = gid.x;
    if (idx < {@NUM_BEZIERS}u) {
        let key = in_keys[idx];
        let digit = (key >> sort_uniforms.shift) & 255u;
        atomicAdd(&hist[digit * W + wid.x], 1u);
    }
}

@compute @workgroup_size(256)
fn scan(@builtin(local_invocation_id) lid: vec3u) {
    let W = ({@NUM_BEZIERS}u + WG_SIZE - 1u) / WG_SIZE;
    let bucket = lid.x; // 0..255
    
    var accum = 0u;
    for (var w = 0u; w < W; w++) {
        let idx = bucket * W + w;
        let val = atomicLoad(&hist[idx]);
        atomicStore(&hist[idx], accum);
        accum += val;
    }
    
    var<workgroup> bucket_totals: array<u32, 256>;
    bucket_totals[bucket] = accum;
    workgroupBarrier();
    
    var base = 0u;
    for (var b = 0u; b < bucket; b++) {
        base += bucket_totals[b];
    }
    
    for (var w = 0u; w < W; w++) {
        let idx = bucket * W + w;
        let val = atomicLoad(&hist[idx]);
        atomicStore(&hist[idx], val + base);
    }
}

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3u, @builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let W = ({@NUM_BEZIERS}u + WG_SIZE - 1u) / WG_SIZE;
    let idx = gid.x;
    
    var digit = 256u; // invalid
    var key = 0u;
    var val = 0u;
    
    if (idx < {@NUM_BEZIERS}u) {
        key = in_keys[idx];
        val = in_indices[idx];
        digit = (key >> sort_uniforms.shift) & 255u;
    }
    
    var<workgroup> shared_digits: array<u32, 256>;
    shared_digits[lid.x] = digit;
    workgroupBarrier();
    
    if (idx < {@NUM_BEZIERS}u) {
        var local_rank = 0u;
        for (var i = 0u; i < lid.x; i++) {
            if (shared_digits[i] == digit) {
                local_rank++;
            }
        }
        
        let global_base = atomicLoad(&hist[digit * W + wid.x]);
        let dst_idx = global_base + local_rank;
        
        out_keys[dst_idx] = key;
        out_indices[dst_idx] = val;
    }
}
