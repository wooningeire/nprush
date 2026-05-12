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

struct BinningUniforms {
    vp: mat4x4f,
    grid_width: u32,
    grid_height: u32,
    max_instances: u32,
    _pad: u32,
}

struct SortUniforms {
    shift: u32,
    word_idx: u32, // 0 for depth, 1 for tile_id
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> beziers: BezierArray;
@group(0) @binding(1) var<storage, read_write> instance_keys: array<vec2u>; // x = depth, y = tile_id
@group(0) @binding(2) var<storage, read_write> instance_vals: array<u32>;   // bezier_id
@group(0) @binding(3) var<storage, read_write> atomic_count: atomic<u32>;
@group(0) @binding(4) var<uniform> binning_uniforms: BinningUniforms;

fn bezier_at(p0: vec2f, p1: vec2f, p2: vec2f, p3: vec2f, t: f32) -> vec2f {
    let omt = 1.0 - t;
    return omt*omt*omt * p0
         + 3.0 * omt*omt * t * p1
         + 3.0 * omt * t*t * p2
         + t*t*t * p3;
}

fn project_center(vp: mat4x4f, pos3: vec3f, aspect: f32) -> vec3f {
    let clip = vp * vec4f(pos3, 1.0);
    return vec3f(clip.x / clip.w * aspect, clip.y / clip.w, clip.w);
}

@compute @workgroup_size(256)
fn instantiate(@builtin(global_invocation_id) gid: vec3u) {
    let bezier_id = gid.x;
    if (bezier_id >= {@NUM_BEZIERS}u) { return; }

    let b = beziers.items[bezier_id];
    if (b.color.a < {@BEZIER_KILL_ALPHA_THRESH}) { return; }

    let dims_y = f32(binning_uniforms.grid_height * 16u);
    let aspect = f32(binning_uniforms.grid_width * 16u) / dims_y;

    let proj0 = project_center(binning_uniforms.vp, b.p0.xyz, aspect);
    let proj1 = project_center(binning_uniforms.vp, b.p1.xyz, aspect);
    let proj2 = project_center(binning_uniforms.vp, b.p2.xyz, aspect);
    let proj3 = project_center(binning_uniforms.vp, b.p3.xyz, aspect);

    const DEPTH_NEAR_CULL = 0.1;
    if (proj0.z < DEPTH_NEAR_CULL || proj1.z < DEPTH_NEAR_CULL || proj2.z < DEPTH_NEAR_CULL || proj3.z < DEPTH_NEAR_CULL) { return; }

    let width = max(b.p0.w, 0.001);
    let softness = max(b.p1.w, 0.001);

    let p0 = proj0.xy;
    let p1 = proj1.xy;
    let p2 = proj2.xy;
    let p3 = proj3.xy;

    let pm1 = bezier_at(p0, p1, p2, p3, 0.25);
    let pm2 = bezier_at(p0, p1, p2, p3, 0.5);
    let pm3 = bezier_at(p0, p1, p2, p3, 0.75);

    let outer_cull = width + softness;
    let min_p = min(min(min(p0, p3), min(pm1, pm2)), pm3) - vec2f(outer_cull);
    let max_p = max(max(max(p0, p3), max(pm1, pm2)), pm3) + vec2f(outer_cull);

    let min_px = (min_p.x / aspect * 0.5 + 0.5) * f32(binning_uniforms.grid_width * 16u);
    let max_px = (max_p.x / aspect * 0.5 + 0.5) * f32(binning_uniforms.grid_width * 16u);
    // Be careful with y-axis: p.y is -1 (bottom) to 1 (top) usually
    // pixel_to_p does: p.y = - (uv.y * 2 - 1). So p.y = 1 - 2*uv_y.
    // uv_y = (1 - p.y) / 2
    let min_py = (1.0 - max_p.y) * 0.5 * dims_y;
    let max_py = (1.0 - min_p.y) * 0.5 * dims_y;

    let tile_min_x = u32(clamp(min_px / 16.0, 0.0, f32(binning_uniforms.grid_width)));
    let tile_max_x = u32(clamp(ceil(max_px / 16.0), 0.0, f32(binning_uniforms.grid_width)));
    let tile_min_y = u32(clamp(min_py / 16.0, 0.0, f32(binning_uniforms.grid_height)));
    let tile_max_y = u32(clamp(ceil(max_py / 16.0), 0.0, f32(binning_uniforms.grid_height)));

    if (tile_min_x >= tile_max_x || tile_min_y >= tile_max_y) { return; }

    let p_center = 0.125 * b.p0.xyz + 0.375 * b.p1.xyz + 0.375 * b.p2.xyz + 0.125 * b.p3.xyz;
    let clip = binning_uniforms.vp * vec4f(p_center, 1.0);
    let w = clip.w;
    let depth_key = ~bitcast<u32>(w); // Invert float to sort descending

    for (var y = tile_min_y; y < tile_max_y; y++) {
        for (var x = tile_min_x; x < tile_max_x; x++) {
            let idx = atomicAdd(&atomic_count, 1u);
            if (idx < binning_uniforms.max_instances) {
                let tile_id = y * binning_uniforms.grid_width + x;
                instance_keys[idx] = vec2u(depth_key, tile_id);
                instance_vals[idx] = bezier_id;
            }
        }
    }
}

// -------------------------------------------------------------
// RADIX SORT PASSES
// -------------------------------------------------------------

@group(0) @binding(5) var<storage, read> in_keys: array<vec2u>;
@group(0) @binding(6) var<storage, read> in_vals: array<u32>;
@group(0) @binding(7) var<storage, read_write> out_keys: array<vec2u>;
@group(0) @binding(8) var<storage, read_write> out_vals: array<u32>;
@group(0) @binding(9) var<uniform> sort_uniforms: SortUniforms;
@group(0) @binding(10) var<storage, read_write> hist: array<atomic<u32>>;

const WG_SIZE = 256u;

@compute @workgroup_size(256)
fn count(@builtin(global_invocation_id) gid: vec3u, @builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let count_val = atomicLoad(&atomic_count);
    let count = min(count_val, binning_uniforms.max_instances);
    let W = (count + WG_SIZE - 1u) / WG_SIZE;
    
    atomicStore(&hist[lid.x * W + wid.x], 0u);
    workgroupBarrier();

    let idx = gid.x;
    if (idx < count) {
        let key = in_keys[idx];
        let word = select(key.x, key.y, sort_uniforms.word_idx == 1u);
        let digit = (word >> sort_uniforms.shift) & 255u;
        atomicAdd(&hist[digit * W + wid.x], 1u);
    }
}

@compute @workgroup_size(256)
fn scan(@builtin(local_invocation_id) lid: vec3u) {
    let count_val = atomicLoad(&atomic_count);
    let count = min(count_val, binning_uniforms.max_instances);
    let W = (count + WG_SIZE - 1u) / WG_SIZE;
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
    let count_val = atomicLoad(&atomic_count);
    let count = min(count_val, binning_uniforms.max_instances);
    let W = (count + WG_SIZE - 1u) / WG_SIZE;
    let idx = gid.x;
    
    var digit = 256u; // invalid
    var key = vec2u(0u, 0u);
    var val = 0u;
    
    if (idx < count) {
        key = in_keys[idx];
        val = in_vals[idx];
        let word = select(key.x, key.y, sort_uniforms.word_idx == 1u);
        digit = (word >> sort_uniforms.shift) & 255u;
    }
    
    var<workgroup> shared_digits: array<u32, 256>;
    shared_digits[lid.x] = digit;
    workgroupBarrier();
    
    if (idx < count) {
        var local_rank = 0u;
        for (var i = 0u; i < lid.x; i++) {
            if (shared_digits[i] == digit) {
                local_rank++;
            }
        }
        
        let global_base = atomicLoad(&hist[digit * W + wid.x]);
        let dst_idx = global_base + local_rank;
        
        out_keys[dst_idx] = key;
        out_vals[dst_idx] = val;
    }
}

// -------------------------------------------------------------
// TILE RANGES PASS
// -------------------------------------------------------------

@group(0) @binding(11) var<storage, read_write> tile_starts: array<u32>;
@group(0) @binding(12) var<storage, read_write> tile_ends: array<u32>;

@compute @workgroup_size(256)
fn calc_ranges(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let count_val = atomicLoad(&atomic_count);
    let count = min(count_val, binning_uniforms.max_instances);
    
    let num_tiles = binning_uniforms.grid_width * binning_uniforms.grid_height;
    if (idx < num_tiles) {
        tile_starts[idx] = 0u;
        tile_ends[idx] = 0u;
    }
    
    if (idx >= count) { return; }

    let tile_id = in_keys[idx].y;
    
    let is_first = (idx == 0u) || (in_keys[idx - 1u].y != tile_id);
    let is_last = (idx == count - 1u) || (in_keys[idx + 1u].y != tile_id);

    if (is_first) {
        tile_starts[tile_id] = idx;
    }
    if (is_last) {
        tile_ends[tile_id] = idx + 1u;
    }
}
