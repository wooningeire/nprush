struct Splat {
    pos_sx: vec4f,
    color: vec4f,
    quat: vec4f,
    sy_shape: vec4f,
    sh1_r: vec4f,
    sh1_g: vec4f,
    sh1_b: vec4f,
    sh1_a: vec4f,
}

struct SplatArray {
    splats: array<Splat, {@NUM_SPLATS}u>,
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

@group(0) @binding(0) var<storage, read> splats: SplatArray;
@group(0) @binding(1) var<storage, read_write> instance_keys: array<vec2u>; // x = depth, y = tile_id
@group(0) @binding(2) var<storage, read_write> instance_vals: array<u32>;   // splat_id
@group(0) @binding(3) var<storage, read_write> atomic_count: atomic<u32>;
@group(0) @binding(4) var<uniform> binning_uniforms: BinningUniforms;

fn quat_rotate(q: vec4f, v: vec3f) -> vec3f {
    let t = 2.0 * cross(q.yzw, v);
    return v + q.x * t + cross(q.yzw, t);
}

fn project_axis(vp: mat4x4f, ax_world: vec3f, clip_xy: vec2f, w: f32, aspect: f32) -> vec2f {
    let ac = vp * vec4f(ax_world, 0.0);
    return vec2f(
        (ac.x * w - clip_xy.x * ac.w) / (w * w) * aspect,
        (ac.y * w - clip_xy.y * ac.w) / (w * w)
    );
}

@compute @workgroup_size(256)
fn instantiate(@builtin(global_invocation_id) gid: vec3u) {
    let splat_id = gid.x;
    if (splat_id >= {@NUM_SPLATS}u) { return; }

    let s = splats.splats[splat_id];
    if (s.color.a < 0.005) { return; }

    let clip = binning_uniforms.vp * vec4f(s.pos_sx.xyz, 1.0);
    let w = clip.w;
    if (w < 0.1) { return; } // Near cull

    let dims_y = f32(binning_uniforms.grid_height * 16u);
    let aspect = f32(binning_uniforms.grid_width * 16u) / dims_y;

    let proj_xy = vec2f(clip.x / w * aspect, clip.y / w);

    let sx = max(s.pos_sx.w, 0.0001);
    let sy = max(s.sy_shape.x, 0.0001);
    let sz = max(s.sy_shape.w, 0.0001);

    let q = s.quat;
    let ax_w = quat_rotate(q, vec3f(1.0, 0.0, 0.0));
    let ay_w = quat_rotate(q, vec3f(0.0, 1.0, 0.0));
    let az_w = quat_rotate(q, vec3f(0.0, 0.0, 1.0));

    let ax_s = project_axis(binning_uniforms.vp, ax_w, proj_xy, w, aspect);
    let ay_s = project_axis(binning_uniforms.vp, ay_w, proj_xy, w, aspect);
    let az_s = project_axis(binning_uniforms.vp, az_w, proj_xy, w, aspect);

    let m0x = ax_s.x * sx; let m0y = ax_s.y * sx;
    let m1x = ay_s.x * sy; let m1y = ay_s.y * sy;
    let m2x = az_s.x * sz; let m2y = az_s.y * sz;
    var cov00 = m0x*m0x + m1x*m1x + m2x*m2x;
    var cov11 = m0y*m0y + m1y*m1y + m2y*m2y;

    let filter_std = 0.3 * (2.0 / dims_y);
    let filter2 = filter_std * filter_std;
    cov00 += filter2;
    cov11 += filter2;

    let max_r = 4.5 * sqrt(max(max(cov00, cov11), 1e-9));

    let min_p = proj_xy - vec2f(max_r);
    let max_p = proj_xy + vec2f(max_r);

    // Convert NDC [-aspect, aspect]x[-1, 1] to tile coords
    // p = uv * 2 - 1 -> uv = (p + 1) / 2. (Wait, x is -aspect to aspect, y is 1 to -1)
    // Actually:
    // px = (clip.x / w * 0.5 + 0.5) * dims_x
    // py = (-clip.y / w * 0.5 + 0.5) * dims_y
    let min_px = (min_p.x / aspect * 0.5 + 0.5) * f32(binning_uniforms.grid_width * 16u);
    let max_px = (max_p.x / aspect * 0.5 + 0.5) * f32(binning_uniforms.grid_width * 16u);
    let min_py = (-max_p.y * 0.5 + 0.5) * dims_y;
    let max_py = (-min_p.y * 0.5 + 0.5) * dims_y;

    let tile_min_x = u32(clamp(min_px / 16.0, 0.0, f32(binning_uniforms.grid_width)));
    let tile_max_x = u32(clamp(ceil(max_px / 16.0), 0.0, f32(binning_uniforms.grid_width)));
    let tile_min_y = u32(clamp(min_py / 16.0, 0.0, f32(binning_uniforms.grid_height)));
    let tile_max_y = u32(clamp(ceil(max_py / 16.0), 0.0, f32(binning_uniforms.grid_height)));

    if (tile_min_x >= tile_max_x || tile_min_y >= tile_max_y) { return; }

    let depth_key = ~bitcast<u32>(w); // Invert float to sort descending

    for (var y = tile_min_y; y < tile_max_y; y++) {
        for (var x = tile_min_x; x < tile_max_x; x++) {
            let idx = atomicAdd(&atomic_count, 1u);
            if (idx < binning_uniforms.max_instances) {
                let tile_id = y * binning_uniforms.grid_width + x;
                instance_keys[idx] = vec2u(depth_key, tile_id);
                instance_vals[idx] = splat_id;
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
    
    // Also reset tile_starts and tile_ends in a separate pass?
    // We can clear them where idx < grid_width * grid_height.
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
