struct Splat {
    transform: vec4f,
    color: vec4f,
    rot_pad: vec4f,
}

struct SplatArray {
    splats: array<Splat, NUM_SPLATS>,
}

struct GradArray {
    data: array<atomic<i32>, NUM_PARAMS>,
}

@group(0) @binding(0) var<storage, read> splats: SplatArray;
@group(0) @binding(1) var<storage, read_write> grads: GradArray;
@group(0) @binding(2) var targetTex: texture_2d<f32>;
@group(0) @binding(3) var targetEdgeTex: texture_2d<f32>;

const MAX_TILE_SPLATS = 1024u;
var<workgroup> tile_mask: array<atomic<u32>, NUM_SPLATS_DIV_32>;
var<workgroup> tile_splats: array<u32, MAX_TILE_SPLATS>;
var<workgroup> tile_splat_count: atomic<u32>;

fn pixel_to_p(px: vec2u, dims: vec2u, aspect: f32) -> vec2f {
    let uv = (vec2f(px) + vec2f(0.5)) / vec2f(dims);
    var p = uv * 2.0 - 1.0;
    p.y = -p.y;
    p.x = p.x * aspect;
    return p;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3u, @builtin(workgroup_id) workgroup_id: vec3u, @builtin(local_invocation_id) local_id: vec3u) {
    let dims = textureDimensions(targetTex);
    let aspect = f32(dims.x) / f32(dims.y);
    
    // --- 1. COOPERATIVE TILE BINNING ---
    let local_idx = local_id.y * 16u + local_id.x;
    
    for (var i = local_idx; i < NUM_SPLATS_DIV_32; i += 256u) {
        atomicStore(&tile_mask[i], 0u);
    }
    if (local_idx == 0u) {
        atomicStore(&tile_splat_count, 0u);
    }
    workgroupBarrier();
    
    let tile_min_px = workgroup_id.xy * 16u;
    let tile_max_px = min(tile_min_px + vec2u(16u), dims);
    let p00 = pixel_to_p(tile_min_px, dims, aspect);
    let p11 = pixel_to_p(tile_max_px, dims, aspect);
    let tile_min_p = vec2f(p00.x, p11.y);
    let tile_max_p = vec2f(p11.x, p00.y);
    
    for (var splat_id = local_idx; splat_id < NUM_SPLATS; splat_id += 256u) {
        let s = splats.splats[splat_id];
        if (s.color.a < 0.005) { continue; }
        
        let pos = s.transform.xy;
        let scale = s.transform.zw;
        let safe_shape_b_cull = max(s.rot_pad.z, 0.0001);
        let R = pow(15.0 / safe_shape_b_cull, 1.0 / s.rot_pad.y);
        let max_r = R * max(scale.x, scale.y);
        
        let splat_min_p = pos - vec2f(max_r);
        let splat_max_p = pos + vec2f(max_r);
        
        if (!(splat_min_p.x > tile_max_p.x || splat_max_p.x < tile_min_p.x || 
              splat_min_p.y > tile_max_p.y || splat_max_p.y < tile_min_p.y)) {
            let word_idx = splat_id / 32u;
            let bit_idx = splat_id % 32u;
            atomicOr(&tile_mask[word_idx], 1u << bit_idx);
        }
    }
    workgroupBarrier();
    
    if (local_idx == 0u) {
        var count = 0u;
        for (var word_idx = 0u; word_idx < NUM_SPLATS_DIV_32; word_idx++) {
            var word = atomicLoad(&tile_mask[word_idx]);
            while (word != 0u) {
                let bit_idx = countTrailingZeros(word);
                let splat_id = word_idx * 32u + bit_idx;
                if (count < MAX_TILE_SPLATS) {
                    tile_splats[count] = splat_id;
                    count++;
                }
                word ^= (1u << bit_idx);
            }
        }
        atomicStore(&tile_splat_count, count);
    }
    workgroupBarrier();
    
    let splat_count = atomicLoad(&tile_splat_count);
    
    // --- 2. PIXEL EVALUATION ---
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }
    
    let p = pixel_to_p(global_id.xy, dims, aspect);
    let tgt_color = textureLoad(targetTex, global_id.xy, 0).rgb;
    let tgt_edge = textureLoad(targetEdgeTex, global_id.xy, 0).r;

    var alphas = array<f32, MAX_TILE_SPLATS>();
    var Ts = array<f32, MAX_TILE_SPLATS + 1u>();
    Ts[0] = 1.0;
    var C_pred = vec3f(0.0);
    
    for (var idx = 0u; idx < splat_count; idx++) {
        let i = tile_splats[idx];
        let s = splats.splats[i];
        let pos = s.transform.xy;
        let scale = s.transform.zw;
        let rot = s.rot_pad.x;
        let color = s.color.rgb;
        let opacity = s.color.a;
        
        let d = p - pos;
        
        let co = cos(rot);
        let si = sin(rot);
        let local_p = vec2f(co * d.x + si * d.y, -si * d.x + co * d.y);
        let safe_scale = max(vec2f(0.0001), scale);
        let scaled_p = local_p / safe_scale;
        let r = max(length(scaled_p), 0.0001);
        let power = -s.rot_pad.z * pow(r, s.rot_pad.y);
        
        var a = 0.0;
        if (power > -15.0) {
            a = exp(power) * opacity;
        }
        
        a = clamp(a, 0.0, 0.999);
        
        alphas[idx] = a;
        C_pred += Ts[idx] * a * color;
        Ts[idx+1] = Ts[idx] * (1.0 - a);
    }

    let background = vec3f(0.1);
    C_pred += Ts[splat_count] * background;
    
    let dC = 2.0 * (C_pred - tgt_color);
    let edge_weight = 1.0 + tgt_edge * 4.0;
    let dC_weighted = dC * edge_weight;
    
    var dT = dot(dC_weighted, background);

    for (var j = 0u; j < splat_count; j++) {
        let idx = splat_count - 1u - j;
        let i = tile_splats[idx];
        let a = alphas[idx];
        if (a < 0.001) { continue; }
        
        let s = splats.splats[i];
        let pos = s.transform.xy;
        let scale = s.transform.zw;
        let rot = s.rot_pad.x;
        let color = s.color.rgb;
        let opacity = s.color.a;

        let T_prev = Ts[idx];
        
        let dColor = dC_weighted * (T_prev * a);
        let da = dT * (-T_prev) + dot(dC_weighted, T_prev * color);
        dT = dT * (1.0 - a) + dot(dC_weighted, a * color);

        let d = p - pos;
        let co = cos(rot);
        let si = sin(rot);
        let local_p = vec2f(co * d.x + si * d.y, -si * d.x + co * d.y);
        let safe_scale = max(vec2f(0.0001), scale);
        let scaled_p = local_p / safe_scale;
        let r = max(length(scaled_p), 0.0001);
        let shape_a = s.rot_pad.y;
        let shape_b = s.rot_pad.z;
        let power = -shape_b * pow(r, shape_a);
        
        var d_opacity = 0.0;
        var d_pos = vec2f(0.0);
        var d_scale = vec2f(0.0);
        var d_rot = 0.0;
        var d_shape_a = 0.0;
        var d_shape_b = 0.0;
        
        if (power > -15.0) {
            let a_unscaled = exp(power);
            d_opacity = da * a_unscaled;
            let d_power = da * opacity * a_unscaled;
            
            let r_pow_a = pow(r, shape_a);
            let r_pow_a_minus_2 = pow(r, shape_a - 2.0);
            
            d_shape_a = d_power * (-shape_b * r_pow_a * log(r));
            d_shape_b = d_power * (-r_pow_a);
            
            let d_scaled_p = d_power * (-shape_b * shape_a * r_pow_a_minus_2) * scaled_p;
            d_scale = d_scaled_p * (-local_p / (safe_scale * safe_scale));
            let d_local_p = d_scaled_p / safe_scale;
            
            let d_co = d_local_p.x * d.x + d_local_p.y * d.y;
            let d_si = d_local_p.x * d.y - d_local_p.y * d.x;
            d_rot = d_co * (-si) + d_si * co;
            
            let d_d = vec2f(
                d_local_p.x * co - d_local_p.y * si,
                d_local_p.x * si + d_local_p.y * co
            );
            d_pos = -d_d;
        }

        let FP_SCALE_POS = 10000.0;
        let FP_SCALE_COL = 100000.0;
        
        let base_idx = i * 11u;
        atomicAdd(&grads.data[base_idx + 0u], i32(d_pos.x * FP_SCALE_POS));
        atomicAdd(&grads.data[base_idx + 1u], i32(d_pos.y * FP_SCALE_POS));
        atomicAdd(&grads.data[base_idx + 2u], i32(d_scale.x * FP_SCALE_POS));
        atomicAdd(&grads.data[base_idx + 3u], i32(d_scale.y * FP_SCALE_POS));
        atomicAdd(&grads.data[base_idx + 4u], i32(dColor.r * FP_SCALE_COL));
        atomicAdd(&grads.data[base_idx + 5u], i32(dColor.g * FP_SCALE_COL));
        atomicAdd(&grads.data[base_idx + 6u], i32(dColor.b * FP_SCALE_COL));
        atomicAdd(&grads.data[base_idx + 7u], i32(d_opacity * FP_SCALE_COL));
        atomicAdd(&grads.data[base_idx + 8u], i32(d_rot * FP_SCALE_POS));
        atomicAdd(&grads.data[base_idx + 9u], i32(d_shape_a * FP_SCALE_POS));
        atomicAdd(&grads.data[base_idx + 10u], i32(d_shape_b * FP_SCALE_POS));
    }
}
