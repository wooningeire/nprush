// Differentiable rasterization of cubic bezier curves.
// Now in 3D: curves are defined by 3D control points projected to the screen.

struct Bezier {
    p0: vec4f,    // x, y, z, width
    p1: vec4f,    // x, y, z, softness
    p2: vec4f,    // x, y, z, _pad
    p3: vec4f,    // x, y, z, _pad
    color: vec4f, // r, g, b, a
}

struct BezierArray {
    items: array<Bezier, NUM_BEZIERS>,
}

struct GradArray {
    data: array<atomic<i32>, NUM_BEZIER_PARAMS>,
}

struct BezierUniforms {
    vp: mat4x4f,
    mode: f32, // 0: Edge, 1: Color+Depth
    _pad: vec3f,
}

struct ADCArray {
    grad_accum: array<f32, NUM_BEZIERS>,
    loss_accum: array<f32, NUM_BEZIERS>,
}

@group(0) @binding(0) var<storage, read> beziers: BezierArray;
@group(0) @binding(1) var<storage, read_write> grads: GradArray;
@group(0) @binding(2) var targetTex: texture_2d<f32>;
@group(0) @binding(3) var targetDepthTex: texture_2d<f32>;
@group(0) @binding(4) var<uniform> uniforms: BezierUniforms;
@group(0) @binding(5) var bgTex: texture_2d<f32>;
@group(0) @binding(6) var bgDepthTex: texture_2d<f32>;
@group(0) @binding(7) var<storage, read_write> adc: ADCArray;

const N_SEG: u32 = 16u;

fn bezier_at(p0: vec2f, p1: vec2f, p2: vec2f, p3: vec2f, t: f32) -> vec2f {
    let omt = 1.0 - t;
    return omt*omt*omt * p0
         + 3.0 * omt*omt * t * p1
         + 3.0 * omt * t*t * p2
         + t*t*t * p3;
}

fn bernstein(t: f32) -> vec4f {
    let omt = 1.0 - t;
    return vec4f(omt*omt*omt, 3.0*omt*omt*t, 3.0*omt*t*t, t*t*t);
}

fn project_center(vp: mat4x4f, pos3: vec3f, aspect: f32) -> vec3f {
    let clip = vp * vec4f(pos3, 1.0);
    return vec3f(clip.x / clip.w * aspect, clip.y / clip.w, clip.w);
}

fn backproject_gradient(vp: mat4x4f, pos3: vec3f, aspect: f32, dp2d: vec2f) -> vec3f {
    let clip = vp * vec4f(pos3, 1.0);
    let w = clip.w;
    let w2 = w * w;
    var dp3d = vec3f(0.0);
    for (var ax = 0u; ax < 3u; ax++) {
        let vp_0j = vp[ax][0];
        let vp_1j = vp[ax][1];
        let vp_3j = vp[ax][3];
        let ds_dx = aspect * (vp_0j * w - clip.x * vp_3j) / w2;
        let ds_dy = (vp_1j * w - clip.y * vp_3j) / w2;
        dp3d[ax] = dp2d.x * ds_dx + dp2d.y * ds_dy;
    }
    return dp3d;
}

const MAX_TILE_BEZIERS = 1024u;
var<workgroup> tile_mask: array<atomic<u32>, NUM_BEZIERS_DIV_32>;
var<workgroup> tile_beziers: array<u32, MAX_TILE_BEZIERS>;
var<workgroup> tile_bezier_count: atomic<u32>;

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
    
    for (var i = local_idx; i < NUM_BEZIERS_DIV_32; i += 256u) {
        atomicStore(&tile_mask[i], 0u);
    }
    if (local_idx == 0u) {
        atomicStore(&tile_bezier_count, 0u);
    }
    workgroupBarrier();
    
    let tile_min_px = workgroup_id.xy * 16u;
    let tile_max_px = min(tile_min_px + vec2u(16u), dims);
    let p00 = pixel_to_p(tile_min_px, dims, aspect);
    let p11 = pixel_to_p(tile_max_px, dims, aspect);
    let tile_min_p = vec2f(p00.x, p11.y);
    let tile_max_p = vec2f(p11.x, p00.y);
    
    for (var bezier_id = local_idx; bezier_id < NUM_BEZIERS; bezier_id += 256u) {
        let b = beziers.items[bezier_id];
        if (b.color.a < 0.005) { continue; }
        
        let width = max(b.p0.w, 0.001);
        let softness = max(b.p1.w, 0.001);
        
        let proj0 = project_center(uniforms.vp, b.p0.xyz, aspect);
        let proj1 = project_center(uniforms.vp, b.p1.xyz, aspect);
        let proj2 = project_center(uniforms.vp, b.p2.xyz, aspect);
        let proj3 = project_center(uniforms.vp, b.p3.xyz, aspect);
        if (proj0.z < 0.0 || proj1.z < 0.0 || proj2.z < 0.0 || proj3.z < 0.0) { continue; }

        let p0 = proj0.xy;
        let p1 = proj1.xy;
        let p2 = proj2.xy;
        let p3 = proj3.xy;
        
        let outer_cull = width + softness;
        let min_p = min(min(p0, p1), min(p2, p3)) - vec2f(outer_cull);
        let max_p = max(max(p0, p1), max(p2, p3)) + vec2f(outer_cull);
        
        if (!(min_p.x > tile_max_p.x || max_p.x < tile_min_p.x || 
              min_p.y > tile_max_p.y || max_p.y < tile_min_p.y)) {
            let word_idx = bezier_id / 32u;
            let bit_idx = bezier_id % 32u;
            atomicOr(&tile_mask[word_idx], 1u << bit_idx);
        }
    }
    workgroupBarrier();
    
    if (local_idx == 0u) {
        var count = 0u;
        for (var word_idx = 0u; word_idx < NUM_BEZIERS_DIV_32; word_idx++) {
            var word = atomicLoad(&tile_mask[word_idx]);
            while (word != 0u) {
                let bit_idx = countTrailingZeros(word);
                let bezier_id = word_idx * 32u + bit_idx;
                if (count < MAX_TILE_BEZIERS) {
                    tile_beziers[count] = bezier_id;
                    count++;
                }
                word ^= (1u << bit_idx);
            }
        }
        atomicStore(&tile_bezier_count, count);
    }
    workgroupBarrier();
    
    let bezier_count = atomicLoad(&tile_bezier_count);

    // --- 2. PIXEL EVALUATION ---
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    let p = pixel_to_p(global_id.xy, dims, aspect);
    let tgt_color = textureLoad(targetTex, global_id.xy, 0).rgb;
    let tgt_depth = textureLoad(targetDepthTex, global_id.xy, 0).r;

    var alphas = array<f32, MAX_TILE_BEZIERS>();
    var min_seg = array<u32, MAX_TILE_BEZIERS>();
    var depths = array<f32, MAX_TILE_BEZIERS>();
    var Ts = array<f32, MAX_TILE_BEZIERS + 1u>();
    Ts[0] = 1.0;
    var C_pred = vec3f(0.0);
    var D_pred = 0.0;

    for (var idx = 0u; idx < bezier_count; idx++) {
        let i = tile_beziers[idx];
        let b = beziers.items[i];
        
        let width = max(b.p0.w, 0.001);
        let softness = max(b.p1.w, 0.001);
        let opacity = b.color.a;
        
        let proj0 = project_center(uniforms.vp, b.p0.xyz, aspect);
        let proj1 = project_center(uniforms.vp, b.p1.xyz, aspect);
        let proj2 = project_center(uniforms.vp, b.p2.xyz, aspect);
        let proj3 = project_center(uniforms.vp, b.p3.xyz, aspect);
        let p0 = proj0.xy;
        let p1 = proj1.xy;
        let p2 = proj2.xy;
        let p3 = proj3.xy;

        var min_d = 1e9;
        var min_k = 1u;
        var min_u = 0.0;
        var prev = p0;
        for (var k = 1u; k <= N_SEG; k = k + 1u) {
            let curr = bezier_at(p0, p1, p2, p3, f32(k) / f32(N_SEG));
            let seg = curr - prev;
            let len2 = max(dot(seg, seg), 1e-8);
            let u = clamp(dot(p - prev, seg) / len2, 0.0, 1.0);
            let proj = prev + u * seg;
            let d = length(p - proj);
            if (d < min_d) {
                min_d = d;
                min_k = k;
                min_u = u;
            }
            prev = curr;
        }
        let t = (f32(min_k - 1u) + min_u) / f32(N_SEG);
        let pressure = smoothstep(0.0, 0.5, t) * smoothstep(1.0, 0.5, t);
        let local_width = width * pressure;
        let local_softness = softness * pressure;
        let local_opacity = opacity * pressure;

        let inner = local_width - local_softness;
        let outer = local_width + local_softness;
        let a_geom = 1.0 - smoothstep(inner, outer, min_d);
        var a = clamp(a_geom * local_opacity, 0.0, 0.999);

        // Interpolate depth (w-component of projected points)
        let B = bernstein(t);
        let d_val = dot(B, vec4f(proj0.z, proj1.z, proj2.z, proj3.z));

        alphas[idx] = a;
        min_seg[idx] = min_k;
        depths[idx] = d_val;
        C_pred += Ts[idx] * a * b.color.rgb;
        D_pred += Ts[idx] * a * d_val;
        Ts[idx + 1u] = Ts[idx] * (1.0 - a);
    }

    let background_sample = textureLoad(bgTex, global_id.xy, 0).rgb;
    let bg_depth_sample = textureLoad(bgDepthTex, global_id.xy, 0).r;
    
    var background = vec3f(0.0);
    var bg_depth = 1.0;
    let color_mode = uniforms.mode > 0.5;
    background = select(vec3f(0.0), background_sample, color_mode);
    bg_depth   = select(1.0, bg_depth_sample, color_mode);

    C_pred += Ts[bezier_count] * background;
    D_pred += Ts[bezier_count] * bg_depth;

    let dC = 2.0 * (C_pred - tgt_color);
    // let dD_total = 2.0 * (D_pred - tgt_depth);
    let dD_total = 0.0;
    var dT = dot(dC, background) + dD_total * bg_depth;

    let FP_SCALE_POS = 10000.0;
    let FP_SCALE_COL = 100000.0;

    for (var j = 0u; j < bezier_count; j++) {
        let idx = bezier_count - 1u - j;
        let a = alphas[idx];
        if (a < 0.001) { continue; }
        
        let i = tile_beziers[idx];
        let b = beziers.items[i];
        let width = max(b.p0.w, 0.001);
        let softness = max(b.p1.w, 0.001);
        let opacity = b.color.a;
        let color = b.color.rgb;

        let p0 = project_center(uniforms.vp, b.p0.xyz, aspect).xy;
        let p1 = project_center(uniforms.vp, b.p1.xyz, aspect).xy;
        let p2 = project_center(uniforms.vp, b.p2.xyz, aspect).xy;
        let p3 = project_center(uniforms.vp, b.p3.xyz, aspect).xy;

        let T_prev = Ts[idx];
        let d_val = depths[idx];

        let dColor = dC * (T_prev * a);
        let da = dT * (-T_prev) + dot(dC, T_prev * color) + dD_total * (T_prev * d_val);
        dT = dT * (1.0 - a) + dot(dC, a * color) + dD_total * (a * d_val);

        let k = min_seg[idx];
        let t_prev = f32(k - 1u) / f32(N_SEG);
        let t_curr = f32(k) / f32(N_SEG);
        let prev_pt = bezier_at(p0, p1, p2, p3, t_prev);
        let curr_pt = bezier_at(p0, p1, p2, p3, t_curr);
        let seg = curr_pt - prev_pt;
        let len2 = max(dot(seg, seg), 1e-8);
        let u_clamped = clamp(dot(p - prev_pt, seg) / len2, 0.0, 1.0);
        let proj = prev_pt + u_clamped * seg;
        let d_vec = p - proj;
        let d = max(length(d_vec), 1e-6);

        let t_pixel = (f32(k - 1u) + u_clamped) / f32(N_SEG);
        let pressure = smoothstep(0.0, 0.5, t_pixel) * smoothstep(1.0, 0.5, t_pixel);
        let local_width = width * pressure;
        let local_softness = softness * pressure;
        let local_opacity = opacity * pressure;

        let inner = local_width - local_softness;
        let outer = local_width + local_softness;
        let denom = max(outer - inner, 1e-6);
        let x_inner = clamp((d - inner) / denom, 0.0, 1.0);
        let smoothstep_deriv = 6.0 * x_inner * (1.0 - x_inner) / denom;
        let in_softband = (d > inner) && (d < outer);

        let a_geom = a / max(local_opacity, 1e-4);
        
        // Edge mode weighting: penalize opacity if not on an edge
        let OFF_EDGE_ALPHA = 0.35;
        let off_w = select(0.0, OFF_EDGE_ALPHA * (1.0 - tgt_depth), uniforms.mode < 0.5);

        var d_opacity = (da * (1.0 - smoothstep(inner, outer, d)) + off_w * a_geom) * pressure;

        var dD = 0.0;
        var dWidth = 0.0;
        var dSoft = 0.0;
        dD     = select(0.0, -(da + off_w) * local_opacity * smoothstep_deriv, in_softband);
        dWidth = select(0.0, ((da + off_w) * local_opacity * smoothstep_deriv / denom) * pressure, in_softband);
        dSoft  = select(0.0, (-(da + off_w) * local_opacity * smoothstep_deriv * (local_width - d)
                / max(2.0 * local_softness * local_softness, 1e-6)) * pressure, in_softband);

        // Depth gradient with respect to control points (z/w component)
        let B_pixel = bernstein(t_pixel);
        let dDepth_dZs = dD_total * (T_prev * a) * B_pixel;



        let dProj = -dD * d_vec / d;
        let dPrevPt = (1.0 - u_clamped) * dProj;
        let dCurrPt = u_clamped * dProj;

        let B_prev = bernstein(t_prev);
        let B_curr = bernstein(t_curr);
        let dP0_2d = B_prev.x * dPrevPt + B_curr.x * dCurrPt;
        let dP1_2d = B_prev.y * dPrevPt + B_curr.y * dCurrPt;
        let dP2_2d = B_prev.z * dPrevPt + B_curr.z * dCurrPt;
        let dP3_2d = B_prev.w * dPrevPt + B_curr.w * dCurrPt;

        // Backproject 2D gradients to 3D
        let dP0_3d = backproject_gradient(uniforms.vp, b.p0.xyz, aspect, dP0_2d);
        let dP1_3d = backproject_gradient(uniforms.vp, b.p1.xyz, aspect, dP1_2d);
        let dP2_3d = backproject_gradient(uniforms.vp, b.p2.xyz, aspect, dP2_2d);
        let dP3_3d = backproject_gradient(uniforms.vp, b.p3.xyz, aspect, dP3_2d);

        let base = i * 18u;
        atomicAdd(&grads.data[base + 0u], i32(dP0_3d.x * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 1u], i32(dP0_3d.y * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 2u], i32((dP0_3d.z + dDepth_dZs.x) * FP_SCALE_POS));
        
        atomicAdd(&grads.data[base + 3u], i32(dP1_3d.x * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 4u], i32(dP1_3d.y * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 5u], i32((dP1_3d.z + dDepth_dZs.y) * FP_SCALE_POS));
        
        atomicAdd(&grads.data[base + 6u], i32(dP2_3d.x * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 7u], i32(dP2_3d.y * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 8u], i32((dP2_3d.z + dDepth_dZs.z) * FP_SCALE_POS));
        
        atomicAdd(&grads.data[base + 9u], i32(dP3_3d.x * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 10u], i32(dP3_3d.y * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 11u], i32((dP3_3d.z + dDepth_dZs.w) * FP_SCALE_POS));
        
        atomicAdd(&grads.data[base + 12u], i32(dColor.r * FP_SCALE_COL));
        atomicAdd(&grads.data[base + 13u], i32(dColor.g * FP_SCALE_COL));
        atomicAdd(&grads.data[base + 14u], i32(dColor.b * FP_SCALE_COL));
        atomicAdd(&grads.data[base + 15u], i32(d_opacity * FP_SCALE_COL));
        
        atomicAdd(&grads.data[base + 16u], i32(dWidth * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 17u], i32(dSoft * FP_SCALE_POS));

        // Accumulate this bezier's contribution to the color loss for ADC pruning.
        let color_loss_contrib = dot(dC * dC, vec3f(1.0)) * (T_prev * a);
        adc.loss_accum[i] += color_loss_contrib;
    }
}
