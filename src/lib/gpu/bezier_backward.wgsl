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
    items: array<Bezier, {@NUM_BEZIERS}u>,
}

struct GradArray {
    data: array<atomic<i32>, {@NUM_BEZIER_PARAMS}u>,
}

struct BezierUniforms {
    vp: mat4x4f,             // offset 0,   size 64
    mode: f32,               // offset 64,  size 4
    max_width: f32,          // offset 68,  size 4
    prune_alpha_thresh: f32, // offset 72,  size 4
    prune_width_thresh: f32, // offset 76,  size 4
    bg_penalty: f32,         // offset 80,  size 4
    _pad0: f32,              // offset 84,  size 4
    _pad1: f32,              // offset 88,  size 4
    _pad2: f32,              // offset 92,  size 4
    vp_inv: mat4x4f,         // offset 96,  size 64
    // total: 160 bytes
}

struct ADCArray {
    grad_accum: array<f32, {@NUM_BEZIERS}u>,
    loss_accum: array<f32, {@NUM_BEZIERS}u>,
}

@group(0) @binding(0) var<storage, read> beziers: BezierArray;
@group(0) @binding(1) var<storage, read_write> grads: GradArray;
@group(0) @binding(2) var targetTex: texture_2d<f32>;
@group(0) @binding(3) var targetDepthTex: texture_2d<f32>;
@group(0) @binding(4) var<uniform> uniforms: BezierUniforms;
@group(0) @binding(5) var bgTex: texture_2d<f32>;
@group(0) @binding(6) var bgDepthTex: texture_2d<f32>;
@group(0) @binding(7) var<storage, read_write> adc: ADCArray;
@group(0) @binding(8) var normalTex: texture_2d<f32>;
// Per-pixel residual loss map: accumulated as fixed-point i32 (scale 10000).
// ADC reads this to find high-loss regions and seeds new beziers there.
@group(0) @binding(9) var<storage, read_write> pixel_loss: array<atomic<i32>, {@PIXEL_LOSS_SIZE}u>;

const N_SEG: u32 = 16u;
// Reciprocal depth near-plane constant — must match mesh.wgsl and splat_forward.wgsl.
const DEPTH_NEAR_BEZ: f32 = 0.1;

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

const MAX_TILE_BEZIERS = {@BEZIER_MAX_TILE_BEZIERS}u;
var<workgroup> tile_mask: array<atomic<u32>, {@NUM_BEZIERS_DIV_32}u>;
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
    
    for (var i = local_idx; i < {@NUM_BEZIERS_DIV_32}; i += 256u) {
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
    
    for (var bezier_id = local_idx; bezier_id < {@NUM_BEZIERS}; bezier_id += 256u) {
        let b = beziers.items[bezier_id];
        if (b.color.a < {@BEZIER_KILL_ALPHA_THRESH}) { continue; }
        
        let width = max(b.p0.w, 0.001);
        let softness = max(b.p1.w, 0.001);
        
        let proj0 = project_center(uniforms.vp, b.p0.xyz, aspect);
        let proj1 = project_center(uniforms.vp, b.p1.xyz, aspect);
        let proj2 = project_center(uniforms.vp, b.p2.xyz, aspect);
        let proj3 = project_center(uniforms.vp, b.p3.xyz, aspect);
        // Skip if any control point is behind the camera — same rule as the
        // forward pass so tile binning stays consistent with what gets rendered.
        if (proj0.z <= 0.0 || proj1.z <= 0.0 || proj2.z <= 0.0 || proj3.z <= 0.0) { continue; }

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
        for (var word_idx = 0u; word_idx < {@NUM_BEZIERS_DIV_32}; word_idx++) {
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
        let dt = t - 0.5;
        let pressure = 1.0 - 4.0 * dt * dt;
        let local_width = width * pressure;
        let local_softness = softness * pressure;
        let local_opacity = opacity * pressure;

        let inner = local_width - local_softness;
        let outer = local_width + local_softness;
        let a_geom = 1.0 - smoothstep(inner, outer, min_d);
        var a = clamp(a_geom * local_opacity, 0.0, 0.999);

        // Interpolate depth (w-component of projected points) and apply the same
        // reciprocal encoding used by mesh.wgsl / splat_forward.wgsl so that
        // D_pred is in the same [0,1) space as tgt_depth.
        let B = bernstein(t);
        let raw_w = dot(B, vec4f(proj0.z, proj1.z, proj2.z, proj3.z));
        let linear_w = max(raw_w, DEPTH_NEAR_BEZ);
        let d_val = clamp(1.0 - DEPTH_NEAR_BEZ / linear_w, 0.0, 1.0);

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

    let dC_raw = 2.0 * (C_pred - tgt_color);

    // Depth loss in color mode: constrain beziers to lie on the mesh surface.
    // Only applied on foreground pixels (tgt_depth < 0.99) to avoid fighting
    // the color loss on background. Weight is small so color dominates.
    let DEPTH_LOSS_WEIGHT = 0.5;
    let is_foreground_bez = tgt_depth < 0.99;
    let dD_depth = select(0.0, DEPTH_LOSS_WEIGHT * 2.0 * (D_pred - tgt_depth), is_foreground_bez && color_mode);

    // Luminance/contrast-weighted color loss.
    // 1. Decompose error into luminance and chrominance components.
    //    Luma weights (BT.709): Y = 0.2126 R + 0.7152 G + 0.0722 B
    let luma_w = vec3f(0.2126, 0.7152, 0.0722);
    let luma_err = dot(dC_raw * 0.5, luma_w); // signed luma error (before *2)
    // 2. Local contrast: magnitude of spatial color gradient at this pixel.
    //    High contrast → loss matters more; flat regions → down-weight.
    let px_i = vec2i(global_id.xy);
    let px_dims_i = vec2i(dims);
    let px_r2 = clamp(px_i + vec2i(1, 0), vec2i(0), px_dims_i - 1);
    let px_l2 = clamp(px_i - vec2i(1, 0), vec2i(0), px_dims_i - 1);
    let px_u2 = clamp(px_i + vec2i(0, 1), vec2i(0), px_dims_i - 1);
    let px_d2 = clamp(px_i - vec2i(0, 1), vec2i(0), px_dims_i - 1);
    let luma_r = dot(textureLoad(targetTex, px_r2, 0).rgb, luma_w);
    let luma_l = dot(textureLoad(targetTex, px_l2, 0).rgb, luma_w);
    let luma_u = dot(textureLoad(targetTex, px_u2, 0).rgb, luma_w);
    let luma_d = dot(textureLoad(targetTex, px_d2, 0).rgb, luma_w);
    let contrast = sqrt((luma_r - luma_l) * (luma_r - luma_l) + (luma_u - luma_d) * (luma_u - luma_d));
    // Contrast weight: 1.0 baseline + boost in high-contrast areas, capped.
    let contrast_weight = 1.0 + clamp(contrast * 8.0, 0.0, 3.0);
    // 3. Luma-weighted gradient: luma channel gets 3x weight vs chroma.
    //    dC = dC_chroma + 3 * dC_luma_component
    let dC_luma = luma_err * luma_w * 6.0; // 2 (from MSE) * 3 (luma boost)
    let dC_chroma = dC_raw - dot(dC_raw, luma_w) * luma_w; // chroma residual
    let dC = (dC_luma + dC_chroma) * contrast_weight;
    // let dD_total = 2.0 * (D_pred - tgt_depth);
    let dD_total = dD_depth;
    var dT = dot(dC, background) + dD_total * bg_depth;

    // Edge mode: coverage loss driving total alpha to match the edge map.
    let EDGE_LOSS_WEIGHT = 2.0;
    let coverage = 1.0 - Ts[bezier_count];
    let edge_target = tgt_color.r;
    let d_coverage_edge = select(0.0, EDGE_LOSS_WEIGHT * 2.0 * (coverage - edge_target), uniforms.mode < 0.5);
    dT += -d_coverage_edge;

    // Color mode: penalize opacity directly on background pixels (tgt_depth ≈ 1 = no geometry).
    // With reciprocal depth encoding, mesh surface pixels are well below 1.0 and
    // background (clear value) is exactly 1.0. Use a threshold that only catches
    // the true background clear value.
    let is_background = step(0.995, tgt_depth);

    let FP_SCALE_POS = f32({@BEZIER_FP_SCALE_POS});
    let FP_SCALE_COL = f32({@BEZIER_FP_SCALE_COL});

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
        let dt_pixel = t_pixel - 0.5;
        let pressure = 1.0 - 4.0 * dt_pixel * dt_pixel;
        let B_pixel = bernstein(t_pixel);
        let local_width = width * pressure;
        let local_softness = softness * pressure;
        let local_opacity = opacity * pressure;

        let inner = local_width - local_softness;
        let outer = local_width + local_softness;
        let denom = max(outer - inner, 1e-6);
        let x_inner = clamp((d - inner) / denom, 0.0, 1.0);
        let smoothstep_deriv = 6.0 * x_inner * (1.0 - x_inner) / denom;
        let in_softband = (d > inner) && (d < outer);

        // da/d(opacity): chain through a = a_geom * local_opacity * pressure
        let a_geom = 1.0 - smoothstep(inner, outer, d);
        // Direct background penalty: push opacity to zero on background pixels.
        // Weight is per-layer (0 = disabled for base color, >0 for fine color layer).
        let bg_opacity_penalty = uniforms.bg_penalty * is_background;
        var d_opacity = da * a_geom * pressure + bg_opacity_penalty;

        // da/d(d): chain through smoothstep
        // da/d(width) and da/d(softness): chain through inner/outer
        // inner = (width - softness)*pressure, outer = (width + softness)*pressure
        // d(inner)/d(width) = pressure, d(outer)/d(width) = pressure
        // d(inner)/d(softness) = -pressure, d(outer)/d(softness) = pressure
        var dD = 0.0;
        var dWidth = 0.0;
        var dSoft = 0.0;
        let da_eff = da * local_opacity;
        dD     = select(0.0, -da_eff * smoothstep_deriv, in_softband);
        // d(smoothstep)/d(width) = smoothstep_deriv * d(x_inner)/d(width)
        // x_inner = (d - inner)/denom, denom = outer - inner = 2*softness*pressure
        // d(x_inner)/d(width) = (-d(inner)/d(width)*denom - (d-inner)*d(denom)/d(width)) / denom^2
        // d(inner)/d(width)=pressure, d(denom)/d(width)=0 => d(x_inner)/d(width) = -pressure/denom
        // => d(smoothstep)/d(width) = smoothstep_deriv * (-pressure/denom)
        // => da/d(width) = -da_eff * smoothstep_deriv * pressure / denom
        dWidth = select(0.0, da_eff * smoothstep_deriv * pressure / denom, in_softband);
        // d(x_inner)/d(softness): inner=-softness*pressure, outer=+softness*pressure
        // denom=2*softness*pressure, d(inner)/d(s)=-pressure, d(denom)/d(s)=2*pressure
        // d(x_inner)/d(s) = (pressure*denom - (d-inner)*2*pressure) / denom^2
        //                 = pressure*(denom - 2*(d-inner)) / denom^2
        //                 = pressure*(2*softness*pressure - 2*(d-inner)) / denom^2
        let d_minus_inner = clamp(d - inner, 0.0, denom);
        let dx_ds = pressure * (denom - 2.0 * d_minus_inner) / max(denom * denom, 1e-12);
        dSoft  = select(0.0, -da_eff * smoothstep_deriv * dx_ds, in_softband);

        // Depth gradient with respect to control points (z/w component).
        // d_val = 1 - DEPTH_NEAR / raw_w  =>  d(d_val)/d(raw_w) = DEPTH_NEAR / raw_w²
        // raw_w = dot(B, [w0,w1,w2,w3])   =>  d(raw_w)/d(wi) = B[i]
        // Chain: d_loss/d(wi) = dD_total * T_prev * a * d(d_val)/d(raw_w) * B[i]
        // Re-project to get the w (clip.w) for each control point — .z from project_center is clip.w.
        let w0 = project_center(uniforms.vp, b.p0.xyz, aspect).z;
        let w1 = project_center(uniforms.vp, b.p1.xyz, aspect).z;
        let w2 = project_center(uniforms.vp, b.p2.xyz, aspect).z;
        let w3 = project_center(uniforms.vp, b.p3.xyz, aspect).z;
        let raw_w_bwd = dot(B_pixel, vec4f(w0, w1, w2, w3));
        let linear_w_bwd = max(raw_w_bwd, DEPTH_NEAR_BEZ);
        let d_dval_d_w = DEPTH_NEAR_BEZ / (linear_w_bwd * linear_w_bwd);
        let dDepth_dZs = dD_total * (T_prev * a) * d_dval_d_w * B_pixel;

        let dProj = -dD * d_vec / d;
        let dPrevPt = (1.0 - u_clamped) * dProj;
        let dCurrPt = u_clamped * dProj;

        let B_prev = bernstein(t_prev);
        let B_curr = bernstein(t_curr);

        // --- Regularization (fine color layer only: max_width > 0) ---
        let is_fine = uniforms.max_width > 0.0;
        let base = i * 18u;

        // 1. Softness → 0: loss = REG_SOFT * softness^2
        //    d_soft += REG_SOFT * 2 * softness
        let REG_SOFT = 5.0;
        dSoft += select(0.0, REG_SOFT * 2.0 * softness, is_fine);

        // 2. Direction regularization: align tangent with the local flow field.
        //    Flow field = normalized combination of color and normal gradients.
        //    Normal gradient is weighted 2x — it gives cleaner directional signal
        //    on 3D surfaces than color alone.
        //    loss = REG_DIR * (1 - (tangent · flow_dir)^2)
        //    which is minimised when tangent is parallel to flow_dir.
        const REG_DIR: f32 = 1.5;
        if (is_fine && len2 > 1e-10) {
            let tangent = seg / sqrt(len2); // seg = curr_pt - prev_pt

            // Color gradient via 2-pixel central differences for a smoother field
            let px = vec2i(global_id.xy);
            let px_dims = vec2i(dims);
            let px_r = clamp(px + vec2i(2, 0), vec2i(0), px_dims - 1);
            let px_l = clamp(px - vec2i(2, 0), vec2i(0), px_dims - 1);
            let px_u = clamp(px + vec2i(0, 2), vec2i(0), px_dims - 1);
            let px_d = clamp(px - vec2i(0, 2), vec2i(0), px_dims - 1);
            let cr = dot(textureLoad(targetTex, px_r, 0).rgb, vec3f(0.333));
            let cl = dot(textureLoad(targetTex, px_l, 0).rgb, vec3f(0.333));
            let cu = dot(textureLoad(targetTex, px_u, 0).rgb, vec3f(0.333));
            let cd = dot(textureLoad(targetTex, px_d, 0).rgb, vec3f(0.333));
            let grad_x = (cr - cl) * 0.25 * aspect; // 1/(2*step=4)
            let grad_y = -(cu - cd) * 0.25;
            let grad_color = vec2f(grad_x, grad_y);

            // Normal gradient via 2-pixel central differences (weighted 2x)
            let nr_scalar = dot(textureLoad(normalTex, px_r, 0).rgb, vec3f(0.333));
            let nl_scalar = dot(textureLoad(normalTex, px_l, 0).rgb, vec3f(0.333));
            let nu_scalar = dot(textureLoad(normalTex, px_u, 0).rgb, vec3f(0.333));
            let nd_scalar = dot(textureLoad(normalTex, px_d, 0).rgb, vec3f(0.333));
            let grad_norm_x = (nr_scalar - nl_scalar) * 0.25 * aspect;
            let grad_norm_y = -(nu_scalar - nd_scalar) * 0.25;
            let grad_normal = vec2f(grad_norm_x, grad_norm_y);

            // Flow vector: normals weighted 2x for stronger surface-following signal.
            // Normalize so REG_DIR strength is independent of gradient magnitude.
            let flow_raw = grad_color + 2.0 * grad_normal;
            let flow_len = length(flow_raw);
            // Only apply when there's a meaningful gradient; skip flat regions.
            if (flow_len > 1e-4) {
                let flow_dir = flow_raw / flow_len;

                // Penalise tangent aligning with flow_dir (i.e. crossing color/normal
                // boundaries). Minimised when tangent is orthogonal to flow_dir —
                // the stroke runs *along* isocurves, parallel to the surface flow.
                // loss = REG_DIR * (tangent · flow_dir)^2
                let tg = dot(tangent, flow_dir);
                let d_loss_dir = REG_DIR * 2.0 * tg;

                // d_loss/d_tangent = d_loss_dir * flow_dir
                let d_tangent_vec = d_loss_dir * flow_dir;

                // d_tangent/d_seg: tangent = seg / |seg|
                let inv_len = 1.0 / sqrt(len2);
                let d_seg = (d_tangent_vec - tangent * dot(d_tangent_vec, tangent)) * inv_len;

                let dPrevPt_dir = -d_seg;
                let dCurrPt_dir =  d_seg;

                let dP0_dir = B_prev.x * dPrevPt_dir + B_curr.x * dCurrPt_dir;
                let dP1_dir = B_prev.y * dPrevPt_dir + B_curr.y * dCurrPt_dir;
                let dP2_dir = B_prev.z * dPrevPt_dir + B_curr.z * dCurrPt_dir;
                let dP3_dir = B_prev.w * dPrevPt_dir + B_curr.w * dCurrPt_dir;

                let dP0_dir3 = backproject_gradient(uniforms.vp, b.p0.xyz, aspect, dP0_dir);
                let dP1_dir3 = backproject_gradient(uniforms.vp, b.p1.xyz, aspect, dP1_dir);
                let dP2_dir3 = backproject_gradient(uniforms.vp, b.p2.xyz, aspect, dP2_dir);
                let dP3_dir3 = backproject_gradient(uniforms.vp, b.p3.xyz, aspect, dP3_dir);

                atomicAdd(&grads.data[base + 0u], i32(dP0_dir3.x * FP_SCALE_POS));
                atomicAdd(&grads.data[base + 1u], i32(dP0_dir3.y * FP_SCALE_POS));
                atomicAdd(&grads.data[base + 2u], i32(dP0_dir3.z * FP_SCALE_POS));
                atomicAdd(&grads.data[base + 3u], i32(dP1_dir3.x * FP_SCALE_POS));
                atomicAdd(&grads.data[base + 4u], i32(dP1_dir3.y * FP_SCALE_POS));
                atomicAdd(&grads.data[base + 5u], i32(dP1_dir3.z * FP_SCALE_POS));
                atomicAdd(&grads.data[base + 6u], i32(dP2_dir3.x * FP_SCALE_POS));
                atomicAdd(&grads.data[base + 7u], i32(dP2_dir3.y * FP_SCALE_POS));
                atomicAdd(&grads.data[base + 8u], i32(dP2_dir3.z * FP_SCALE_POS));
                atomicAdd(&grads.data[base + 9u], i32(dP3_dir3.x * FP_SCALE_POS));
                atomicAdd(&grads.data[base + 10u], i32(dP3_dir3.y * FP_SCALE_POS));
                atomicAdd(&grads.data[base + 11u], i32(dP3_dir3.z * FP_SCALE_POS));
            }
        }



        let dP0_2d = B_prev.x * dPrevPt + B_curr.x * dCurrPt;
        let dP1_2d = B_prev.y * dPrevPt + B_curr.y * dCurrPt;
        let dP2_2d = B_prev.z * dPrevPt + B_curr.z * dCurrPt;
        let dP3_2d = B_prev.w * dPrevPt + B_curr.w * dCurrPt;

        // Backproject 2D gradients to 3D
        let dP0_3d = backproject_gradient(uniforms.vp, b.p0.xyz, aspect, dP0_2d);
        let dP1_3d = backproject_gradient(uniforms.vp, b.p1.xyz, aspect, dP1_2d);
        let dP2_3d = backproject_gradient(uniforms.vp, b.p2.xyz, aspect, dP2_2d);
        let dP3_3d = backproject_gradient(uniforms.vp, b.p3.xyz, aspect, dP3_2d);

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

    // Accumulate per-pixel residual loss for ADC seeding.
    // Use the uncovered MSE: pixels with high transmittance (no bezier covers them)
    // and high color error are the best candidates for new bezier placement.
    let residual = dot(dC_raw * 0.5, dC_raw * 0.5) * Ts[bezier_count];
    let px_idx = global_id.y * dims.x + global_id.x;
    let FP_LOSS = 10000.0;
    atomicAdd(&pixel_loss[px_idx], i32(residual * FP_LOSS));
}
