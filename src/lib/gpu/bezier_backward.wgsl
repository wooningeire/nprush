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
    vp: mat4x4f,             // offset 0,  size 64
    mode: f32,               // offset 64, size 4
    max_width: f32,          // offset 68, size 4
    prune_alpha_thresh: f32, // offset 72, size 4
    prune_width_thresh: f32, // offset 76, size 4
    bg_penalty: f32,         // offset 80, size 4
    _pad0: f32,              // offset 84, size 4
    _pad1: f32,              // offset 88, size 4
    _pad2: f32,              // offset 92, size 4
    // total: 96 bytes
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
@group(0) @binding(8) var normalTex: texture_2d<f32>;

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
        let dt = t - 0.5;
        let pressure = 1.0 - 4.0 * dt * dt;
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

    let dC_raw = 2.0 * (C_pred - tgt_color);

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
    let dD_total = 0.0;
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
        let dt_pixel = t_pixel - 0.5;
        let pressure = 1.0 - 4.0 * dt_pixel * dt_pixel;
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

        // Depth gradient with respect to control points (z/w component)
        let B_pixel = bernstein(t_pixel);
        let dDepth_dZs = dD_total * (T_prev * a) * B_pixel;

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

        // 2. Direction regularization: align with direction of least/most color change.
        //    Tangent variant (commented out): loss = REG_DIR * (tangent · grad_color)^2
        //    Bitangent variant (active): loss = REG_DIR * (bitangent · grad_color)^2
        //    Bitangent = perp(tangent), so this pushes the curve to run *across* color
        //    gradients (i.e. along isocurves) rather than along them.
        let REG_DIR = 0.5;
        if (is_fine && len2 > 1e-10) {
            let tangent = seg / sqrt(len2); // seg = curr_pt - prev_pt
            // Bitangent: 90° CCW rotation of tangent in screen space
            //let bitangent = vec2f(-tangent.y, tangent.x);

            // Color gradient via central differences (1-pixel step in NDC)
            let px = vec2i(global_id.xy);
            let px_dims = vec2i(dims);
            let px_r = clamp(px + vec2i(1, 0), vec2i(0), px_dims - 1);
            let px_l = clamp(px - vec2i(1, 0), vec2i(0), px_dims - 1);
            let px_u = clamp(px + vec2i(0, 1), vec2i(0), px_dims - 1);
            let px_d = clamp(px - vec2i(0, 1), vec2i(0), px_dims - 1);
            let cr = dot(textureLoad(targetTex, px_r, 0).rgb, vec3f(0.333));
            let cl = dot(textureLoad(targetTex, px_l, 0).rgb, vec3f(0.333));
            let cu = dot(textureLoad(targetTex, px_u, 0).rgb, vec3f(0.333));
            let cd = dot(textureLoad(targetTex, px_d, 0).rgb, vec3f(0.333));
            let grad_x = (cr - cl) * 0.5 * aspect;
            let grad_y = -(cu - cd) * 0.5;
            let grad_color = vec2f(grad_x, grad_y);

            // Normal gradient via central differences
            let nr_scalar = dot(textureLoad(normalTex, px_r, 0).rgb, vec3f(0.333));
            let nl_scalar = dot(textureLoad(normalTex, px_l, 0).rgb, vec3f(0.333));
            let nu_scalar = dot(textureLoad(normalTex, px_u, 0).rgb, vec3f(0.333));
            let nd_scalar = dot(textureLoad(normalTex, px_d, 0).rgb, vec3f(0.333));
            let grad_norm_x = (nr_scalar - nl_scalar) * 0.5 * aspect;
            let grad_norm_y = -(nu_scalar - nd_scalar) * 0.5;
            let grad_normal = vec2f(grad_norm_x, grad_norm_y);

            // Flow vector combines color and normal gradients equally
            let flow_vec = grad_color + grad_normal;

            // --- Tangent influence (commented out) ---
            // loss = REG_DIR * (tangent · flow_vec)^2
            // let dir_vec = tangent;

            // --- Bitangent influence (active) ---
            // loss = REG_DIR * (bitangent · flow_vec)^2
            // Penalises the bitangent aligning with the flow vector,
            // which pushes the tangent to align with the flow instead —
            // i.e. the curve runs across isocurves (perpendicular to flat regions).
            let dir_vec = tangent;

            let tg = dot(dir_vec, flow_vec);
            let d_loss_dir = REG_DIR * 2.0 * tg;

            // d_loss/d_dir_vec = d_loss_dir * flow_vec
            // For bitangent = (-ty, tx): d_bitangent/d_tangent = [[ 0,-1],[1, 0]]
            // d_loss/d_tangent = d_loss/d_bitangent * d_bitangent/d_tangent
            //   = d_loss_dir * flow_vec * [[0,1],[-1,0]]
            //   = d_loss_dir * vec2f(flow_vec.y, -flow_vec.x)
            // For tangent variant just use: d_loss_dir * flow_vec directly.
            let d_tangent_vec = d_loss_dir * vec2f(flow_vec.y, -flow_vec.x); // bitangent chain rule
            // let d_tangent_vec = d_loss_dir * flow_vec; // tangent variant

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
}
