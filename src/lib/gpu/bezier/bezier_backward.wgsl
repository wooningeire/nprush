// Differentiable rasterization of cubic bezier curves.
// Now in 3D: curves are defined by 3D control points projected to the screen.

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

struct GradArray {
    data: array<atomic<i32>, {@NUM_BEZIER_PARAMS}u>,
}

struct BezierUniforms {
    vp: mat4x4f,
    mode: f32,
    max_width: f32,
    prune_alpha_thresh: f32,
    prune_width_thresh: f32,
    bg_penalty: f32,
    _pad0: f32,
    _pad1: f32,
    adc_period_steps: f32,
    optim_width: f32,
    optim_height: f32,
    _pad_res: vec2f,
    vp_inv: mat4x4f,
    cam_world: vec4f,
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
@group(0) @binding(7) var<storage, read_write> adc: ADCArray;
@group(0) @binding(8) var normalTex: texture_2d<f32>;
// Per-pixel residual loss map: accumulated as fixed-point i32 (scale 10000).
// ADC reads this to find high-loss regions and seeds new beziers there.
@group(0) @binding(9) var<storage, read_write> pixel_loss: array<atomic<i32>, {@PIXEL_LOSS_SIZE}u>;
@group(0) @binding(10) var<storage, read> instance_vals: array<u32>;
@group(0) @binding(11) var<storage, read> tile_starts: array<u32>;
@group(0) @binding(12) var<storage, read> tile_ends: array<u32>;

const N_SEG: u32 = {@BEZIER_POLY_SEG}u;
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

const SH_C1_B: f32 = 0.4886025119029199;

fn bezier_pos_world(b: Bezier, tt: f32) -> vec3f {
    let omt = 1.0 - tt;
    return omt*omt*omt*b.p0.xyz + 3.0*omt*omt*tt*b.p1.xyz + 3.0*omt*tt*tt*b.p2.xyz + tt*tt*tt*b.p3.xyz;
}

fn bezier_deriv_world(b: Bezier, tt: f32) -> vec3f {
    let omt = 1.0 - tt;
    return 3.0*omt*omt*(b.p1.xyz - b.p0.xyz) + 6.0*omt*tt*(b.p2.xyz - b.p1.xyz) + 3.0*tt*tt*(b.p3.xyz - b.p2.xyz);
}

fn bezier_dirs_sh(cam_xyz: vec3f, pos_w: vec3f, tang: vec3f) -> vec3f {
    let Vcam = normalize(cam_xyz - pos_w);
    var ez = tang;
    if (dot(ez, ez) < 1e-10) {
        ez = vec3f(1.0, 0.0, 0.0);
    } else {
        ez = normalize(ez);
    }
    var ex = cross(vec3f(0.0, 1.0, 0.0), ez);
    if (dot(ex, ex) < 1e-12) {
        ex = cross(vec3f(1.0, 0.0, 0.0), ez);
    }
    ex = normalize(ex);
    let ey = normalize(cross(ez, ex));
    return vec3f(dot(Vcam, ex), dot(Vcam, ey), dot(Vcam, ez));
}

fn bezier_rgb_linear_dl(b: Bezier, dl: vec3f) -> vec3f {
    let lx = dl.x;
    let ly = dl.y;
    let lz = dl.z;
    let rr = b.color.r + SH_C1_B * (ly*b.sh1_r.x + lz*b.sh1_r.y + lx*b.sh1_r.z);
    let gg = b.color.g + SH_C1_B * (ly*b.sh1_g.x + lz*b.sh1_g.y + lx*b.sh1_g.z);
    let bb = b.color.b + SH_C1_B * (ly*b.sh1_b.x + lz*b.sh1_b.y + lx*b.sh1_b.z);
    return vec3f(rr, gg, bb);
}

const TILE_CACHE_DIM: u32 = 20u;
const TILE_CACHE_SZ: u32 = 400u;
var<workgroup> tile_tgt_luma: array<f32, TILE_CACHE_SZ>;
var<workgroup> tile_tgt_gray: array<f32, TILE_CACHE_SZ>;
var<workgroup> tile_norm_scalar: array<f32, TILE_CACHE_SZ>;

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

    let local_idx = local_id.y * 16u + local_id.x;

    let grid_width = (dims.x + 15u) / 16u;
    let tile_id = workgroup_id.y * grid_width + workgroup_id.x;
    let start_idx = tile_starts[tile_id];
    let end_idx = tile_ends[tile_id];
    let bezier_count = end_idx - start_idx;

    // Cooperatively cache target luma / fine gray / normal scalar for ±2 neighborhood.
    let ox = i32(workgroup_id.x * 16u);
    let oy = i32(workgroup_id.y * 16u);
    let load_ox = ox - 2;
    let load_oy = oy - 2;
    let max_gx = i32(dims.x) - 1;
    let max_gy = i32(dims.y) - 1;
    let luma_w_cache = vec3f(0.2126, 0.7152, 0.0722);
    for (var ti = local_idx; ti < TILE_CACHE_SZ; ti = ti + 256u) {
        let lx = ti % TILE_CACHE_DIM;
        let ly = ti / TILE_CACHE_DIM;
        let gx = clamp(load_ox + i32(lx), 0, max_gx);
        let gy = clamp(load_oy + i32(ly), 0, max_gy);
        let rgb = textureLoad(targetTex, vec2u(u32(gx), u32(gy)), 0).rgb;
        tile_tgt_luma[ti] = dot(rgb, luma_w_cache);
        tile_tgt_gray[ti] = dot(rgb, vec3f(0.333));
        let nrgb = textureLoad(normalTex, vec2u(u32(gx), u32(gy)), 0).rgb;
        tile_norm_scalar[ti] = dot(nrgb, vec3f(0.333));
    }
    workgroupBarrier();

    // --- 2. PIXEL EVALUATION ---
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    let p = pixel_to_p(global_id.xy, dims, aspect);
    let tgt_color = textureLoad(targetTex, global_id.xy, 0).rgb;
    let tgt_depth = textureLoad(targetDepthTex, global_id.xy, 0).r;

    var C_pred = vec3f(0.0);
    var T_final = 1.0;

    for (var idx = 0u; idx < bezier_count; idx++) {
        let i = instance_vals[end_idx - 1u - idx];
        let b = beziers.items[i];
        
        let width = max(b.p0.w, 0.001);
        let softness = max(b.p1.w, 0.001);
        
        let proj0 = project_center(uniforms.vp, b.p0.xyz, aspect);
        let proj1 = project_center(uniforms.vp, b.p1.xyz, aspect);
        let proj2 = project_center(uniforms.vp, b.p2.xyz, aspect);
        let proj3 = project_center(uniforms.vp, b.p3.xyz, aspect);
        let p0 = proj0.xy;
        let p1 = proj1.xy;
        let p2 = proj2.xy;
        let p3 = proj3.xy;

        var min_d2 = 1e9;
        var min_k = 1u;
        var min_u = 0.0;
        var prev = p0;
        for (var k = 1u; k <= N_SEG; k = k + 1u) {
            let curr = bezier_at(p0, p1, p2, p3, f32(k) / f32(N_SEG));
            let seg = curr - prev;
            let len2 = max(dot(seg, seg), 1e-8);
            let u = clamp(dot(p - prev, seg) / len2, 0.0, 1.0);
            let proj = prev + u * seg;
            let diff = p - proj;
            let d2 = dot(diff, diff);
            if (d2 < min_d2) {
                min_d2 = d2;
                min_k = k;
                min_u = u;
            }
            prev = curr;
        }
        let min_d = sqrt(min_d2);
        let t = (f32(min_k - 1u) + min_u) / f32(N_SEG);
        let dt = t - 0.5;
        let pressure = 1.0 - 4.0 * dt * dt;

        let B = bernstein(t);
        let raw_w = dot(B, vec4f(proj0.z, proj1.z, proj2.z, proj3.z));
        let linear_w = max(raw_w, DEPTH_NEAR_BEZ);
        let inv_w = 1.0 / linear_w;

        let local_width = width * pressure * inv_w;
        let local_softness = softness * pressure * inv_w;

        let pos_w_f = bezier_pos_world(b, t);
        let tang_f = bezier_deriv_world(b, t);
        let dl_f = bezier_dirs_sh(uniforms.cam_world.xyz, pos_w_f, tang_f);
        let lx_f = dl_f.x;
        let ly_f = dl_f.y;
        let lz_f = dl_f.z;
        let o_lin_f = b.color.a + SH_C1_B * (ly_f * b.sh1_a.x + lz_f * b.sh1_a.y + lx_f * b.sh1_a.z);
        let opacity_f = clamp(o_lin_f, 0.0, 1.0);
        let local_opacity = opacity_f * pressure;

        let inner = local_width - local_softness;
        let outer = local_width + local_softness;
        let a_geom = 1.0 - smoothstep(inner, outer, min_d);
        var a = clamp(a_geom * local_opacity, 0.0, 0.999);



        let rgb_lin_f = bezier_rgb_linear_dl(b, dl_f);
        let rgb_vis_f = max(rgb_lin_f, vec3f(0.0));

        C_pred += T_final * a * rgb_vis_f;
        T_final *= (1.0 - a);
    }

    let background_sample = textureLoad(bgTex, global_id.xy, 0).rgb;
    var background = vec3f(0.0);
    let color_mode = uniforms.mode > 0.5;
    background = select(vec3f(0.0), background_sample, color_mode);

    C_pred += T_final * background;

    let dC_raw = 2.0 * (C_pred - tgt_color);


    // Luminance/contrast-weighted color loss.
    // 1. Decompose error into luminance and chrominance components.
    //    Luma weights (BT.709): Y = 0.2126 R + 0.7152 G + 0.0722 B
    let luma_w = vec3f(0.2126, 0.7152, 0.0722);
    let luma_err = dot(dC_raw * 0.5, luma_w); // signed luma error (before *2)
    // 2. Local contrast: magnitude of spatial color gradient at this pixel.
    //    High contrast → loss matters more; flat regions → down-weight.
    let gxi = i32(global_id.x);
    let gyi = i32(global_id.y);
    let ci = u32((gyi - load_oy) * i32(TILE_CACHE_DIM) + (gxi - load_ox));
    let luma_r = tile_tgt_luma[ci + 1u];
    let luma_l = tile_tgt_luma[ci - 1u];
    let luma_u = tile_tgt_luma[ci + TILE_CACHE_DIM];
    let luma_d = tile_tgt_luma[ci - TILE_CACHE_DIM];
    let contrast = sqrt((luma_r - luma_l) * (luma_r - luma_l) + (luma_u - luma_d) * (luma_u - luma_d));
    // Contrast weight: 1.0 baseline + boost in high-contrast areas, capped.
    let contrast_weight = 1.0 + clamp(contrast * 8.0, 0.0, 3.0);
    // 3. Luma-weighted gradient: luma channel gets 3x weight vs chroma.
    //    dC = dC_chroma + 3 * dC_luma_component
    let dC_luma = luma_err * luma_w * 6.0; // 2 (from MSE) * 3 (luma boost)
    let dC_chroma = dC_raw - dot(dC_raw, luma_w) * luma_w; // chroma residual
    let dC = (dC_luma + dC_chroma) * contrast_weight;

    // Edge mode: coverage loss driving total alpha to match the edge map.
    let EDGE_LOSS_WEIGHT = 2.0;
    let coverage = 1.0 - T_final;
    let edge_target = tgt_color.r;
    let d_coverage_edge = select(0.0, EDGE_LOSS_WEIGHT * 2.0 * (coverage - edge_target), uniforms.mode < 0.5);

    // Color mode: penalize opacity directly on background pixels (tgt_depth ≈ 1 = no geometry).
    // With reciprocal depth encoding, mesh surface pixels are well below 1.0 and
    // background (clear value) is exactly 1.0. Use a threshold that only catches
    // the true background clear value.
    let is_background = step(0.995, tgt_depth);

    let FP_SCALE_POS = f32({@BEZIER_FP_SCALE_POS});
    let FP_SCALE_COL = f32({@BEZIER_FP_SCALE_COL});

    // Direction regularization (fine layer): flow from target/normal texels depends only on this pixel.
    let is_fine = uniforms.max_width > 0.0;
    var dir_flow_dir = vec2f(0.0);
    var dir_flow_use = false;
    if (is_fine) {
        let cr = tile_tgt_gray[ci + 2u];
        let cl = tile_tgt_gray[ci - 2u];
        let cu = tile_tgt_gray[ci + 2u * TILE_CACHE_DIM];
        let cd = tile_tgt_gray[ci - 2u * TILE_CACHE_DIM];
        let grad_x = (cr - cl) * 0.25 * aspect;
        let grad_y = -(cu - cd) * 0.25;

        let nr_scalar = tile_norm_scalar[ci + 2u];
        let nl_scalar = tile_norm_scalar[ci - 2u];
        let nu_scalar = tile_norm_scalar[ci + 2u * TILE_CACHE_DIM];
        let nd_scalar = tile_norm_scalar[ci - 2u * TILE_CACHE_DIM];
        let grad_norm_x = (nr_scalar - nl_scalar) * 0.25 * aspect;
        let grad_norm_y = -(nu_scalar - nd_scalar) * 0.25;

        let flow_raw = vec2f(grad_x, grad_y) + 2.0 * vec2f(grad_norm_x, grad_norm_y);
        let flow_len = length(flow_raw);
        dir_flow_use = flow_len > 1e-4;
        dir_flow_dir = select(vec2f(0.0), flow_raw / flow_len, dir_flow_use);
    }

    var C_accum = vec3f(0.0);
    var T_accum = 1.0;

    for (var idx = 0u; idx < bezier_count; idx++) {
        let i = instance_vals[end_idx - 1u - idx];
        let b = beziers.items[i];
        let width = max(b.p0.w, 0.001);
        let softness = max(b.p1.w, 0.001);

        let proj0 = project_center(uniforms.vp, b.p0.xyz, aspect);
        let proj1 = project_center(uniforms.vp, b.p1.xyz, aspect);
        let proj2 = project_center(uniforms.vp, b.p2.xyz, aspect);
        let proj3 = project_center(uniforms.vp, b.p3.xyz, aspect);
        let p0 = proj0.xy;
        let p1 = proj1.xy;
        let p2 = proj2.xy;
        let p3 = proj3.xy;

        var min_d2 = 1e9;
        var min_k = 1u;
        var min_u = 0.0;
        var prev = p0;
        for (var k_idx = 1u; k_idx <= N_SEG; k_idx = k_idx + 1u) {
            let curr = bezier_at(p0, p1, p2, p3, f32(k_idx) / f32(N_SEG));
            let seg_curr = curr - prev;
            let len2_curr = max(dot(seg_curr, seg_curr), 1e-8);
            let u = clamp(dot(p - prev, seg_curr) / len2_curr, 0.0, 1.0);
            let proj = prev + u * seg_curr;
            let diff = p - proj;
            let d2 = dot(diff, diff);
            if (d2 < min_d2) {
                min_d2 = d2;
                min_k = k_idx;
                min_u = u;
            }
            prev = curr;
        }

        let k = min_k;
        let u_clamped = min_u;
        let t_prev = f32(k - 1u) / f32(N_SEG);
        let t_curr = f32(k) / f32(N_SEG);
        let prev_pt = bezier_at(p0, p1, p2, p3, t_prev);
        let curr_pt = bezier_at(p0, p1, p2, p3, t_curr);
        let seg = curr_pt - prev_pt;
        let len2 = max(dot(seg, seg), 1e-8);
        let proj = prev_pt + u_clamped * seg;
        let d_vec = p - proj;
        let d = max(length(d_vec), 1e-6);

        let t_geom = (f32(k - 1u) + u_clamped) / f32(N_SEG);
        let dt_pixel = t_geom - 0.5;
        let pressure = 1.0 - 4.0 * dt_pixel * dt_pixel;
        let B_pixel = bernstein(t_geom);

        let pos_w_b = bezier_pos_world(b, t_geom);
        let dl_b = bezier_dirs_sh(uniforms.cam_world.xyz, pos_w_b, bezier_deriv_world(b, t_geom));
        let lx_b = dl_b.x;
        let ly_b = dl_b.y;
        let lz_b = dl_b.z;
        let opacity_lin_b = b.color.a + SH_C1_B * (ly_b*b.sh1_a.x + lz_b*b.sh1_a.y + lx_b*b.sh1_a.z);
        let opacity = clamp(opacity_lin_b, 0.0, 1.0);
        
        let local_opacity_tmp = opacity * pressure;
        let raw_w_tmp = dot(B_pixel, vec4f(proj0.z, proj1.z, proj2.z, proj3.z));
        let linear_w_tmp = max(raw_w_tmp, DEPTH_NEAR_BEZ);
        let inv_w_tmp = 1.0 / linear_w_tmp;
        let local_width_tmp = width * pressure * inv_w_tmp;
        let local_softness_tmp = softness * pressure * inv_w_tmp;
        let inner_tmp = local_width_tmp - local_softness_tmp;
        let outer_tmp = local_width_tmp + local_softness_tmp;
        let a_geom_tmp = 1.0 - smoothstep(inner_tmp, outer_tmp, d);
        let a = clamp(a_geom_tmp * local_opacity_tmp, 0.0, 0.999);

        let rr_lin = b.color.r + SH_C1_B * (ly_b*b.sh1_r.x + lz_b*b.sh1_r.y + lx_b*b.sh1_r.z);
        let gg_lin = b.color.g + SH_C1_B * (ly_b*b.sh1_g.x + lz_b*b.sh1_g.y + lx_b*b.sh1_g.z);
        let bb_lin = b.color.b + SH_C1_B * (ly_b*b.sh1_b.x + lz_b*b.sh1_b.y + lx_b*b.sh1_b.z);
        let rgb_vis = max(vec3f(rr_lin, gg_lin, bb_lin), vec3f(0.0));
        let color = rgb_vis;

        let T_prev = T_accum;
        C_accum += T_prev * a * color;
        T_accum *= (1.0 - a);
        
        if (a < 0.001) { continue; }
        
        let clamp_gate_o = select(0.0, 1.0, opacity_lin_b > 1e-6 && opacity_lin_b < 1.0 - 1e-6);

        let dColor = dC * (T_prev * a);
        let inv_T = select(1.0 / T_accum, 0.0, T_accum < 1e-5);
        let C_rest = (C_pred - C_accum) * inv_T;
        // Edge loss dT goes backwards, but mathematically it acts just like background.
        // dT was initialized to `dot(dC, background) - d_coverage_edge`.
        // C_rest handles the `background` part nicely.
        // For the edge loss part, we can inject it as an additive term.
        // Actually, d_coverage_edge is just added to the initial dT.
        // So dT = dot(dC, C_rest) - d_coverage_edge.
        // Then da = dT * (-T_prev) + dot(dC, T_prev * color).
        let dT = dot(dC, C_rest) - d_coverage_edge;
        let da = dT * (-T_prev) + dot(dC, T_prev * color);

        let raw_w = dot(B_pixel, vec4f(proj0.z, proj1.z, proj2.z, proj3.z));
        let linear_w = max(raw_w, DEPTH_NEAR_BEZ);
        let inv_w = 1.0 / linear_w;

        let local_width = width * pressure * inv_w;
        let local_softness = softness * pressure * inv_w;
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
        // Weight is per-layer (0 = disabled for coarse bezier, >0 for fine bezier layer).
        let bg_opacity_penalty = uniforms.bg_penalty * is_background;
        let d_opacity_lin_only = da * a_geom * pressure * clamp_gate_o;
        var d_opacity = d_opacity_lin_only + bg_opacity_penalty;


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
        // dWidth and dSoft are gradients of the world-space width/softness parameters.
        // Since local_width = width * pressure * inv_w, we must chain inv_w through.
        dWidth = select(0.0, da_eff * smoothstep_deriv * pressure * inv_w / denom, in_softband);
        
        // d(x_inner)/d(softness): x_inner = (d - (width-softness)*p*inv_w) / (2*softness*p*inv_w)
        // Let W = width*p*inv_w, S = softness*p*inv_w. x_inner = (d - (W-S)) / (2*S)
        // d(x_inner)/d(softness) = d(x_inner)/dS * (p*inv_w)
        // d(x_inner)/dS = (1*(2S) - (d-(W-S))*2) / (4S^2) = (2S - 2d + 2W - 2S) / (4S^2) = (2W - 2d) / (4S^2)
        // = (W - d) / (2*S^2)
        let W = width * pressure * inv_w;
        let dx_ds = (W - d) / max(2.0 * softness * softness * pressure * inv_w, 1e-12);
        dSoft  = select(0.0, -da_eff * smoothstep_deriv * dx_ds * (pressure * inv_w), in_softband);

        let dProj = -dD * d_vec / d;
        let dPrevPt = (1.0 - u_clamped) * dProj;
        let dCurrPt = u_clamped * dProj;

        let B_prev = bernstein(t_prev);
        let B_curr = bernstein(t_curr);

        // --- Regularization (fine bezier layer only: max_width > 0) ---
        let base = i * {@BEZIER_PARAMS_PER}u;

        // 1. Softness → 0: loss = REG_SOFT * softness^2
        //    d_soft += REG_SOFT * 2 * softness
        let REG_SOFT = 5.0;
        dSoft += select(0.0, REG_SOFT * 2.0 * softness, is_fine);

        // 2. Direction regularization (flow_dir precomputed once per pixel above).
        const REG_DIR: f32 = 1.5;
        if (is_fine && dir_flow_use && len2 > 1e-10) {
            let tangent = seg / sqrt(len2);
            let flow_dir = dir_flow_dir;
            let tg = dot(tangent, flow_dir);
            let d_loss_dir = REG_DIR * 2.0 * tg;
            let d_tangent_vec = d_loss_dir * flow_dir;
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
        atomicAdd(&grads.data[base + 2u], i32(dP0_3d.z * FP_SCALE_POS));
        
        atomicAdd(&grads.data[base + 3u], i32(dP1_3d.x * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 4u], i32(dP1_3d.y * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 5u], i32(dP1_3d.z * FP_SCALE_POS));
        
        atomicAdd(&grads.data[base + 6u], i32(dP2_3d.x * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 7u], i32(dP2_3d.y * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 8u], i32(dP2_3d.z * FP_SCALE_POS));
        
        atomicAdd(&grads.data[base + 9u], i32(dP3_3d.x * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 10u], i32(dP3_3d.y * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 11u], i32(dP3_3d.z * FP_SCALE_POS));

        let d_relu_r = select(0.0, dColor.r, rr_lin > 0.0);
        let d_relu_g = select(0.0, dColor.g, gg_lin > 0.0);
        let d_relu_b = select(0.0, dColor.b, bb_lin > 0.0);

        atomicAdd(&grads.data[base + 12u], i32(d_relu_r * FP_SCALE_COL));
        atomicAdd(&grads.data[base + 13u], i32(d_relu_g * FP_SCALE_COL));
        atomicAdd(&grads.data[base + 14u], i32(d_relu_b * FP_SCALE_COL));
        atomicAdd(&grads.data[base + 15u], i32(d_opacity * FP_SCALE_COL));
        
        atomicAdd(&grads.data[base + 16u], i32(dWidth * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 17u], i32(dSoft * FP_SCALE_POS));

        // ∂rgb_lin/∂sh matches bezier_rgb_linear_dl: coef (x,y,z) maps to (ly, lz, lx) * SH_C1.
        let k_r = SH_C1_B * d_relu_r * FP_SCALE_COL;
        let k_g = SH_C1_B * d_relu_g * FP_SCALE_COL;
        let k_b = SH_C1_B * d_relu_b * FP_SCALE_COL;

        atomicAdd(&grads.data[base + 18u], i32(ly_b * k_r));
        atomicAdd(&grads.data[base + 19u], i32(lz_b * k_r));
        atomicAdd(&grads.data[base + 20u], i32(lx_b * k_r));
        atomicAdd(&grads.data[base + 21u], i32(ly_b * k_g));
        atomicAdd(&grads.data[base + 22u], i32(lz_b * k_g));
        atomicAdd(&grads.data[base + 23u], i32(lx_b * k_g));
        atomicAdd(&grads.data[base + 24u], i32(ly_b * k_b));
        atomicAdd(&grads.data[base + 25u], i32(lz_b * k_b));
        atomicAdd(&grads.data[base + 26u], i32(lx_b * k_b));

        let k_o = SH_C1_B * d_opacity_lin_only * FP_SCALE_COL;
        atomicAdd(&grads.data[base + 27u], i32(ly_b * k_o));
        atomicAdd(&grads.data[base + 28u], i32(lz_b * k_o));
        atomicAdd(&grads.data[base + 29u], i32(lx_b * k_o));

        // Accumulate this bezier's contribution to the color loss for ADC pruning.
        let color_loss_contrib = dot(dC * dC, vec3f(1.0)) * (T_prev * a);
        adc.loss_accum[i] += color_loss_contrib;
    }

    // Accumulate per-pixel residual loss for ADC seeding.
    // Use the uncovered MSE: pixels with high transmittance (no bezier covers them)
    // and high color error are the best candidates for new bezier placement.
    let residual = dot(dC_raw * 0.5, dC_raw * 0.5) * T_final;
    let px_idx = global_id.y * dims.x + global_id.x;
    let FP_LOSS = 10000.0;
    atomicAdd(&pixel_loss[px_idx], i32(residual * FP_LOSS));
}
