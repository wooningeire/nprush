struct Splat {
    pos_sx: vec4f,
    color: vec4f,
    quat: vec4f,
    sy_shape: vec4f,
}

struct SplatArray {
    splats: array<Splat, NUM_SPLATS>,
}

struct GradArray {
    data: array<atomic<i32>, NUM_PARAMS>,
}

struct SplatUniforms {
    vp: mat4x4f,
    blur_enabled: f32,
    _pad: vec3f,
}

@group(0) @binding(0) var<storage, read> splats: SplatArray;
@group(0) @binding(1) var<storage, read_write> grads: GradArray;
@group(0) @binding(2) var targetTex: texture_2d<f32>;
@group(0) @binding(3) var targetDepthTex: texture_2d<f32>;
@group(0) @binding(4) var<uniform> splat_uniforms: SplatUniforms;

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

fn quat_rotate(q: vec4f, v: vec3f) -> vec3f {
    let t = 2.0 * cross(q.yzw, v);
    return v + q.x * t + cross(q.yzw, t);
}

fn project_center(vp: mat4x4f, pos3: vec3f, aspect: f32) -> vec3f {
    let clip = vp * vec4f(pos3, 1.0);
    return vec3f(clip.x / clip.w * aspect, clip.y / clip.w, clip.w);
}

fn project_axis(vp: mat4x4f, ax_world: vec3f, clip_xy: vec2f, w: f32, aspect: f32) -> vec2f {
    let ac = vp * vec4f(ax_world, 0.0);
    return vec2f(
        (ac.x * w - clip_xy.x * ac.w) / (w * w) * aspect,
        (ac.y * w - clip_xy.y * ac.w) / (w * w)
    );
}

struct ProjectedSplat {
    screen: vec2f,
    sp: vec2f,
    r: f32,
    m00: f32, m01: f32, m10: f32, m11: f32,
    inv_det: f32,
    ax_screen: vec2f,
    ay_screen: vec2f,
    d: vec2f,
    w: f32,
    clip_xy: vec2f,
}

fn eval_splat(s: Splat, p: vec2f, aspect: f32) -> ProjectedSplat {
    var ps: ProjectedSplat;
    let clip = splat_uniforms.vp * vec4f(s.pos_sx.xyz, 1.0);
    ps.w = clip.w;
    ps.clip_xy = vec2f(clip.x, clip.y);
    ps.screen = vec2f(clip.x / clip.w * aspect, clip.y / clip.w);
    ps.d = p - ps.screen;
    let q = s.quat;
    let sx = max(s.pos_sx.w, 0.0001);
    let sy = max(s.sy_shape.x, 0.0001);
    let ax_w = quat_rotate(q, vec3f(1.0, 0.0, 0.0));
    let ay_w = quat_rotate(q, vec3f(0.0, 1.0, 0.0));
    ps.ax_screen = project_axis(splat_uniforms.vp, ax_w, ps.clip_xy, ps.w, aspect);
    ps.ay_screen = project_axis(splat_uniforms.vp, ay_w, ps.clip_xy, ps.w, aspect);
    ps.m00 = ps.ax_screen.x * sx; ps.m10 = ps.ax_screen.y * sx;
    ps.m01 = ps.ay_screen.x * sy; ps.m11 = ps.ay_screen.y * sy;
    let det = ps.m00 * ps.m11 - ps.m01 * ps.m10;
    ps.inv_det = select(1.0 / det, 0.0, abs(det) < 1e-10);
    ps.sp = vec2f(
        ( ps.m11 * ps.d.x - ps.m01 * ps.d.y) * ps.inv_det,
        (-ps.m10 * ps.d.x + ps.m00 * ps.d.y) * ps.inv_det
    );
    ps.r = max(length(ps.sp), 0.0001);
    return ps;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3u, @builtin(workgroup_id) workgroup_id: vec3u, @builtin(local_invocation_id) local_id: vec3u) {
    let dims = textureDimensions(targetTex);
    let aspect = f32(dims.x) / f32(dims.y);

    let local_idx = local_id.y * 16u + local_id.x;
    for (var i = local_idx; i < NUM_SPLATS_DIV_32; i += 256u) {
        atomicStore(&tile_mask[i], 0u);
    }
    if (local_idx == 0u) { atomicStore(&tile_splat_count, 0u); }
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
        let proj = project_center(splat_uniforms.vp, s.pos_sx.xyz, aspect);
        if (proj.z < 0.0) { continue; }
        let sx = max(s.pos_sx.w, 0.0001);
        let sy = max(s.sy_shape.x, 0.0001);
        let safe_sb = max(s.sy_shape.z, 0.0001);
        let R = pow(15.0 / safe_sb, 1.0 / s.sy_shape.y);
        let q = s.quat;
        let ax_w = quat_rotate(q, vec3f(1.0, 0.0, 0.0));
        let ay_w = quat_rotate(q, vec3f(0.0, 1.0, 0.0));
        let clip = splat_uniforms.vp * vec4f(s.pos_sx.xyz, 1.0);
        let w = clip.w;
        let clip_xy = vec2f(clip.x, clip.y);
        let ax_s = project_axis(splat_uniforms.vp, ax_w, clip_xy, w, aspect);
        let ay_s = project_axis(splat_uniforms.vp, ay_w, clip_xy, w, aspect);
        let max_r = R * max(length(ax_s) * sx, length(ay_s) * sy);
        let sc = proj.xy;
        let smin = sc - vec2f(max_r);
        let smax = sc + vec2f(max_r);
        if (!(smin.x > tile_max_p.x || smax.x < tile_min_p.x || smin.y > tile_max_p.y || smax.y < tile_min_p.y)) {
            let wi = splat_id / 32u;
            let bi = splat_id % 32u;
            atomicOr(&tile_mask[wi], 1u << bi);
        }
    }
    workgroupBarrier();

    if (local_idx == 0u) {
        var count = 0u;
        for (var wi = 0u; wi < NUM_SPLATS_DIV_32; wi++) {
            var word = atomicLoad(&tile_mask[wi]);
            while (word != 0u) {
                let bi = countTrailingZeros(word);
                if (count < MAX_TILE_SPLATS) { tile_splats[count] = wi * 32u + bi; count++; }
                word ^= (1u << bi);
            }
        }
        atomicStore(&tile_splat_count, count);
    }
    workgroupBarrier();

    let splat_count = atomicLoad(&tile_splat_count);
    if (global_id.x >= dims.x || global_id.y >= dims.y) { return; }

    let p = pixel_to_p(global_id.xy, dims, aspect);
    let tgt_color = textureLoad(targetTex, global_id.xy, 0).rgb;
    let tgt_depth = textureLoad(targetDepthTex, global_id.xy, 0).r;

    var alphas = array<f32, MAX_TILE_SPLATS>();
    var Ts = array<f32, MAX_TILE_SPLATS + 1u>();
    Ts[0] = 1.0;
    var C_pred = vec3f(0.0);
    var D_pred = 0.0;

    // Reciprocal depth encoding matching mesh.wgsl: 1 - DEPTH_NEAR / depth
    // This gives high precision for nearby objects stored in the 8-bit depth texture.
    const DEPTH_NEAR = 0.1;

    for (var idx = 0u; idx < splat_count; idx++) {
        let i = tile_splats[idx];
        let s = splats.splats[i];
        let ps = eval_splat(s, p, aspect);
        let shape_a = s.sy_shape.y;
        let shape_b = s.sy_shape.z;
        let power = -shape_b * pow(ps.r, shape_a);
        let a = clamp(select(0.0, exp(power) * s.color.a, power > -15.0), 0.0, 0.999);
        alphas[idx] = a;
        C_pred += Ts[idx] * a * s.color.rgb;
        let linear_depth = max(ps.w, DEPTH_NEAR);
        let depth = clamp(1.0 - DEPTH_NEAR / linear_depth, 0.0, 1.0);
        D_pred += Ts[idx] * a * depth;
        Ts[idx+1] = Ts[idx] * (1.0 - a);
    }

    let background = vec3f(0.05);
    C_pred += Ts[splat_count] * background;
    D_pred += Ts[splat_count] * 1.0;
    
    let dC = 2.0 * (C_pred - tgt_color);
    // Only apply depth loss on foreground pixels — background depth (1.0) would
    // fight the color loss and prevent splats from covering the model.
    // With reciprocal encoding, mesh surface pixels are well below 1.0.
    let is_foreground = tgt_depth < 0.99;
    let dD = select(0.0, 2.0 * (D_pred - tgt_depth), is_foreground);
    var dT_C = dot(dC, background);
    var dT_D = dD * 1.0;

    for (var j = 0u; j < splat_count; j++) {
        let idx = splat_count - 1u - j;
        let i = tile_splats[idx];
        let a = alphas[idx];
        if (a < 0.001) { continue; }
        let s = splats.splats[i];
        let color = s.color.rgb;
        let opacity = s.color.a;
        let T_prev = Ts[idx];
        let ps = eval_splat(s, p, aspect);

        let dColor = dC * (T_prev * a);
        let linear_depth_bwd = max(ps.w, DEPTH_NEAR);
        let depth = clamp(1.0 - DEPTH_NEAR / linear_depth_bwd, 0.0, 1.0);
        let da_C = dT_C * (-T_prev) + dot(dC, T_prev * color);
        let da_D = dT_D * (-T_prev) + dD * T_prev * depth;
        let da = da_C + da_D;
        
        dT_C = dT_C * (1.0 - a) + dot(dC, a * color);
        dT_D = dT_D * (1.0 - a) + dD * a * depth;
        
        let d_depth = dD * T_prev * a;
        let sx = max(s.pos_sx.w, 0.0001);
        let sy = max(s.sy_shape.x, 0.0001);
        let shape_a = s.sy_shape.y;
        let shape_b = s.sy_shape.z;
        let power = -shape_b * pow(ps.r, shape_a);

        var d_opacity = 0.0;
        var d_screen = vec2f(0.0);
        var d_sx = 0.0;
        var d_sy = 0.0;
        var d_shape_a = 0.0;
        var d_shape_b = 0.0;
        var d_ax_screen = vec2f(0.0);
        var d_ay_screen = vec2f(0.0);

        let above_floor = power > -15.0;
        let a_un = select(0.0, exp(power), above_floor);
        d_opacity = select(0.0, da * a_un, above_floor);
        let d_power = select(0.0, da * opacity * a_un, above_floor);
        let r_pow_a   = pow(ps.r, shape_a);
        let r_pow_a_m2 = pow(ps.r, shape_a - 2.0);
        d_shape_a = select(0.0, d_power * (-shape_b * r_pow_a * log(ps.r)), above_floor);
        d_shape_b = select(0.0, d_power * (-r_pow_a), above_floor);
        let d_sp_raw = d_power * (-shape_b * shape_a * r_pow_a_m2) * ps.sp;
        let d_sp = select(vec2f(0.0), d_sp_raw, above_floor);

        // d_sp -> d_d via M_inv^T
        let d_d = vec2f(
            d_sp.x * ps.m11 * ps.inv_det + d_sp.y * (-ps.m10) * ps.inv_det,
            d_sp.x * (-ps.m01) * ps.inv_det + d_sp.y * ps.m00 * ps.inv_det
        );
        d_screen = -d_d;

        // d_sp -> d_M_scaled columns via inverse derivative
        // sp = M_inv * d, so d_M_ij = -(M_inv^T * d_sp)_i * (M_inv * d)_... 
        // Actually: d(M_inv)/d(M_kl) = -M_inv[:,k] * M_inv[l,:] 
        // d_loss/d_M_kl = -sum_ij d_sp_i * M_inv_ik * sp_j  where j indexes d
        // Wait: sp = M_inv * d. d_loss/d_M_inv_ij = d_sp_i * d_j
        // d_loss/d_M_kl = -sum_ij (M_inv^T)_ki * d_sp_i * d_j * (M_inv^T)_lj
        // = -(M_inv^T * d_sp)_k * (M_inv^T * d)_l... no
        // sp = M_inv * d. Let me use: d_M = -M_inv^T * outer(d_sp, d) * M_inv^T... 
        // Actually simpler: sp_i = sum_j (M_inv)_ij * d_j
        // d_loss/d_M_kl = sum_i d_sp_i * d(M_inv_ij)/d(M_kl) * d_j
        // d(M_inv)/d(M_kl) => (M * M_inv = I) => dM * M_inv + M * dM_inv = 0
        //   dM_inv = -M_inv * dM * M_inv
        // d(M_inv)_ij / d(M_kl) = -sum_m (M_inv)_im * delta_mk * (M_inv)_lj
        //                        = -(M_inv)_ik * (M_inv)_lj
        // So: d_loss/d_M_kl = -sum_ij d_sp_i * (M_inv)_ik * (M_inv)_lj * d_j
        //                   = -(M_inv^T * d_sp)_k * (M_inv^T * d)_l... 
        // Wait: sum_i (M_inv)_ik * d_sp_i = (M_inv^T * d_sp)_k, and
        //        sum_j (M_inv)_lj * d_j = sp_l (since sp = M_inv * d)
        // Hmm no: sum_j (M_inv)_lj * d_j = (M_inv * d)_l = sp_l
        // So d_loss/d_M_kl = -(M_inv^T * d_sp)_k * sp_l
        let minvT_dsp = vec2f(
            d_sp.x * ps.m11 * ps.inv_det + d_sp.y * (-ps.m10) * ps.inv_det,
            d_sp.x * (-ps.m01) * ps.inv_det + d_sp.y * ps.m00 * ps.inv_det
        );
        // d_M = -outer(minvT_dsp, sp)
        // M = [[m00, m01], [m10, m11]] = [[ax_screen.x*sx, ay_screen.x*sy], [ax_screen.y*sx, ay_screen.y*sy]]
        let d_m00 = -minvT_dsp.x * ps.sp.x;
        let d_m01 = -minvT_dsp.x * ps.sp.y;
        let d_m10 = -minvT_dsp.y * ps.sp.x;
        let d_m11 = -minvT_dsp.y * ps.sp.y;
        // m00 = ax_screen.x * sx => d_ax_screen.x += d_m00 * sx, d_sx += d_m00 * ax_screen.x
        d_ax_screen = vec2f(d_m00 * sx, d_m10 * sx);
        d_ay_screen = vec2f(d_m01 * sy, d_m11 * sy);
        d_sx = d_m00 * ps.ax_screen.x + d_m10 * ps.ax_screen.y;
        d_sy = d_m01 * ps.ay_screen.x + d_m11 * ps.ay_screen.y;

        // Project d_screen back to d_pos (3D)
        // screen = (clip.x/w * aspect, clip.y/w) where clip = VP * (pos,1), w = clip.w
        // d_clip.x = d_screen.x / aspect * (1/w)... etc. This is complex.
        // Simplified: treat VP rows as constants, d_pos_j = sum over screen components
        // screen.x = clip.x/w * aspect, clip = VP * (pos,1)
        // d_screen.x/d_pos_j = aspect * (VP[0][j]*w - clip.x*VP[3][j]) / w^2
        // d_screen.y/d_pos_j = (VP[1][j]*w - clip.y*VP[3][j]) / w^2
        let w = ps.w;
        let w2 = w * w;
        var d_pos = vec3f(0.0);
        for (var ax = 0u; ax < 3u; ax++) {
            let vp_0j = splat_uniforms.vp[ax][0];
            let vp_1j = splat_uniforms.vp[ax][1];
            let vp_3j = splat_uniforms.vp[ax][3];
            let ds_dx = aspect * (vp_0j * w - ps.clip_xy.x * vp_3j) / w2;
            let ds_dy = (vp_1j * w - ps.clip_xy.y * vp_3j) / w2;
            d_pos[ax] = d_screen.x * ds_dx + d_screen.y * ds_dy;
            // d(depth)/d(w) for reciprocal encoding: depth = 1 - DEPTH_NEAR/w => d/dw = DEPTH_NEAR/w²
            // d(w)/d(pos[ax]) = VP[3][ax] (homogeneous row)
            let d_depth_d_w = DEPTH_NEAR / (w * w);
            d_pos[ax] += select(0.0, vp_3j * d_depth_d_w * d_depth, ps.w > DEPTH_NEAR);
        }

        // d_ax_screen -> d_quat (approximate: ignore Jacobian's dependence on quat)
        // ax_screen = project_axis(vp, quat_rotate(q, e_x), clip_xy, w, aspect)
        // For simplicity, use finite-rotation gradient approximation:
        // d_quat from ax_screen and ay_screen changes
        // ax_world = R(q) * e_x. d(ax_world)/d(q) is complex.
        // Use: d_ax_world from d_ax_screen, then d_quat from d_ax_world + d_ay_world
        // ax_screen = f(vp, ax_world, clip_xy, w, aspect)
        // d_ax_world_j = d_ax_screen.x * aspect*(VP[j][0]*w - clip.x*VP[j][3])/(w*w) + d_ax_screen.y * (VP[j][1]*w - clip.y*VP[j][3])/(w*w)
        var d_ax_world = vec3f(0.0);
        var d_ay_world = vec3f(0.0);
        for (var ax2 = 0u; ax2 < 3u; ax2++) {
            let vp_0j2 = splat_uniforms.vp[ax2][0];
            let vp_1j2 = splat_uniforms.vp[ax2][1];
            let vp_3j2 = splat_uniforms.vp[ax2][3];
            let j0 = aspect * (vp_0j2 * w - ps.clip_xy.x * vp_3j2) / w2;
            let j1 = (vp_1j2 * w - ps.clip_xy.y * vp_3j2) / w2;
            d_ax_world[ax2] = d_ax_screen.x * j0 + d_ax_screen.y * j1;
            d_ay_world[ax2] = d_ay_screen.x * j0 + d_ay_screen.y * j1;
        }

        // d_quat from d_ax_world and d_ay_world
        // quat_rotate(q, v) = v + q.x * t + cross(q.yzw, t), t = 2 * cross(q.yzw, v)
        // Gradient of quat_rotate w.r.t. q:
        let ex = vec3f(1.0, 0.0, 0.0);
        let ey = vec3f(0.0, 1.0, 0.0);
        let tx = 2.0 * cross(s.quat.yzw, ex);
        let ty = 2.0 * cross(s.quat.yzw, ey);
        // d/d(q.x): result += q.x * t => d_qx = dot(d_out, t)
        var d_qx = dot(d_ax_world, tx) + dot(d_ay_world, ty);
        // d/d(q.yzw) through t and cross(q.yzw, t):
        // t = 2 * cross(u, v) where u = q.yzw
        // d_result/d_u = q.x * d_t/d_u + d_cross(u,t)/d_u
        // d_cross(u,t)/d_u: cross(u,t) => skew(t)^T, plus u affects t
        // This is getting very involved. Use numeric-friendly formulation:
        // For quat_rotate: result = (1 - 2(qy²+qz²)) * vx + ... (rotation matrix form)
        // Let me just compute d_quat via rotation matrix form
        let qw = s.quat.x; let qx2 = s.quat.y; let qy = s.quat.z; let qz = s.quat.w;
        // R column 0 (ax_world for ex):
        // [1-2(qy²+qz²), 2(qx*qy+qw*qz), 2(qx*qz-qw*qy)]
        // R column 1 (ay_world for ey):
        // [2(qx*qy-qw*qz), 1-2(qx²+qz²), 2(qy*qz+qw*qx)]
        // d_R00/d_qw = 0, d_R00/d_qx = 0, d_R00/d_qy = -4qy, d_R00/d_qz = -4qz
        // etc.
        // d_qw from ax: d_ax_world.y * 2*qz + d_ax_world.z * (-2*qy)
        // d_qw from ay: d_ay_world.x * (-2*qz) + d_ay_world.z * 2*qx2
        let d_qw_ax = d_ax_world.y * 2.0*qz - d_ax_world.z * 2.0*qy;
        let d_qw_ay = -d_ay_world.x * 2.0*qz + d_ay_world.z * 2.0*qx2;
        let d_qx_ax = d_ax_world.y * 2.0*qy + d_ax_world.z * 2.0*qz;
        let d_qx_ay = d_ay_world.x * 2.0*qy - d_ay_world.y * 4.0*qx2 + d_ay_world.z * 2.0*qw;
        let d_qy_ax = -d_ax_world.x * 4.0*qy + d_ax_world.y * 2.0*qx2 + d_ax_world.z * (-2.0*qw);
        let d_qy_ay = d_ay_world.x * 2.0*qx2 + d_ay_world.z * 2.0*qz;
        let d_qz_ax = -d_ax_world.x * 4.0*qz + d_ax_world.y * 2.0*qw + d_ax_world.z * 2.0*qx2;
        let d_qz_ay = -d_ay_world.x * 2.0*qw - d_ay_world.y * 4.0*qz + d_ay_world.z * 2.0*qy;

        let d_qw_total = d_qw_ax + d_qw_ay;
        let d_qx_total = d_qx_ax + d_qx_ay;
        let d_qy_total = d_qy_ax + d_qy_ay;
        let d_qz_total = d_qz_ax + d_qz_ay;

        // Shape regularization: push toward flat-topped profile (gouache-like)
        // shape_a=6 → flat plateau, shape_b=0.3 → wide coverage before falloff
        let REG_SHAPE_STRENGTH = 0.005;
        let target_shape_a = 6.0;
        let target_shape_b = 0.3;
        d_shape_a += REG_SHAPE_STRENGTH * 2.0 * (shape_a - target_shape_a);
        d_shape_b += REG_SHAPE_STRENGTH * 2.0 * (shape_b - target_shape_b);

        let FP_SCALE_POS = 10000.0;
        let FP_SCALE_COL = 100000.0;

        let base_idx = i * 15u;
        atomicAdd(&grads.data[base_idx + 0u], i32(d_pos.x * FP_SCALE_POS));
        atomicAdd(&grads.data[base_idx + 1u], i32(d_pos.y * FP_SCALE_POS));
        atomicAdd(&grads.data[base_idx + 2u], i32(d_pos.z * FP_SCALE_POS));
        atomicAdd(&grads.data[base_idx + 3u], i32(d_sx * FP_SCALE_POS));
        atomicAdd(&grads.data[base_idx + 4u], i32(dColor.r * FP_SCALE_COL));
        atomicAdd(&grads.data[base_idx + 5u], i32(dColor.g * FP_SCALE_COL));
        atomicAdd(&grads.data[base_idx + 6u], i32(dColor.b * FP_SCALE_COL));
        atomicAdd(&grads.data[base_idx + 7u], i32(d_opacity * FP_SCALE_COL));
        atomicAdd(&grads.data[base_idx + 8u], i32(d_qw_total * FP_SCALE_POS));
        atomicAdd(&grads.data[base_idx + 9u], i32(d_qx_total * FP_SCALE_POS));
        atomicAdd(&grads.data[base_idx + 10u], i32(d_qy_total * FP_SCALE_POS));
        atomicAdd(&grads.data[base_idx + 11u], i32(d_qz_total * FP_SCALE_POS));
        atomicAdd(&grads.data[base_idx + 12u], i32(d_sy * FP_SCALE_POS));
        atomicAdd(&grads.data[base_idx + 13u], i32(d_shape_a * FP_SCALE_POS));
        atomicAdd(&grads.data[base_idx + 14u], i32(d_shape_b * FP_SCALE_POS));
    }
}
