struct Splat {
    pos_sx: vec4f,    // x, y, z, sx
    color: vec4f,     // linear RGB base + opacity DC; SH1 adds directional residual on RGB and opacity
    quat: vec4f,
    sy_shape: vec4f,
    sh1_r: vec4f,     // xyz: red directional SH coefficients (scaled by SH_C1 in shading)
    sh1_g: vec4f,
    sh1_b: vec4f,
    sh1_a: vec4f,     // xyz: degree-1 opacity SH in local frame (same layout as RGB SH)
}

const SH_C1: f32 = 0.4886025119029199;

fn quat_rotate(q: vec4f, v: vec3f) -> vec3f {
    let t = 2.0 * cross(q.yzw, v);
    return v + q.x * t + cross(q.yzw, t);
}

fn quat_conj(q: vec4f) -> vec4f {
    return vec4f(q.x, -q.y, -q.z, -q.w);
}

fn splat_view_dir_local(cam_world: vec3f, pos: vec3f, q: vec4f) -> vec3f {
    let v_w = cam_world - pos;
    let inv_len = inverseSqrt(max(dot(v_w, v_w), 1e-18));
    let dir_w = v_w * inv_len;
    return quat_rotate(quat_conj(q), dir_w);
}

fn splat_rgb_sh1(s: Splat, dir_l: vec3f) -> vec3f {
    let x = dir_l.x;
    let y = dir_l.y;
    let z = dir_l.z;
    let rr = s.color.r + SH_C1 * (y * s.sh1_r.x + z * s.sh1_r.y + x * s.sh1_r.z);
    let gg = s.color.g + SH_C1 * (y * s.sh1_g.x + z * s.sh1_g.y + x * s.sh1_g.z);
    let bb = s.color.b + SH_C1 * (y * s.sh1_b.x + z * s.sh1_b.y + x * s.sh1_b.z);
    return max(vec3f(rr, gg, bb), vec3f(0.0));
}

fn splat_opacity_sh1(s: Splat, dir_l: vec3f) -> f32 {
    let x = dir_l.x;
    let y = dir_l.y;
    let z = dir_l.z;
    let o_lin = s.color.a + SH_C1 * (y * s.sh1_a.x + z * s.sh1_a.y + x * s.sh1_a.z);
    return clamp(o_lin, 0.0, 1.0);
}

struct SplatArray {
    splats: array<Splat, {@NUM_SPLATS}u>,
}

@group(0) @binding(0) var<storage, read> splats: SplatArray;

struct ForwardUniforms {
    vp: mat4x4f,
    dims: vec2f,
    _pad: vec2f,
    cam_world: vec4f,
}
@group(0) @binding(1) var<uniform> uniforms: ForwardUniforms;

@group(0) @binding(2) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var outDepthTex: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(4) var<storage, read> instance_vals: array<u32>;
@group(0) @binding(5) var<storage, read> tile_starts: array<u32>;
@group(0) @binding(6) var<storage, read> tile_ends: array<u32>;

var<workgroup> shared_splats: array<Splat, 128>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3u, @builtin(workgroup_id) workgroup_id: vec3u, @builtin(local_invocation_id) local_id: vec3u) {
    let dims = vec2u(uniforms.dims);
    let aspect = uniforms.dims.x / uniforms.dims.y;
    
    let tile_id = workgroup_id.y * u32(ceil(uniforms.dims.x / 16.0)) + workgroup_id.x;
    let start_idx = tile_starts[tile_id];
    let end_idx = tile_ends[tile_id];
    let splat_count = end_idx - start_idx;
    
    if (global_id.x >= dims.x || global_id.y >= dims.y) { return; }

    let p_ndc = (vec2f(global_id.xy) + 0.5) / vec2f(dims) * 2.0 - 1.0;
    let p = vec2f(p_ndc.x * aspect, -p_ndc.y);

    var C_pred = vec3f(0.05, 0.05, 0.05); // Background
    var D_pred = 1.0;

    let local_idx = local_id.y * 16u + local_id.x;

    for (var chunk = 0u; chunk < splat_count; chunk += 128u) {
        let load_idx = chunk + local_idx;
        if (local_idx < 128u && load_idx < splat_count) {
            let splat_idx = instance_vals[start_idx + load_idx];
            shared_splats[local_idx] = splats.splats[splat_idx];
        }
        workgroupBarrier();

        let valid_count = min(128u, splat_count - chunk);
        for (var i = 0u; i < valid_count; i++) {
            let s = shared_splats[i];
            
            let pos3 = s.pos_sx.xyz;
            let sx = max(s.pos_sx.w, 0.0001);
            let sy = max(s.sy_shape.x, 0.0001);
            let sz = max(s.sy_shape.w, 0.0001);
            let q = s.quat;
            
            let clip_center = uniforms.vp * vec4f(pos3, 1.0);
            let w = clip_center.w;
            
            // Standard near-plane culling
            if (w < 0.1) { continue; }
            
            let proj_center = vec2f(clip_center.x / w * aspect, -clip_center.y / w);
            
            // 3D Jacobian approach (simplified for compute)
            let q_mat = mat3x3f(
                quat_rotate(q, vec3f(1.0, 0.0, 0.0)),
                quat_rotate(q, vec3f(0.0, 1.0, 0.0)),
                quat_rotate(q, vec3f(0.0, 0.0, 1.0))
            );
            let scale_mat = mat3x3f(
                vec3f(sx, 0.0, 0.0),
                vec3f(0.0, sy, 0.0),
                vec3f(0.0, 0.0, sz)
            );
            let M = q_mat * scale_mat;
            let Sigma = M * transpose(M);
            
            let J = mat3x3f(
                vec3f(aspect / w, 0.0, -clip_center.x / (w * w) * aspect),
                vec3f(0.0, -1.0 / w, clip_center.y / (w * w)),
                vec3f(0.0, 0.0, 0.0) // We only care about 2D projection
            );
            
            let V2D = J * Sigma * transpose(J);
            // Add low-pass filter (0.3 pixel radius in NDC-ish space)
            let filter = 0.3 * (2.0 / uniforms.dims.y);
            let cov = vec3f(V2D[0][0] + filter*filter, V2D[0][1], V2D[1][1] + filter*filter);
            
            let det = cov.x * cov.z - cov.y * cov.y;
            if (det < 1e-10) { continue; }
            let conic = vec3f(cov.z / det, -cov.y / det, cov.x / det);
            
            let dx = p - proj_center;
            let power = -0.5 * (conic.x * dx.x * dx.x + 2.0 * conic.y * dx.x * dx.y + conic.z * dx.y * dx.y);
            if (power > 0.0) { continue; }
            
            let alpha = exp(power);
            if (alpha < 1.0/255.0) { continue; }
            
            let dir_l = splat_view_dir_local(uniforms.cam_world.xyz, pos3, q);
            let rgb = splat_rgb_sh1(s, dir_l);
            let opacity = splat_opacity_sh1(s, dir_l);
            let a = alpha * opacity;
            
            if (a < 1.0/255.0) { continue; }

            C_pred = C_pred * (1.0 - a) + rgb * a;
            D_pred = D_pred * (1.0 - a) + (clip_center.z / w) * a;
        }
        workgroupBarrier();
    }

    textureStore(outTex, global_id.xy, vec4f(C_pred, 1.0));
    textureStore(outDepthTex, global_id.xy, vec4f(vec3f(D_pred), 1.0));
}
