struct Bezier {
    p0: vec4f, // x, y, z, width
    p1: vec4f, // x, y, z, softness
    p2: vec4f, // x, y, z, _pad
    p3: vec4f, // x, y, z, _pad
    color: vec4f,
    sh1_r: vec4f,
    sh1_g: vec4f,
    sh1_b: vec4f,
    sh1_a: vec4f,
}

struct BezierArray {
    items: array<Bezier, {@NUM_BEZIERS}u>,
}

struct ForwardUniforms {
    vp: mat4x4f,
    dims: vec2f,
    _pad: vec2f,
    cam_world: vec4f,
}

@group(0) @binding(0) var<storage, read> beziers: BezierArray;
@group(0) @binding(1) var<uniform> uniforms: ForwardUniforms;
@group(0) @binding(2) var brush_sampler: sampler;
@group(0) @binding(3) var brush_texture: texture_2d<f32>;

@group(0) @binding(4) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var<storage, read> instance_vals: array<u32>;
@group(0) @binding(6) var<storage, read> tile_starts: array<u32>;
@group(0) @binding(7) var<storage, read> tile_ends: array<u32>;

const N_SEG: u32 = {@BEZIER_POLY_SEG}u;
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

fn bezier_at(p0: vec2f, p1: vec2f, p2: vec2f, p3: vec2f, t: f32) -> vec2f {
    let omt = 1.0 - t;
    return omt*omt*omt * p0
         + 3.0 * omt*omt * t * p1
         + 3.0 * omt * t*t * p2
         + t*t*t * p3;
}

fn project_center(vp: mat4x4f, pos3: vec3f, aspect: f32) -> vec3f {
    let clip = vp * vec4f(pos3, 1.0);
    return vec3f(clip.x / clip.w * aspect, -clip.y / clip.w, clip.w);
}

fn bernstein(t: f32) -> vec4f {
    let omt = 1.0 - t;
    return vec4f(omt*omt*omt, 3.0*omt*omt*t, 3.0*omt*t*t, t*t*t);
}

var<workgroup> shared_beziers: array<Bezier, 128>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3u, @builtin(workgroup_id) workgroup_id: vec3u, @builtin(local_invocation_id) local_id: vec3u) {
    let dims = vec2u(uniforms.dims);
    let aspect = uniforms.dims.x / uniforms.dims.y;
    
    let tile_id = workgroup_id.y * u32(ceil(uniforms.dims.x / 16.0)) + workgroup_id.x;
    let start_idx = tile_starts[tile_id];
    let end_idx = tile_ends[tile_id];
    let bezier_count = end_idx - start_idx;
    
    if (global_id.x >= dims.x || global_id.y >= dims.y) { return; }

    let p_ndc = (vec2f(global_id.xy) + 0.5) / vec2f(dims) * 2.0 - 1.0;
    let p = vec2f(p_ndc.x * aspect, -p_ndc.y);

    var C_pred = vec3f(0.0);
    var T_final = 1.0;

    let local_idx = local_id.y * 16u + local_id.x;

    for (var chunk = 0u; chunk < bezier_count; chunk += 128u) {
        let load_idx = chunk + local_idx;
        if (local_idx < 128u && load_idx < bezier_count) {
            let bezier_id = instance_vals[start_idx + load_idx];
            shared_beziers[local_idx] = beziers.items[bezier_id];
        }
        workgroupBarrier();

        let valid_count = min(128u, bezier_count - chunk);
        for (var i = 0u; i < valid_count; i++) {
            let b = shared_beziers[i];
            
            let proj0 = project_center(uniforms.vp, b.p0.xyz, aspect);
            let proj1 = project_center(uniforms.vp, b.p1.xyz, aspect);
            let proj2 = project_center(uniforms.vp, b.p2.xyz, aspect);
            let proj3 = project_center(uniforms.vp, b.p3.xyz, aspect);
            
            const DEPTH_NEAR_CULL = 0.1;
            if (proj0.z < DEPTH_NEAR_CULL || proj1.z < DEPTH_NEAR_CULL || proj2.z < DEPTH_NEAR_CULL || proj3.z < DEPTH_NEAR_CULL) { continue; }

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
                let proj_pt = prev + u * seg;
                let diff = p - proj_pt;
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
            let w = dot(B, vec4f(proj0.z, proj1.z, proj2.z, proj3.z));
            let inv_w = 1.0 / max(w, 0.001);
            
            let width = max(b.p0.w, 0.0001) * inv_w;
            let softness = max(b.p1.w, 0.0001) * inv_w;
            let local_width = width * pressure;
            let local_softness = softness * pressure;
            
            let inner = local_width - local_softness;
            let outer = local_width + local_softness;
            let a_geom = 1.0 - smoothstep(inner, outer, min_d);
            
            if (a_geom < 0.001) { continue; }

            let pos_w = bezier_pos_world(b, t);
            let tang = bezier_deriv_world(b, t);
            let dl_b = bezier_dirs_sh(uniforms.cam_world.xyz, pos_w, tang);
            let lx_b = dl_b.x;
            let ly_b = dl_b.y;
            let lz_b = dl_b.z;
            
            let o_lin = b.color.a + SH_C1_B * (ly_b * b.sh1_a.x + lz_b * b.sh1_a.y + lx_b * b.sh1_a.z);
            let opacity = clamp(o_lin, 0.0, 1.0);
            
            // Brush lookup
            let t_prev = f32(min_k - 1u) / f32(N_SEG);
            let t_curr = f32(min_k) / f32(N_SEG);
            let best_prev = bezier_at(p0, p1, p2, p3, t_prev);
            let best_curr = bezier_at(p0, p1, p2, p3, t_curr);
            let best_seg = best_curr - best_prev;
            let best_len = max(length(best_seg), 1e-4);
            let best_dir = best_seg / best_len;
            let best_proj = best_prev + min_u * best_seg;
            let best_diff = p - best_proj;
            let signed_cross = best_diff.x * (-best_dir.y) + best_diff.y * best_dir.x;
            
            let brush_u = t;
            let brush_v = clamp(signed_cross / max(local_width + local_softness, 1e-6) * 0.5 + 0.5, 0.0, 1.0);
            let brush_alpha = textureSampleLevel(brush_texture, brush_sampler, vec2f(brush_u, brush_v), 0.0).r;
            
            let a = clamp(a_geom * brush_alpha * opacity * pressure, 0.0, 0.999);
            if (a < 0.001) { continue; }

            let rr_lin = b.color.r + SH_C1_B * (ly_b * b.sh1_r.x + lz_b * b.sh1_r.y + lx_b * b.sh1_r.z);
            let gg_lin = b.color.g + SH_C1_B * (ly_b * b.sh1_g.x + lz_b * b.sh1_g.y + lx_b * b.sh1_g.z);
            let bb_lin = b.color.b + SH_C1_B * (ly_b * b.sh1_b.x + lz_b * b.sh1_b.y + lx_b * b.sh1_b.z);
            let rgb = max(vec3f(rr_lin, gg_lin, bb_lin), vec3f(0.0));

            C_pred = C_pred * (1.0 - a) + rgb * a;
            T_final *= (1.0 - a);
        }
        workgroupBarrier();
    }

    textureStore(outTex, global_id.xy, vec4f(C_pred, 1.0));
}
