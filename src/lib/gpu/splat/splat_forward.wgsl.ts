import { interpolateWgslTemplate } from "../wgsl-templates/interpolateWgslTemplate.ts";
import { Splat } from "./Splat.wgsl.ts";

export default interpolateWgslTemplate`
const SH_C1: f32 = 0.4886025119029199;

fn quat_conj(q: vec4f) -> vec4f {
    return vec4f(q.x, -q.y, -q.z, -q.w);
}

fn splat_view_dir_local(cam_world: vec3f, pos: vec3f, q: vec4f) -> vec3f {
    let v_w = cam_world - pos;
    let inv_len = inverseSqrt(max(dot(v_w, v_w), 1e-18));
    let dir_w = v_w * inv_len;
    return quat_rotate(quat_conj(q), dir_w);
}

fn splat_rgb_sh1(s: ${Splat}, dir_l: vec3f) -> vec3f {
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

@group(0) @binding(2) var<storage, read> sort_order: array<u32, {@NUM_SPLATS}u>;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) d: vec2f,
    @location(1) @interpolate(flat) instance_idx: u32,
    @location(2) depth: f32,
    @location(3) @interpolate(flat) conic: vec3f,
    @location(4) @interpolate(flat) rgb: vec3f,
    @location(5) @interpolate(flat) opacity_view: f32,
}

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

@vertex
fn vert(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    // Use sort order: instance 0 = farthest splat (drawn first, behind everything)
    let splat_idx = sort_order[ii];
    let s = splats.splats[splat_idx];
    
    let pos3 = s.pos_sx.xyz;
    let sx = max(s.pos_sx.w, 0.0001);
    let sy = max(s.sy_shape.x, 0.0001);
    let sz = max(s.sy_shape.w, 0.0001);
    let q = s.quat;
    
    // 3DGS cutoff radius. Increased to 4.5 sigma so exp(-0.5 * 4.5^2) < 0.001
    // preventing hard quad edges from being visible before discard.
    let R = 4.5;
    
    let clip_center = uniforms.vp * vec4f(pos3, 1.0);
    let w = clip_center.w;
    
    // Collapse quad for splats behind the camera or too close to near plane
    const DEPTH_NEAR_CULL = 0.1;
    if (w < DEPTH_NEAR_CULL) {
        var o_clip: VsOut;
        o_clip.pos = vec4f(0.0, 0.0, 2.0, 1.0);
        o_clip.d = vec2f(0.0);
        o_clip.instance_idx = splat_idx;
        o_clip.depth = 0.0;
        o_clip.conic = vec3f(0.0);
        o_clip.rgb = vec3f(0.0);
        o_clip.opacity_view = 0.0;
        return o_clip;
    }
    
    let clip_xy = vec2f(clip_center.x, clip_center.y);
    let aspect = uniforms.dims.x / uniforms.dims.y;
    
    let ax_w = quat_rotate(q, vec3f(1.0, 0.0, 0.0));
    let ay_w = quat_rotate(q, vec3f(0.0, 1.0, 0.0));
    let az_w = quat_rotate(q, vec3f(0.0, 0.0, 1.0));
    
    let ax_s = project_axis(uniforms.vp, ax_w, clip_xy, w, aspect);
    let ay_s = project_axis(uniforms.vp, ay_w, clip_xy, w, aspect);
    let az_s = project_axis(uniforms.vp, az_w, clip_xy, w, aspect);
    
    let m0x = ax_s.x * sx; let m0y = ax_s.y * sx;
    let m1x = ay_s.x * sy; let m1y = ay_s.y * sy;
    let m2x = az_s.x * sz; let m2y = az_s.y * sz;
    
    var cov00 = m0x*m0x + m1x*m1x + m2x*m2x;
    var cov01 = m0x*m0y + m1x*m1y + m2x*m2y;
    var cov11 = m0y*m0y + m1y*m1y + m2y*m2y;
    
    // Low-pass filter (0.3px) to prevent aliasing for distant splats
    let filter_std = 0.3 * (2.0 / uniforms.dims.y);
    let filter2 = filter_std * filter_std;
    cov00 += filter2;
    cov11 += filter2;
    
    let det = cov00 * cov11 - cov01 * cov01;
    let inv_det = select(1.0 / det, 0.0, abs(det) < 1e-10);
    let A = cov11 * inv_det;
    let B = -cov01 * inv_det;
    let C = cov00 * inv_det;
    
    let extent_x = R * sqrt(max(cov00, 1e-9));
    let extent_y = R * sqrt(max(cov11, 1e-9));
    
    let quad_x = array<f32, 6>(-1.0,  1.0, -1.0, -1.0,  1.0,  1.0);
    let quad_y = array<f32, 6>(-1.0, -1.0,  1.0,  1.0, -1.0,  1.0);
    let lx = quad_x[vi] * extent_x;
    let ly = quad_y[vi] * extent_y;
    
    var clip = clip_center;
    clip.x += lx * w / aspect;
    clip.y += ly * w;
    
    let dir_l = splat_view_dir_local(uniforms.cam_world.xyz, pos3, q);
    let rgb = splat_rgb_sh1(s, dir_l);
    let opacity_view = splat_opacity_sh1(s, dir_l);

    var o: VsOut;
    o.pos = clip;
    o.d = vec2f(lx, ly);
    o.instance_idx = splat_idx;
    o.depth = w;
    o.conic = vec3f(A, B, C);
    o.rgb = rgb;
    o.opacity_view = opacity_view;
    return o;
}

struct FragOut {
    @location(0) color: vec4f,
}

@fragment
fn frag(v: VsOut) -> FragOut {
    let A = v.conic.x;
    let B = v.conic.y;
    let C = v.conic.z;
    let dx = v.d.x;
    let dy = v.d.y;
    let r2 = A * dx * dx + 2.0 * B * dx * dy + C * dy * dy;
    
    // Standard 3DGS Gaussian falloff
    let pw = -0.5 * r2;

    var a = select(0.0, exp(pw) * v.opacity_view, pw > -15.0);
    a = clamp(a, 0.0, 0.999);
    
    if (a < 0.001) {
        discard;
    }
    
    var out: FragOut;
    out.color = vec4f(v.rgb, a);
    return out;
}
`;