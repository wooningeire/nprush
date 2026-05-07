struct Splat {
    pos_sx: vec4f,    // x, y, z, sx
    color: vec4f,     // r, g, b, opacity
    quat: vec4f,      // qw, qx, qy, qz
    sy_shape: vec4f,  // sy, shape_a, shape_b, _pad
}

struct SplatArray {
    splats: array<Splat, {@NUM_SPLATS}u>,
}

@group(0) @binding(0) var<storage, read> splats: SplatArray;

struct ForwardUniforms {
    vp: mat4x4f,
    dims: vec2f,
    _pad: vec2f,
}
@group(0) @binding(1) var<uniform> uniforms: ForwardUniforms;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) d: vec2f,
    @location(1) @interpolate(flat) instance_idx: u32,
    @location(2) depth: f32,
    @location(3) @interpolate(flat) conic: vec3f,
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
    let splat_idx = {@NUM_SPLATS}u - 1u - ii;
    let s = splats.splats[splat_idx];
    
    let pos3 = s.pos_sx.xyz;
    let sx = max(s.pos_sx.w, 0.0001);
    let sy = max(s.sy_shape.x, 0.0001);
    let sz = max(s.sy_shape.w, 0.0001);
    let shape_a = s.sy_shape.y;
    let shape_b = s.sy_shape.z;
    let q = s.quat;
    
    let safe_shape_b = max(shape_b, 0.0001);
    let R = pow(15.0 / safe_shape_b, 1.0 / shape_a);
    
    let clip_center = uniforms.vp * vec4f(pos3, 1.0);
    let w = clip_center.w;
    
    // Collapse quad for splats behind the camera or too close to near plane
    if (w < 0.1) {
        var o_clip: VsOut;
        o_clip.pos = vec4f(0.0, 0.0, 2.0, 1.0); // Clip it
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
    
    var o: VsOut;
    o.pos = clip;
    o.d = vec2f(lx, ly);
    o.instance_idx = splat_idx;
    o.depth = w;
    o.conic = vec3f(A, B, C);
    return o;
}

struct FragOut {
    @location(0) color: vec4f,
    @location(1) depth: vec4f,
}

@fragment
fn frag(v: VsOut) -> FragOut {
    let s = splats.splats[v.instance_idx];
    let shape_a = s.sy_shape.y;
    let shape_b = s.sy_shape.z;
    
    let A = v.conic.x;
    let B = v.conic.y;
    let C = v.conic.z;
    let dx = v.d.x;
    let dy = v.d.y;
    let r2 = A * dx * dx + 2.0 * B * dx * dy + C * dy * dy;
    let r = sqrt(max(r2, 0.0001));
    let pw = -shape_b * pow(r, shape_a);

    var a = select(0.0, exp(pw) * s.color.a, pw > -15.0);
    a = clamp(a, 0.0, 0.999);
    
    if (a < 0.001) {
        discard;
    }
    
    // Reciprocal depth encoding matching mesh.wgsl: 1 - DEPTH_NEAR / w
    const DEPTH_NEAR = 0.1;
    let linear_depth = max(v.depth, DEPTH_NEAR);
    let enc_depth = clamp(1.0 - DEPTH_NEAR / linear_depth, 0.0, 1.0);

    var out: FragOut;
    out.color = vec4f(s.color.rgb, a);
    out.depth = vec4f(enc_depth, enc_depth, enc_depth, a);
    return out;
}
