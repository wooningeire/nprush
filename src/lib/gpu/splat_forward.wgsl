struct Splat {
    pos_sx: vec4f,    // x, y, z, sx
    color: vec4f,     // r, g, b, opacity
    quat: vec4f,      // qw, qx, qy, qz
    sy_shape: vec4f,  // sy, shape_a, shape_b, _pad
}

struct SplatArray {
    splats: array<Splat, NUM_SPLATS>,
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
    @location(0) local_p: vec2f,
    @location(1) @interpolate(flat) instance_idx: u32,
    @location(2) depth: f32,
}

fn quat_rotate(q: vec4f, v: vec3f) -> vec3f {
    let t = 2.0 * cross(q.yzw, v);
    return v + q.x * t + cross(q.yzw, t);
}

@vertex
fn vert(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    let splat_idx = NUM_SPLATS - 1u - ii;
    let s = splats.splats[splat_idx];
    
    let pos3 = s.pos_sx.xyz;
    let sx = max(s.pos_sx.w, 0.0001);
    let sy = max(s.sy_shape.x, 0.0001);
    let shape_a = s.sy_shape.y;
    let shape_b = s.sy_shape.z;
    let q = s.quat;
    
    let safe_shape_b = max(shape_b, 0.0001);
    let R = pow(15.0 / safe_shape_b, 1.0 / shape_a);
    
    // Local quad vertices — these ARE the scaled_p coordinates
    let quad_x = array<f32, 6>(-R,  R, -R, -R,  R,  R);
    let quad_y = array<f32, 6>(-R, -R,  R,  R, -R,  R);
    let lx = quad_x[vi];
    let ly = quad_y[vi];
    
    // Scale then rotate into world space
    let local_offset = vec3f(lx * sx, ly * sy, 0.0);
    let world_offset = quat_rotate(q, local_offset);
    let world_pos = pos3 + world_offset;
    
    let clip = uniforms.vp * vec4f(world_pos, 1.0);
    
    var o: VsOut;
    o.pos = clip;
    // Pass the unscaled local coords — they interpolate correctly
    // because the quad is planar in splat-local space
    o.local_p = vec2f(lx, ly);
    o.instance_idx = splat_idx;
    o.depth = clip.w;
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
    
    // v.local_p is already scaled_p (local coordinates / scale)
    let r = max(length(v.local_p), 0.0001);
    let pw = -shape_b * pow(r, shape_a);

    var a = select(0.0, exp(pw) * s.color.a, pw > -15.0);
    a = clamp(a, 0.0, 0.999);
    
    if (a < 0.001) {
        discard;
    }
    
    var out: FragOut;
    out.color = vec4f(s.color.rgb, a);
    out.depth = vec4f(v.depth, v.depth, v.depth, a);
    return out;
}
