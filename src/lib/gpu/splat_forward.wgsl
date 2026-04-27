struct Splat {
    transform: vec4f,
    color: vec4f,
    rot_pad: vec4f,
}

struct SplatArray {
    splats: array<Splat, NUM_SPLATS>,
}

@group(0) @binding(0) var<storage, read> splats: SplatArray;

struct ForwardUniforms {
    dims: vec2f,
}
@group(0) @binding(1) var<uniform> uniforms: ForwardUniforms;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) splat_p: vec2f,
    @location(1) @interpolate(flat) instance_idx: u32,
}

@vertex
fn vert(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    // Render back-to-front by reversing the instance index
    let splat_idx = NUM_SPLATS - 1u - ii;
    let s = splats.splats[splat_idx];
    
    // Opacity drops to 0 when power = -shape_b * pow(R, shape_a) <= -15.0.
    // -15.0 is the cutoff used in the backward pass.
    let shape_a = s.rot_pad.y;
    let shape_b = s.rot_pad.z;
    // Solve: shape_b * pow(R, shape_a) = 15.0
    let safe_shape_b = max(shape_b, 0.0001);
    let R = pow(15.0 / safe_shape_b, 1.0 / shape_a);
    
    let scale = max(vec2f(0.0001), s.transform.zw);
    let rot = s.rot_pad.x;
    let pos = s.transform.xy;
    
    // Local quad vertices (unscaled, unrotated)
    var lx: f32; var ly: f32;
    switch vi {
        case 0u: { lx = -R; ly = -R; }
        case 1u: { lx =  R; ly = -R; }
        case 2u: { lx = -R; ly =  R; }
        case 3u: { lx = -R; ly =  R; }
        case 4u: { lx =  R; ly = -R; }
        default: { lx =  R; ly =  R; }
    }
    
    let local_p = vec2f(lx, ly) * scale;
    
    // Rotate
    let co = cos(rot);
    let si = sin(rot);
    let world_offset = vec2f(
        co * local_p.x - si * local_p.y,
        si * local_p.x + co * local_p.y
    );
    
    let p = pos + world_offset;
    
    let aspect = uniforms.dims.x / uniforms.dims.y;
    let ndc_x = p.x / aspect;
    let ndc_y = p.y;
    
    var o: VsOut;
    o.pos = vec4f(ndc_x, ndc_y, 0.0, 1.0);
    o.splat_p = p;
    o.instance_idx = splat_idx;
    return o;
}

@fragment
fn frag(v: VsOut) -> @location(0) vec4f {
    let s = splats.splats[v.instance_idx];
    let d = v.splat_p - s.transform.xy;
    let rot = s.rot_pad.x;
    let co = cos(rot);
    let si = sin(rot);
    let lp = vec2f(co * d.x + si * d.y, -si * d.x + co * d.y);
    let ss = max(vec2f(0.0001), s.transform.zw);
    let sp = lp / ss;
    let r = max(length(sp), 0.0001);
    let pw = -s.rot_pad.z * pow(r, s.rot_pad.y);

    var a = 0.0;
    if (pw > -15.0) {
        a = exp(pw) * s.color.a;
    }
    a = clamp(a, 0.0, 0.999);
    
    if (a < 0.001) {
        discard;
    }
    
    return vec4f(s.color.rgb, a);
}
