struct Bezier {
    p0: vec4f, // x, y, z, width
    p1: vec4f, // x, y, z, softness
    p2: vec4f, // x, y, z, _pad
    p3: vec4f, // x, y, z, _pad
    color: vec4f,
}

struct BezierArray {
    items: array<Bezier, NUM_BEZIERS>,
}

struct ForwardUniforms {
    vp: mat4x4f,
    dims: vec2f,
    _pad: vec2f,
}

@group(0) @binding(0) var<storage, read> beziers: BezierArray;
@group(0) @binding(1) var<uniform> uniforms: ForwardUniforms;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec4f,
    @location(1) dist: f32,
    @location(2) width: f32,
    @location(3) softness: f32,
}

fn bezier_at(p0: vec3f, p1: vec3f, p2: vec3f, p3: vec3f, t: f32) -> vec3f {
    let omt = 1.0 - t;
    return omt*omt*omt * p0
         + 3.0 * omt*omt * t * p1
         + 3.0 * omt * t*t * p2
         + t*t*t * p3;
}

fn bezier_derivative_at(p0: vec3f, p1: vec3f, p2: vec3f, p3: vec3f, t: f32) -> vec3f {
    let omt = 1.0 - t;
    return 3.0 * omt*omt * (p1 - p0)
         + 6.0 * omt * t * (p2 - p1)
         + 3.0 * t*t * (p3 - p2);
}

@vertex
fn vs_main(
    @builtin(instance_index) ii: u32,
    @builtin(vertex_index) vi: u32
) -> VsOut {
    let b = beziers.items[ii];
    let num_segments = 16u;
    let seg_idx = vi / 2u;
    let side = f32(vi % 2u) * 2.0 - 1.0; // -1 or 1
    
    let t = f32(seg_idx) / f32(num_segments);
    let pressure = smoothstep(0.0, 0.5, t) * smoothstep(1.0, 0.5, t);
    
    let p3 = bezier_at(b.p0.xyz, b.p1.xyz, b.p2.xyz, b.p3.xyz, t);
    let dp3 = bezier_derivative_at(b.p0.xyz, b.p1.xyz, b.p2.xyz, b.p3.xyz, t);
    
    let proj = uniforms.vp * vec4f(p3, 1.0);
    
    // Simple culling
    if (proj.w <= 0.0) {
        return VsOut(vec4f(0.0), vec4f(0.0), 0.0, 0.0, 0.0);
    }
    
    let proj_next = uniforms.vp * vec4f(p3 + dp3 * 0.001, 1.0);
    
    // NDC positions
    let screen_p = proj.xy / proj.w;
    let screen_p_next = proj_next.xy / proj_next.w;
    
    let aspect = uniforms.dims.x / uniforms.dims.y;
    
    // Calculate tangent and normal in aspect-corrected space
    var tangent = vec2f(screen_p_next.x - screen_p.x, (screen_p_next.y - screen_p.y) / aspect);
    if (length(tangent) < 1e-6) {
        tangent = vec2f(1.0, 0.0);
    } else {
        tangent = normalize(tangent);
    }
    
    // Normal in NDC space (undo aspect correction)
    let normal_ndc = vec2f(-tangent.y * aspect, tangent.x);
    
    let width = max(b.p0.w, 0.0001) * pressure;
    let softness = max(b.p1.w, 0.0001) * pressure;
    let total_radius = width + softness;
    
    let offset_pos = screen_p + normal_ndc * side * total_radius;
    
    var out: VsOut;
    out.pos = vec4f(offset_pos * proj.w, proj.z, proj.w);
    out.color = vec4f(b.color.rgb, b.color.a * pressure);
    out.dist = side * total_radius;
    out.width = width;
    out.softness = softness;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    let dist = abs(in.dist);
    let inner = in.width - in.softness;
    let outer = in.width + in.softness;
    let a_geom = 1.0 - smoothstep(inner, outer, dist);
    let a = clamp(a_geom * in.color.a, 0.0, 0.999);
    
    if (a < 0.001) { discard; }
    
    return vec4f(in.color.rgb * a, a);
}
