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
@group(0) @binding(1) var viewTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<uniform> uniforms: ForwardUniforms;

const N_BEZIER_SEG: u32 = 16u;

fn bezier_at(p0: vec2f, p1: vec2f, p2: vec2f, p3: vec2f, t: f32) -> vec2f {
    let omt = 1.0 - t;
    return omt*omt*omt * p0
         + 3.0 * omt*omt * t * p1
         + 3.0 * omt * t*t * p2
         + t*t*t * p3;
}

fn project_center(vp: mat4x4f, pos3: vec3f, aspect: f32) -> vec3f {
    let clip = vp * vec4f(pos3, 1.0);
    return vec3f(clip.x / clip.w * aspect, clip.y / clip.w, clip.w);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let dims = textureDimensions(viewTex);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }
    
    let aspect = f32(dims.x) / f32(dims.y);
    let uv = (vec2f(global_id.xy) + vec2f(0.5)) / vec2f(dims.xy);
    var p = uv * 2.0 - 1.0;
    p.y = -p.y;
    p.x = p.x * aspect;
    
    var Ts = 1.0;
    var C_accum = vec3f(0.0);
    for (var i = 0u; i < NUM_BEZIERS; i = i + 1u) {
        let b = beziers.items[i];
        let width = max(b.p0.w, 0.001);
        let softness = max(b.p1.w, 0.001);
        let opacity = b.color.a;
        if (opacity < 0.005) { continue; }

        let proj0 = project_center(uniforms.vp, b.p0.xyz, aspect);
        let proj1 = project_center(uniforms.vp, b.p1.xyz, aspect);
        let proj2 = project_center(uniforms.vp, b.p2.xyz, aspect);
        let proj3 = project_center(uniforms.vp, b.p3.xyz, aspect);

        // Cull if any point is behind the camera (simplification for now)
        if (proj0.z < 0.0 || proj1.z < 0.0 || proj2.z < 0.0 || proj3.z < 0.0) { continue; }

        let p0 = proj0.xy;
        let p1 = proj1.xy;
        let p2 = proj2.xy;
        let p3 = proj3.xy;

        var min_d = 1e9;
        var prev = p0;
        for (var k = 1u; k <= N_BEZIER_SEG; k = k + 1u) {
            let curr = bezier_at(p0, p1, p2, p3, f32(k) / f32(N_BEZIER_SEG));
            let seg = curr - prev;
            let len2 = max(dot(seg, seg), 1e-8);
            let u = clamp(dot(p - prev, seg) / len2, 0.0, 1.0);
            let proj = prev + u * seg;
            min_d = min(min_d, length(p - proj));
            prev = curr;
        }

        let inner = width - softness;
        let outer = width + softness;
        let a_geom = 1.0 - smoothstep(inner, outer, min_d);
        var a = clamp(a_geom * opacity, 0.0, 0.999);
        C_accum = C_accum + Ts * a * b.color.rgb;
        Ts = Ts * (1.0 - a);
    }
    textureStore(viewTex, global_id.xy, vec4f(C_accum, 1.0 - Ts));
}
