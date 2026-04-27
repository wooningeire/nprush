struct Bezier {
    p0_p1: vec4f,
    p2_p3: vec4f,
    color: vec4f,
    width_soft_pad: vec4f,
}

struct BezierArray {
    items: array<Bezier, NUM_BEZIERS>,
}

@group(0) @binding(0) var<storage, read> beziers: BezierArray;
@group(0) @binding(1) var viewTex: texture_storage_2d<rgba8unorm, write>;

const N_BEZIER_SEG: u32 = 16u;

fn bezier_at(p0: vec2f, p1: vec2f, p2: vec2f, p3: vec2f, t: f32) -> vec2f {
    let omt = 1.0 - t;
    return omt*omt*omt * p0
         + 3.0 * omt*omt * t * p1
         + 3.0 * omt * t*t * p2
         + t*t*t * p3;
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
    var c = 0.0;
    for (var i = 0u; i < NUM_BEZIERS; i = i + 1u) {
        let b = beziers.items[i];
        let p0 = b.p0_p1.xy;
        let p1 = b.p0_p1.zw;
        let p2 = b.p2_p3.xy;
        let p3 = b.p2_p3.zw;
        let width = max(b.width_soft_pad.x, 0.001);
        let softness = max(b.width_soft_pad.y, 0.001);

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
        var a = clamp(a_geom * b.color.a, 0.0, 0.999);
        c = c + Ts * a * dot(b.color.rgb, vec3f(0.333));
        Ts = Ts * (1.0 - a);
    }
    textureStore(viewTex, global_id.xy, vec4f(c, c, c, 1.0));
}
