struct Splat {
    transform: vec4f,
    color: vec4f,
    rot_pad: vec4f,
}

struct SplatArray {
    splats: array<Splat, NUM_SPLATS>,
}

@group(0) @binding(0) var targetTex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> splats: SplatArray;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vert(@builtin(vertex_index) vi: u32) -> VsOut {
    // Fullscreen quad: 2 triangles, 6 vertices
    // tri 0: v0(-1,-1) v1(1,-1) v2(-1,1)
    // tri 1: v3(-1,1) v4(1,-1) v5(1,1)
    var x: f32; var y: f32;
    switch vi {
        case 0u: { x = -1.0; y = -1.0; }
        case 1u: { x =  1.0; y = -1.0; }
        case 2u: { x = -1.0; y =  1.0; }
        case 3u: { x = -1.0; y =  1.0; }
        case 4u: { x =  1.0; y = -1.0; }
        default: { x =  1.0; y =  1.0; }
    }

    var o: VsOut;
    o.pos = vec4f(x, y, 0.0, 1.0);
    o.uv = vec2f(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return o;
}

fn eval_splats(p: vec2f) -> vec3f {
    var Ts = 1.0;
    var c = vec3f(0.0);
    for (var i = 0u; i < NUM_SPLATS; i = i + 1u) {
        let s = splats.splats[i];
        let d = p - s.transform.xy;
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
        c = c + Ts * a * s.color.rgb;
        Ts = Ts * (1.0 - a);
    }
    return c + Ts * vec3f(0.1);
}

@fragment
fn frag(v: VsOut) -> @location(0) vec4f {
    let dims = vec2f(textureDimensions(targetTex));
    let aspect = dims.x / dims.y;

    // Left half: target
    if (v.uv.x < 0.498) {
        let tx = v.uv.x * 2.0;
        let px = vec2i(vec2f(tx, v.uv.y) * dims);
        return textureLoad(targetTex, px, 0);
    }

    // Separator
    if (v.uv.x < 0.502) {
        return vec4f(1.0);
    }

    // Right half: splats
    let su = vec2f((v.uv.x - 0.5) * 2.0, v.uv.y);
    var p = su * 2.0 - 1.0;
    p.y = -p.y;
    return vec4f(eval_splats(p), 1.0);
}
