struct Splat {
    transform: vec4f,
    color: vec4f,
    rot_pad: vec4f,
}

struct SplatArray {
    splats: array<Splat, NUM_SPLATS>,
}

@group(0) @binding(0) var<storage, read> splats: SplatArray;
@group(0) @binding(1) var viewTex: texture_storage_2d<rgba8unorm, write>;

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
    let color = c + Ts * vec3f(0.1);
    textureStore(viewTex, global_id.xy, vec4f(color, 1.0));
}
