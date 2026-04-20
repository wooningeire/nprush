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
@group(0) @binding(2) var targetDepthTex: texture_2d<f32>;
@group(0) @binding(3) var targetEdgeTex: texture_2d<f32>;

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

// Compute splat edge by sampling eval_splats at neighboring pixels
fn eval_splat_edge(p: vec2f, pixel_size: vec2f) -> f32 {
    let tl = dot(eval_splats(p + vec2f(-1.0, -1.0) * pixel_size), vec3f(0.333));
    let tc = dot(eval_splats(p + vec2f( 0.0, -1.0) * pixel_size), vec3f(0.333));
    let tr = dot(eval_splats(p + vec2f( 1.0, -1.0) * pixel_size), vec3f(0.333));
    let ml = dot(eval_splats(p + vec2f(-1.0,  0.0) * pixel_size), vec3f(0.333));
    let mr = dot(eval_splats(p + vec2f( 1.0,  0.0) * pixel_size), vec3f(0.333));
    let bl = dot(eval_splats(p + vec2f(-1.0,  1.0) * pixel_size), vec3f(0.333));
    let bc = dot(eval_splats(p + vec2f( 0.0,  1.0) * pixel_size), vec3f(0.333));
    let br = dot(eval_splats(p + vec2f( 1.0,  1.0) * pixel_size), vec3f(0.333));
    
    let gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
    let gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
    
    return clamp(sqrt(gx * gx + gy * gy) * 8.0, 0.0, 1.0);
}

const STRIP_HEIGHT: f32 = 0.18;
const NUM_PANELS: f32 = 5.0;

@fragment
fn frag(v: VsOut) -> @location(0) vec4f {
    let dims = vec2f(textureDimensions(targetTex));
    
    // Bottom strip: show intermediate textures
    if (v.uv.y > (1.0 - STRIP_HEIGHT)) {
        let strip_uv_y = (v.uv.y - (1.0 - STRIP_HEIGHT)) / STRIP_HEIGHT;
        let panel_idx = floor(v.uv.x * NUM_PANELS);
        let panel_uv_x = fract(v.uv.x * NUM_PANELS);
        
        // Add thin separator between panels
        if (panel_uv_x < 0.005 || panel_uv_x > 0.995) {
            return vec4f(0.3, 0.3, 0.3, 1.0);
        }
        
        // Top border of the strip
        if (strip_uv_y < 0.01) {
            return vec4f(0.3, 0.3, 0.3, 1.0);
        }
        
        let sample_uv = vec2f(panel_uv_x, strip_uv_y);
        let depth_dims = vec2f(textureDimensions(targetDepthTex));
        let edge_dims = vec2f(textureDimensions(targetEdgeTex));
        
        if (panel_idx < 0.5) {
            // Panel 0: Target color
            let px = vec2i(sample_uv * dims);
            return textureLoad(targetTex, px, 0);
        } else if (panel_idx < 1.5) {
            // Panel 1: Splat color
            var p = sample_uv * 2.0 - 1.0;
            p.y = -p.y;
            return vec4f(eval_splats(p), 1.0);
        } else if (panel_idx < 2.5) {
            // Panel 2: Target depth
            let px = vec2i(sample_uv * depth_dims);
            let d = textureLoad(targetDepthTex, px, 0).r;
            return vec4f(d, d, d, 1.0);
        } else if (panel_idx < 3.5) {
            // Panel 3: Target edges
            let px = vec2i(sample_uv * edge_dims);
            let e = textureLoad(targetEdgeTex, px, 0).r;
            return vec4f(e, e, e, 1.0);
        } else {
            // Panel 4: Splat edges (computed live)
            var p = sample_uv * 2.0 - 1.0;
            p.y = -p.y;
            let pixel_size = 2.0 / dims;
            let e = eval_splat_edge(p, pixel_size);
            return vec4f(e, e, e, 1.0);
        }
    }
    
    let main_uv_y = v.uv.y / (1.0 - STRIP_HEIGHT);

    // Left half: target
    if (v.uv.x < 0.498) {
        let tx = v.uv.x * 2.0;
        let px = vec2i(vec2f(tx, main_uv_y) * dims);
        return textureLoad(targetTex, px, 0);
    }

    // Separator
    if (v.uv.x < 0.502) {
        return vec4f(1.0);
    }

    // Right half: splats
    let su = vec2f((v.uv.x - 0.5) * 2.0, main_uv_y);
    var p = su * 2.0 - 1.0;
    p.y = -p.y;
    return vec4f(eval_splats(p), 1.0);
}
