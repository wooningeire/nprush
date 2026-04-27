struct Splat {
    transform: vec4f,
    color: vec4f,
    rot_pad: vec4f,
}

struct SplatArray {
    splats: array<Splat, NUM_SPLATS>,
}

// Edge layer is now a set of cubic bezier curves (see GpuBezierOptimizerManager).
struct Bezier {
    p0_p1: vec4f,
    p2_p3: vec4f,
    color: vec4f,
    width_soft_pad: vec4f,
}

struct BezierArray {
    items: array<Bezier, NUM_BEZIERS>,
}

@group(0) @binding(0) var targetTex: texture_2d<f32>;
@group(0) @binding(1) var<storage, read> splats: SplatArray;
@group(0) @binding(2) var targetDepthTex: texture_2d<f32>;
@group(0) @binding(3) var targetEdgeTex: texture_2d<f32>;
@group(0) @binding(4) var<storage, read> beziers: BezierArray;

struct RenderUniforms {
    beziers_enabled: f32,
    _pad: vec3f,
}
@group(0) @binding(5) var<uniform> uniforms: RenderUniforms;

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

const N_BEZIER_SEG: u32 = 16u;

fn bezier_at(p0: vec2f, p1: vec2f, p2: vec2f, p3: vec2f, t: f32) -> vec2f {
    let omt = 1.0 - t;
    return omt*omt*omt * p0
         + 3.0 * omt*omt * t * p1
         + 3.0 * omt * t*t * p2
         + t*t*t * p3;
}

// Evaluate the edge-layer bezier curves at point p in splat-space and return
// a single grayscale "edge intensity" value, alpha-composited front-to-back
// to match bezier_backward.wgsl. This is used both for the dedicated debug
// panel and as the alpha when compositing the edge layer over the color layer.
fn eval_beziers(p: vec2f) -> f32 {
    if (uniforms.beziers_enabled < 0.5) { return 0.0; }
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
    return c;
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
const NUM_PANELS: f32 = 6.0;

// Fit a source with given aspect into a panel with given aspect
fn fitInPanel(panel_uv: vec2f, panel_aspect: f32, src_aspect: f32) -> vec2f {
    let rel = panel_aspect / src_aspect;
    if (rel > 1.0) {
        // Panel is relatively wider: pillarbox
        let content_width = 1.0 / rel;
        let margin = (1.0 - content_width) * 0.5;
        if (panel_uv.x < margin || panel_uv.x > 1.0 - margin) {
            return vec2f(-1.0);
        }
        return vec2f((panel_uv.x - margin) / content_width, panel_uv.y);
    } else {
        // Panel is relatively taller: letterbox
        let content_height = rel;
        let margin = (1.0 - content_height) * 0.5;
        if (panel_uv.y < margin || panel_uv.y > 1.0 - margin) {
            return vec2f(-1.0);
        }
        return vec2f(panel_uv.x, (panel_uv.y - margin) / content_height);
    }
}

@fragment
fn frag(v: VsOut) -> @location(0) vec4f {
    // targetTex is sized to the visible main panel (half-width x height-minus-strip),
    // so its aspect IS the splat coordinate aspect. The full canvas aspect is recovered
    // by undoing the half-width and the strip-fraction adjustments.
    let dims = vec2f(textureDimensions(targetTex));
    let splat_aspect = dims.x / dims.y;
    let screen_aspect = splat_aspect * 2.0 * (1.0 - STRIP_HEIGHT);
    
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
        
        // Each panel's pixel aspect ratio
        let panel_aspect = screen_aspect / (NUM_PANELS * STRIP_HEIGHT);
        let panel_uv = vec2f(panel_uv_x, strip_uv_y);
        
        let depth_dims = vec2f(textureDimensions(targetDepthTex));
        let edge_dims = vec2f(textureDimensions(targetEdgeTex));
        let depth_aspect = depth_dims.x / depth_dims.y;
        let edge_aspect = edge_dims.x / edge_dims.y;
        let target_aspect = dims.x / dims.y;
        
        let bg = vec4f(0.05, 0.05, 0.05, 1.0);
        
        if (panel_idx < 0.5) {
            // Panel 0: Target color
            let fitted = fitInPanel(panel_uv, panel_aspect, target_aspect);
            if (fitted.x < 0.0) { return bg; }
            let px = vec2i(fitted * dims);
            return textureLoad(targetTex, px, 0);
        } else if (panel_idx < 1.5) {
            // Panel 1: Splat color, in the splats' native aspect-correct domain
            let fitted = fitInPanel(panel_uv, panel_aspect, splat_aspect);
            if (fitted.x < 0.0) { return bg; }
            var p = fitted * 2.0 - 1.0;
            p.y = -p.y;
            p.x = p.x * splat_aspect;
            return vec4f(eval_splats(p), 1.0);
        } else if (panel_idx < 2.5) {
            // Panel 2: Target depth
            let fitted = fitInPanel(panel_uv, panel_aspect, depth_aspect);
            if (fitted.x < 0.0) { return bg; }
            let px = vec2i(fitted * depth_dims);
            let d = textureLoad(targetDepthTex, px, 0).r;
            return vec4f(d, d, d, 1.0);
        } else if (panel_idx < 3.5) {
            // Panel 3: Target edges
            let fitted = fitInPanel(panel_uv, panel_aspect, edge_aspect);
            if (fitted.x < 0.0) { return bg; }
            let px = vec2i(fitted * edge_dims);
            let e = textureLoad(targetEdgeTex, px, 0).r;
            return vec4f(e, e, e, 1.0);
        } else if (panel_idx < 4.5) {
            // Panel 4: Splat edges (Sobel of color-splat output)
            let fitted = fitInPanel(panel_uv, panel_aspect, splat_aspect);
            if (fitted.x < 0.0) { return bg; }
            var p = fitted * 2.0 - 1.0;
            p.y = -p.y;
            p.x = p.x * splat_aspect;
            let pixel_size = vec2f(2.0 * splat_aspect, 2.0) / dims;
            let e = eval_splat_edge(p, pixel_size);
            return vec4f(e, e, e, 1.0);
        } else {
            // Panel 5: Edge layer (cubic beziers trained to reconstruct the
            // depth-edge image directly).
            let fitted = fitInPanel(panel_uv, panel_aspect, splat_aspect);
            if (fitted.x < 0.0) { return bg; }
            var p = fitted * 2.0 - 1.0;
            p.y = -p.y;
            p.x = p.x * splat_aspect;
            let e = eval_beziers(p);
            return vec4f(e, e, e, 1.0);
        }
    }
    
    let main_uv_y = v.uv.y / (1.0 - STRIP_HEIGHT);

    // Left half: target (full-res target texture, sampled across the half-width panel)
    if (v.uv.x < 0.498) {
        let tx = v.uv.x * 2.0;
        let px = vec2i(vec2f(tx, main_uv_y) * dims);
        return textureLoad(targetTex, px, 0);
    }

    // Separator
    if (v.uv.x < 0.502) {
        return vec4f(1.0);
    }

    // Right half: splats. The splats' coordinate domain is [-splat_aspect, splat_aspect] x [-1, 1],
    // and this panel's pixel aspect matches splat_aspect, so circles stay circles.
    let su = vec2f((v.uv.x - 0.5) * 2.0, main_uv_y);
    var p = su * 2.0 - 1.0;
    p.y = -p.y;
    p.x = p.x * splat_aspect;

    // Composite the edge-layer (cubic beziers) over the color-layer reconstruction.
    // eval_beziers returns grayscale "edge intensity"; we use that intensity
    // directly as alpha against a white overlay so dark non-edge regions reveal
    // the underlying color layer rather than being occluded by the curves'
    // accumulated coverage.
    let base = eval_splats(p);
    let edge_a = clamp(eval_beziers(p), 0.0, 1.0);
    var composite = base;
    if (uniforms.beziers_enabled > 0.5) {
        composite = mix(base, vec3f(1.0), edge_a);
    }
    return vec4f(composite, 1.0);
}

