@group(0) @binding(0) var targetTex: texture_2d<f32>;
@group(0) @binding(1) var splatViewTex: texture_2d<f32>;
@group(0) @binding(2) var targetDepthTex: texture_2d<f32>;
@group(0) @binding(3) var targetEdgeTex: texture_2d<f32>;
@group(0) @binding(4) var bezierViewTex: texture_2d<f32>;

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
            let px = vec2i(fitted * vec2f(textureDimensions(splatViewTex)));
            return textureLoad(splatViewTex, px, 0);
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
            
            let px = vec2i(fitted * vec2f(textureDimensions(splatViewTex)));
            let tl = dot(textureLoad(splatViewTex, px + vec2i(-1, -1), 0).rgb, vec3f(0.333));
            let tc = dot(textureLoad(splatViewTex, px + vec2i( 0, -1), 0).rgb, vec3f(0.333));
            let tr = dot(textureLoad(splatViewTex, px + vec2i( 1, -1), 0).rgb, vec3f(0.333));
            let ml = dot(textureLoad(splatViewTex, px + vec2i(-1,  0), 0).rgb, vec3f(0.333));
            let mr = dot(textureLoad(splatViewTex, px + vec2i( 1,  0), 0).rgb, vec3f(0.333));
            let bl = dot(textureLoad(splatViewTex, px + vec2i(-1,  1), 0).rgb, vec3f(0.333));
            let bc = dot(textureLoad(splatViewTex, px + vec2i( 0,  1), 0).rgb, vec3f(0.333));
            let br = dot(textureLoad(splatViewTex, px + vec2i( 1,  1), 0).rgb, vec3f(0.333));
            
            let gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
            let gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;
            
            let e = clamp(sqrt(gx * gx + gy * gy) * 8.0, 0.0, 1.0);
            return vec4f(e, e, e, 1.0);
        } else {
            // Panel 5: Edge layer
            let fitted = fitInPanel(panel_uv, panel_aspect, splat_aspect);
            if (fitted.x < 0.0) { return bg; }
            let px = vec2i(fitted * vec2f(textureDimensions(bezierViewTex)));
            let e = textureLoad(bezierViewTex, px, 0).r;
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

    // Right half: splats.
    let right_half_uv = vec2f((v.uv.x - 0.5) * 2.0, main_uv_y);
    let splat_px = vec2i(right_half_uv * vec2f(textureDimensions(splatViewTex)));
    let base = textureLoad(splatViewTex, splat_px, 0).rgb;
    
    let bezier_px = vec2i(right_half_uv * vec2f(textureDimensions(bezierViewTex)));
    let edge_a = clamp(textureLoad(bezierViewTex, bezier_px, 0).r, 0.0, 1.0);
    
    var composite = base;
    if (uniforms.beziers_enabled > 0.5) {
        composite = mix(base, vec3f(1.0), edge_a);
    }
    return vec4f(composite, 1.0);
}
