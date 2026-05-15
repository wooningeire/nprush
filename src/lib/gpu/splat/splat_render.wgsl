@group(0) @binding(0) var targetTex: texture_2d<f32>;
@group(0) @binding(1) var splatViewTex: texture_2d<f32>;
@group(0) @binding(2) var targetDepthTex: texture_2d<f32>;
@group(0) @binding(3) var targetEdgeTex: texture_2d<f32>;
@group(0) @binding(4) var bezierViewTex: texture_2d<f32>;
@group(0) @binding(5) var baseColorBezierViewTex: texture_2d<f32>;
@group(0) @binding(6) var colorBezierViewTex: texture_2d<f32>;
@group(0) @binding(8) var ptTex: texture_2d<f32>;

struct RenderUniforms {
    edge_beziers_enabled: f32,
    base_color_beziers_enabled: f32,
    color_beziers_enabled: f32,
    mesh_splats_enabled: f32,
    splats_enabled: f32,
    panel_mode: f32,
    canvas_aspect: f32,
    _pad: f32,
}
@group(0) @binding(7) var<uniform> uniforms: RenderUniforms;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vert(@builtin(vertex_index) vi: u32) -> VsOut {
    // Fullscreen quad: 2 triangles, 6 vertices
    // tri 0: v0(-1,-1) v1(1,-1) v2(-1,1)
    // tri 1: v3(-1,1) v4(1,-1) v5(1,1)
    let quad_x = array<f32, 6>(-1.0,  1.0, -1.0, -1.0,  1.0,  1.0);
    let quad_y = array<f32, 6>(-1.0, -1.0,  1.0,  1.0, -1.0,  1.0);
    let x = quad_x[vi];
    let y = quad_y[vi];

    var o: VsOut;
    o.pos = vec4f(x, y, 0.0, 1.0);
    o.uv = vec2f(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return o;
}

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
    let mode = u32(uniforms.panel_mode);
    let bg = vec4f(0.05, 0.05, 0.05, 1.0);

    let splat_dims = vec2f(textureDimensions(splatViewTex));
    let splat_aspect = splat_dims.x / splat_dims.y;
    
    // Use fitInPanel to avoid aspect distortion in different canvases
    var uv = v.uv;

    // Target
    if (mode == 0u) {
        uv = fitInPanel(v.uv, uniforms.canvas_aspect, splat_aspect);
        if (uv.x < 0.0) { return bg; }
        let pt_dims = vec2f(textureDimensions(ptTex));
        let px = vec2i(uv * pt_dims);
        let pt_color = textureLoad(ptTex, clamp(px, vec2i(0), vec2i(pt_dims) - 1), 0);
        let raster_dims = vec2f(textureDimensions(targetTex));
        let raster_px = vec2i(uv * raster_dims);
        let raster = textureLoad(targetTex, clamp(raster_px, vec2i(0), vec2i(raster_dims) - 1), 0);
        return select(raster, pt_color, pt_color.a > 0.0);
    }

    // Splats Composite
    if (mode == 1u) {
        uv = fitInPanel(v.uv, uniforms.canvas_aspect, splat_aspect);
        if (uv.x < 0.0) { return bg; }
        let splat_px = vec2i(uv * splat_dims);
        let splat_color = textureLoad(splatViewTex, splat_px, 0).rgb;
        let base = select(vec3f(0.05), splat_color, uniforms.splats_enabled > 0.5);
        
        let bezier_px = vec2i(uv * vec2f(textureDimensions(bezierViewTex)));
        let edge_a = clamp(textureLoad(bezierViewTex, bezier_px, 0).a, 0.0, 1.0);
        
        let base_color_bezier_px = vec2i(uv * vec2f(textureDimensions(baseColorBezierViewTex)));
        let base_color_bezier = textureLoad(baseColorBezierViewTex, base_color_bezier_px, 0);
        
        let color_bezier_px = vec2i(uv * vec2f(textureDimensions(colorBezierViewTex)));
        let color_bezier = textureLoad(colorBezierViewTex, color_bezier_px, 0);
        
        var composite = base;
        composite = select(composite, composite * (1.0 - base_color_bezier.a) + base_color_bezier.rgb, uniforms.base_color_beziers_enabled > 0.5);
        composite = select(composite, composite * (1.0 - color_bezier.a) + color_bezier.rgb, uniforms.color_beziers_enabled > 0.5);
        const EDGE_DARKEN: f32 = 0.5;
        composite = select(composite, composite - edge_a * EDGE_DARKEN, uniforms.edge_beziers_enabled > 0.5);

        return vec4f(composite, 1.0);
    }

    // Splat Color
    if (mode == 2u) {
        uv = fitInPanel(v.uv, uniforms.canvas_aspect, splat_aspect);
        if (uv.x < 0.0) { return bg; }
        let px = vec2i(uv * splat_dims);
        return textureLoad(splatViewTex, px, 0);
    }

    // Target Depth
    if (mode == 3u) {
        let depth_dims = vec2f(textureDimensions(targetDepthTex));
        let depth_aspect = depth_dims.x / depth_dims.y;
        uv = fitInPanel(v.uv, uniforms.canvas_aspect, depth_aspect);
        if (uv.x < 0.0) { return bg; }
        let px = vec2i(uv * depth_dims);
        let d = textureLoad(targetDepthTex, px, 0).r;
        return vec4f(d, d, d, 1.0);
    }

    // Target Edges
    if (mode == 4u) {
        let edge_dims = vec2f(textureDimensions(targetEdgeTex));
        let edge_aspect = edge_dims.x / edge_dims.y;
        uv = fitInPanel(v.uv, uniforms.canvas_aspect, edge_aspect);
        if (uv.x < 0.0) { return bg; }
        let px = vec2i(uv * edge_dims);
        let e = textureLoad(targetEdgeTex, px, 0).r;
        return vec4f(e, e, e, 1.0);
    }

    // Edge Beziers
    if (mode == 5u) {
        uv = fitInPanel(v.uv, uniforms.canvas_aspect, splat_aspect);
        if (uv.x < 0.0) { return bg; }
        let px = vec2i(uv * vec2f(textureDimensions(bezierViewTex)));
        let e = textureLoad(bezierViewTex, px, 0).a;
        return vec4f(e, e, e, 1.0);
    }

    // Coarse Bezier
    if (mode == 6u) {
        uv = fitInPanel(v.uv, uniforms.canvas_aspect, splat_aspect);
        if (uv.x < 0.0) { return bg; }
        let px = vec2i(uv * vec2f(textureDimensions(baseColorBezierViewTex)));
        return textureLoad(baseColorBezierViewTex, px, 0);
    }

    // Fine Bezier
    if (mode == 7u) {
        uv = fitInPanel(v.uv, uniforms.canvas_aspect, splat_aspect);
        if (uv.x < 0.0) { return bg; }
        let px = vec2i(uv * vec2f(textureDimensions(colorBezierViewTex)));
        return textureLoad(colorBezierViewTex, px, 0);
    }

    return vec4f(0.0);
}
