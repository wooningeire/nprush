@group(0) @binding(0) var targetTex: texture_2d<f32>;
@group(0) @binding(1) var splatViewTex: texture_2d<f32>;
@group(0) @binding(2) var targetDepthTex: texture_2d<f32>;
@group(0) @binding(3) var targetEdgeTex: texture_2d<f32>;
@group(0) @binding(4) var bezierViewTex: texture_2d<f32>;
@group(0) @binding(5) var baseColorBezierViewTex: texture_2d<f32>;
@group(0) @binding(6) var colorBezierViewTex: texture_2d<f32>;
@group(0) @binding(8) var ptTex: texture_2d<f32>;

struct RenderUniforms {
    edge_beziers_enabled: u32,
    base_color_beziers_enabled: u32,
    color_beziers_enabled: u32,
    mesh_splats_enabled: u32,
    splats_enabled: u32,
    canvas_aspect: f32,
    fvc_mode: u32,
    _pad2: f32,
}
@group(0) @binding(7) var<uniform> uniforms: RenderUniforms;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vert(@builtin(vertex_index) vi: u32) -> VsOut {
    let pos = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f( 3.0, -1.0),
        vec2f(-1.0,  3.0)
    );
    let x = pos[vi].x;
    let y = pos[vi].y;

    var o: VsOut;
    o.pos = vec4f(x, y, 0.0, 1.0);
    o.uv = vec2f(x * 0.5 + 0.5, 0.5 - y * 0.5);
    return o;
}

fn fitInPanel(panel_uv: vec2f, panel_aspect: f32, src_aspect: f32) -> vec2f {
    let rel = panel_aspect / src_aspect;
    if (rel > 1.0) {
        let content_width = 1.0 / rel;
        let margin = (1.0 - content_width) * 0.5;
        if (panel_uv.x < margin || panel_uv.x > 1.0 - margin) { return vec2f(-1.0); }
        return vec2f((panel_uv.x - margin) / content_width, panel_uv.y);
    } else {
        let content_height = rel;
        let margin = (1.0 - content_height) * 0.5;
        if (panel_uv.y < margin || panel_uv.y > 1.0 - margin) { return vec2f(-1.0); }
        return vec2f(panel_uv.x, (panel_uv.y - margin) / content_height);
    }
}

@fragment
fn frag_target(v: VsOut) -> @location(0) vec4f {
    const bg = vec4f(0.05, 0.05, 0.05, 0);
    let splat_dims = vec2f(textureDimensions(splatViewTex));
    let splat_aspect = splat_dims.x / splat_dims.y;
    
    let uv = fitInPanel(v.uv, uniforms.canvas_aspect, splat_aspect);
    if (uv.x < 0.0) { return bg; }

    let pt_dims = vec2f(textureDimensions(ptTex));
    let px = vec2i(uv * pt_dims);
    let pt_color = textureLoad(ptTex, clamp(px, vec2i(0), vec2i(pt_dims) - 1), 0);
    let raster_dims = vec2f(textureDimensions(targetTex));
    let raster_px = vec2i(uv * raster_dims);
    let raster = textureLoad(targetTex, clamp(raster_px, vec2i(0), vec2i(raster_dims) - 1), 0);
    return select(raster, pt_color, pt_color.a > 0.0);
}

@fragment
fn frag_composite(v: VsOut) -> @location(0) vec4f {
    const bg = vec4f(0.05, 0.05, 0.05, 0);
    let splat_dims = vec2f(textureDimensions(splatViewTex));
    let splat_aspect = splat_dims.x / splat_dims.y;
    
    let uv = fitInPanel(v.uv, uniforms.canvas_aspect, splat_aspect);
    if (uv.x < 0.0) { return bg; }

    let splat_px = vec2i(uv * splat_dims);
    let splat_color = textureLoad(splatViewTex, splat_px, 0).rgb;
    
    let bezier_px = vec2i(uv * vec2f(textureDimensions(bezierViewTex)));
    let edge_a = clamp(textureLoad(bezierViewTex, bezier_px, 0).a, 0.0, 1.0);
    
    let base_color_bezier_px = vec2i(uv * vec2f(textureDimensions(baseColorBezierViewTex)));
    let base_color_bezier = textureLoad(baseColorBezierViewTex, base_color_bezier_px, 0);
    
    let color_bezier_px = vec2i(uv * vec2f(textureDimensions(colorBezierViewTex)));
    let color_bezier = textureLoad(colorBezierViewTex, color_bezier_px, 0);

    var composite: vec3f;

    if (uniforms.fvc_mode > 0u) {
        // --- Form / Value / Color composite ---
        let luma_w = vec3f(0.2126, 0.7152, 0.0722);

        // Value base: splats forced to grayscale
        let splat_luma = dot(splat_color, luma_w);
        var value = select(vec3f(0.05), vec3f(splat_luma), uniforms.splats_enabled > 0u);

        // Coarse bezier: value strokes (forced grayscale)
        let coarse_luma = dot(base_color_bezier.rgb / max(base_color_bezier.a, 0.001), luma_w) * base_color_bezier.a;
        let coarse_gray = vec4f(vec3f(coarse_luma), base_color_bezier.a);
        value = select(value, value * (1.0 - coarse_gray.a) + coarse_gray.rgb, uniforms.base_color_beziers_enabled > 0u);

        // Color washes over the grayscale value base
        composite = select(value, value * (1.0 - color_bezier.a) + color_bezier.rgb, uniforms.color_beziers_enabled > 0u);

        // Edge darkening
        const EDGE_DARKEN: f32 = 0.5;
        composite = select(composite, composite - edge_a * EDGE_DARKEN, uniforms.edge_beziers_enabled > 0u);
    } else {
        // --- Original scale-based composite ---
        let base = select(vec3f(0.05), splat_color, uniforms.splats_enabled > 0u);
        composite = base;
        composite = select(composite, composite * (1.0 - base_color_bezier.a) + base_color_bezier.rgb, uniforms.base_color_beziers_enabled > 0u);
        composite = select(composite, composite * (1.0 - color_bezier.a) + color_bezier.rgb, uniforms.color_beziers_enabled > 0u);
        const EDGE_DARKEN: f32 = 0.5;
        composite = select(composite, composite - edge_a * EDGE_DARKEN, uniforms.edge_beziers_enabled > 0u);
    }

    return vec4f(composite, 1.0);
}

