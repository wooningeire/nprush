@group(0) @binding(0) var tex: texture_2d<f32>;

struct RenderUniforms {
    edge_beziers_enabled: f32,
    base_color_beziers_enabled: f32,
    color_beziers_enabled: f32,
    mesh_splats_enabled: f32,
    splats_enabled: f32,
    canvas_aspect: f32,
    _pad1: f32,
    _pad2: f32,
}
@group(0) @binding(1) var<uniform> uniforms: RenderUniforms;

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vert(@builtin(vertex_index) vi: u32) -> VsOut {
    let quad_x = array<f32, 6>(-1.0,  1.0, -1.0, -1.0,  1.0,  1.0);
    let quad_y = array<f32, 6>(-1.0, -1.0,  1.0,  1.0, -1.0,  1.0);
    let x = quad_x[vi];
    let y = quad_y[vi];

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
fn frag_blit(v: VsOut) -> @location(0) vec4f {
    const bg = vec4f(0.05, 0.05, 0.05, 0);
    let dims = vec2f(textureDimensions(tex));
    let src_aspect = dims.x / dims.y;
    
    let uv = fitInPanel(v.uv, uniforms.canvas_aspect, src_aspect);
    if (uv.x < 0.0) { return bg; }

    let px = vec2i(uv * dims);
    return textureLoad(tex, px, 0);
}

@fragment
fn frag_blit_r(v: VsOut) -> @location(0) vec4f {
    const bg = vec4f(0.05, 0.05, 0.05, 0);
    let dims = vec2f(textureDimensions(tex));
    let src_aspect = dims.x / dims.y;
    
    let uv = fitInPanel(v.uv, uniforms.canvas_aspect, src_aspect);
    if (uv.x < 0.0) { return bg; }

    let px = vec2i(uv * dims);
    let color = textureLoad(tex, px, 0);
    return vec4f(color.rrr, 1.0);
}

@fragment
fn frag_blit_a(v: VsOut) -> @location(0) vec4f {
    const bg = vec4f(0.05, 0.05, 0.05, 0);
    let dims = vec2f(textureDimensions(tex));
    let src_aspect = dims.x / dims.y;
    
    let uv = fitInPanel(v.uv, uniforms.canvas_aspect, src_aspect);
    if (uv.x < 0.0) { return bg; }

    let px = vec2i(uv * dims);
    let color = textureLoad(tex, px, 0);
    return vec4f(color.aaa, 1.0);
}
