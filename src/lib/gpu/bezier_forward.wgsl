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
@group(0) @binding(1) var<uniform> uniforms: ForwardUniforms;
@group(0) @binding(2) var brush_sampler: sampler;
@group(0) @binding(3) var brush_texture: texture_2d<f32>;

const N_SEG: u32 = 16u;

fn bezier_at(p0: vec2f, p1: vec2f, p2: vec2f, p3: vec2f, t: f32) -> vec2f {
    let omt = 1.0 - t;
    return omt*omt*omt * p0
         + 3.0 * omt*omt * t * p1
         + 3.0 * omt * t*t * p2
         + t*t*t * p3;
}

fn project_to_screen(vp: mat4x4f, pos3: vec3f, aspect: f32) -> vec3f {
    // Returns (x_aspect_corrected, y_ndc, w)
    let clip = vp * vec4f(pos3, 1.0);
    return vec3f(clip.x / clip.w * aspect, clip.y / clip.w, clip.w);
}

struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) @interpolate(flat) bezier_idx: u32,
    // Bounding box in aspect-corrected screen space, passed to fragment
    @location(1) p_screen: vec2f,
}

@vertex
fn vs_main(
    @builtin(instance_index) ii: u32,
    @builtin(vertex_index) vi: u32
) -> VsOut {
    // Draw back-to-front: mirror the gaussian approach where index 0 is drawn last
    // (front-most). The optimizer pushes active curves toward lower indices, so
    // reversing instance order gives an approximate depth sort for free.
    let bezier_idx = NUM_BEZIERS - 1u - ii;
    let b = beziers.items[bezier_idx];
    let aspect = uniforms.dims.x / uniforms.dims.y;

    let proj0 = project_to_screen(uniforms.vp, b.p0.xyz, aspect);
    let proj1 = project_to_screen(uniforms.vp, b.p1.xyz, aspect);
    let proj2 = project_to_screen(uniforms.vp, b.p2.xyz, aspect);
    let proj3 = project_to_screen(uniforms.vp, b.p3.xyz, aspect);

    // Cull if ANY control point is behind the camera (w <= 0).
    // When w <= 0 the perspective divide flips sign, producing a mirrored
    // screen-space position that makes the AABB span the entire viewport.
    // Curves that straddle the near plane must be culled here; the optimizer
    // will pull them back into view via the backward pass.
    if (proj0.z <= 0.0 || proj1.z <= 0.0 || proj2.z <= 0.0 || proj3.z <= 0.0) {
        var out: VsOut;
        out.pos = vec4f(0.0, 0.0, 0.0, -1.0); // degenerate
        out.bezier_idx = bezier_idx;
        out.p_screen = vec2f(0.0);
        return out;
    }

    let p0 = proj0.xy;
    let p1 = proj1.xy;
    let p2 = proj2.xy;
    let p3 = proj3.xy;

    let width = max(b.p0.w, 0.0001);
    let softness = max(b.p1.w, 0.0001);
    let pad = width + softness;

    // Tight AABB around the bezier hull + padding.
    // Clamp to a generous screen-space bound so a single off-screen control
    // point can never produce a quad that covers the entire viewport.
    let SCREEN_BOUND = 4.0; // aspect-corrected NDC units; well beyond any visible pixel
    let min_p = max(min(min(p0, p1), min(p2, p3)) - vec2f(pad), vec2f(-SCREEN_BOUND));
    let max_p = min(max(max(p0, p1), max(p2, p3)) + vec2f(pad), vec2f( SCREEN_BOUND));

    // Emit a quad (triangle-strip: 4 verts) covering the AABB
    // vi: 0=(min_x,max_y), 1=(min_x,min_y), 2=(max_x,max_y), 3=(max_x,min_y)
    let corners = array<vec2f, 4>(
        vec2f(min_p.x, max_p.y),
        vec2f(min_p.x, min_p.y),
        vec2f(max_p.x, max_p.y),
        vec2f(max_p.x, min_p.y),
    );
    let c = corners[vi];

    // Convert from aspect-corrected NDC back to standard NDC for clip space
    let ndc = vec2f(c.x / aspect, c.y);

    var out: VsOut;
    out.pos = vec4f(ndc, 0.0, 1.0);
    out.bezier_idx = bezier_idx;
    out.p_screen = c; // aspect-corrected screen position of this fragment
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    let ii = in.bezier_idx;
    let b = beziers.items[ii];
    let aspect = uniforms.dims.x / uniforms.dims.y;

    let proj0 = project_to_screen(uniforms.vp, b.p0.xyz, aspect);
    let proj1 = project_to_screen(uniforms.vp, b.p1.xyz, aspect);
    let proj2 = project_to_screen(uniforms.vp, b.p2.xyz, aspect);
    let proj3 = project_to_screen(uniforms.vp, b.p3.xyz, aspect);

    let p0 = proj0.xy;
    let p1 = proj1.xy;
    let p2 = proj2.xy;
    let p3 = proj3.xy;

    let p = in.p_screen;

    // Walk segments to find closest point — identical to backward pass
    var min_d = 1e9;
    var min_k = 1u;
    var min_u = 0.0;
    var min_signed_cross = 0.0;
    var prev = p0;
    for (var k = 1u; k <= N_SEG; k++) {
        let curr = bezier_at(p0, p1, p2, p3, f32(k) / f32(N_SEG));
        let seg = curr - prev;
        let len2 = max(dot(seg, seg), 1e-8);
        let u = clamp(dot(p - prev, seg) / len2, 0.0, 1.0);
        let proj_pt = prev + u * seg;
        let diff = p - proj_pt;
        let d = length(diff);
        if (d < min_d) {
            min_d = d;
            min_k = k;
            min_u = u;
            // Signed cross distance: positive = left of stroke direction, negative = right
            let seg_len = sqrt(len2);
            let seg_dir = seg / seg_len;
            min_signed_cross = (diff.x * (-seg_dir.y) + diff.y * seg_dir.x);
        }
        prev = curr;
    }

    let t = (f32(min_k - 1u) + min_u) / f32(N_SEG);
    let dt = t - 0.5;
    let pressure = 1.0 - 4.0 * dt * dt;

    let width = max(b.p0.w, 0.0001);
    let softness = max(b.p1.w, 0.0001);
    let local_width = width * pressure;
    let local_softness = softness * pressure;
    let local_opacity = b.color.a * pressure;

    let inner = local_width - local_softness;
    let outer = local_width + local_softness;
    let a_geom = 1.0 - smoothstep(inner, outer, min_d);

    // Sample brush texture: u = t (along stroke), v = signed cross distance normalized to [0,1]
    // Cross distance is normalized by local_width so the brush fills the stroke width.
    let brush_u = t;
    let brush_v = clamp(min_signed_cross / max(local_width + local_softness, 1e-6) * 0.5 + 0.5, 0.0, 1.0);
    let brush_alpha = textureSample(brush_texture, brush_sampler, vec2f(brush_u, brush_v)).r;

    let a = clamp(a_geom * brush_alpha * local_opacity, 0.0, 0.999);

    if (a < 0.001) { discard; }

    return vec4f(b.color.rgb * a, a);
}
