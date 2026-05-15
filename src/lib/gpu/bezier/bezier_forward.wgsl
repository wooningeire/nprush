struct Bezier {
    p0: vec4f, // x, y, z, width
    p1: vec4f, // x, y, z, softness
    p2: vec4f, // x, y, z, _pad
    p3: vec4f, // x, y, z, _pad
    color: vec4f,
    sh1_r: vec4f,
    sh1_g: vec4f,
    sh1_b: vec4f,
    sh1_a: vec4f,
}

struct BezierArray {
    items: array<Bezier, {@NUM_BEZIERS}u>,
}

struct ForwardUniforms {
    vp: mat4x4f,
    dims: vec2f,
    _pad: vec2f,
    cam_world: vec4f,
}

@group(0) @binding(0) var<storage, read> beziers: BezierArray;
@group(0) @binding(1) var<uniform> uniforms: ForwardUniforms;
@group(0) @binding(2) var brush_sampler: sampler;
@group(0) @binding(3) var brush_texture: texture_2d<f32>;
@group(0) @binding(4) var<storage, read> sort_order: array<u32, {@NUM_BEZIERS}u>;

const N_SEG: u32 = {@BEZIER_POLY_SEG}u;
const SH_C1_B: f32 = 0.4886025119029199;

fn bezier_pos_world(b: Bezier, tt: f32) -> vec3f {
    let omt = 1.0 - tt;
    return omt*omt*omt*b.p0.xyz + 3.0*omt*omt*tt*b.p1.xyz + 3.0*omt*tt*tt*b.p2.xyz + tt*tt*tt*b.p3.xyz;
}

fn bezier_deriv_world(b: Bezier, tt: f32) -> vec3f {
    let omt = 1.0 - tt;
    return 3.0*omt*omt*(b.p1.xyz - b.p0.xyz) + 6.0*omt*tt*(b.p2.xyz - b.p1.xyz) + 3.0*tt*tt*(b.p3.xyz - b.p2.xyz);
}

fn bezier_dirs_sh(cam_xyz: vec3f, pos_w: vec3f, tang: vec3f) -> vec3f {
    let Vcam = normalize(cam_xyz - pos_w);
    var ez = tang;
    if (dot(ez, ez) < 1e-10) {
        ez = vec3f(1.0, 0.0, 0.0);
    } else {
        ez = normalize(ez);
    }
    var ex = cross(vec3f(0.0, 1.0, 0.0), ez);
    if (dot(ex, ex) < 1e-12) {
        ex = cross(vec3f(1.0, 0.0, 0.0), ez);
    }
    ex = normalize(ex);
    let ey = normalize(cross(ez, ex));
    return vec3f(dot(Vcam, ex), dot(Vcam, ey), dot(Vcam, ez));
}

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
    @location(2) @interpolate(flat) proj0: vec3f,
    @location(5) @interpolate(flat) proj1: vec3f,
    @location(8) @interpolate(flat) proj2: vec3f,
    @location(11) @interpolate(flat) proj3: vec3f,
}

@vertex
fn vs_main(
    @builtin(instance_index) ii: u32,
    @builtin(vertex_index) vi: u32
) -> VsOut {
    // Draw back-to-front
    let bezier_idx = sort_order[ii];
    let b = beziers.items[bezier_idx];
    let aspect = uniforms.dims.x / uniforms.dims.y;

    let proj0 = project_to_screen(uniforms.vp, b.p0.xyz, aspect);
    let proj1 = project_to_screen(uniforms.vp, b.p1.xyz, aspect);
    let proj2 = project_to_screen(uniforms.vp, b.p2.xyz, aspect);
    let proj3 = project_to_screen(uniforms.vp, b.p3.xyz, aspect);

    const DEPTH_NEAR_CULL = 0.1;
    if (proj0.z < DEPTH_NEAR_CULL || proj1.z < DEPTH_NEAR_CULL || proj2.z < DEPTH_NEAR_CULL || proj3.z < DEPTH_NEAR_CULL) {
        var out: VsOut;
        out.pos = vec4f(0.0, 0.0, 2.0, 1.0);
        out.bezier_idx = bezier_idx;
        out.p_screen = vec2f(0.0);
        out.proj0 = vec3f(0.0);
        out.proj1 = vec3f(0.0);
        out.proj2 = vec3f(0.0);
        out.proj3 = vec3f(0.0);
        return out;
    }

    let p0 = proj0.xy;
    let p1 = proj1.xy;
    let p2 = proj2.xy;
    let p3 = proj3.xy;

    // Perspective scaling: treat width/softness as world-space units.
    // Use average depth for bounding box expansion.
    let avg_w = (proj0.z + proj1.z + proj2.z + proj3.z) * 0.25;
    let inv_w = 1.0 / max(avg_w, 0.001);

    let width = max(b.p0.w, 0.0001) * inv_w;
    let softness = max(b.p1.w, 0.0001) * inv_w;
    let pad = width + softness;

    // Tight AABB around the bezier hull + padding.
    let SCREEN_BOUND = 4.0;
    let min_p = max(min(min(p0, p1), min(p2, p3)) - vec2f(pad), vec2f(-SCREEN_BOUND));
    let max_p = min(max(max(p0, p1), max(p2, p3)) + vec2f(pad), vec2f(SCREEN_BOUND));

    let corners = array<vec2f, 4>(
        vec2f(min_p.x, max_p.y),
        vec2f(min_p.x, min_p.y),
        vec2f(max_p.x, max_p.y),
        vec2f(max_p.x, min_p.y),
    );
    let c = corners[vi];

    let ndc = vec2f(c.x / aspect, c.y);

    var out: VsOut;
    out.pos = vec4f(ndc, 0.0, 1.0);
    out.bezier_idx = bezier_idx;
    out.p_screen = c;
    out.proj0 = proj0;
    out.proj1 = proj1;
    out.proj2 = proj2;
    out.proj3 = proj3;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    let ii = in.bezier_idx;
    let b = beziers.items[ii];

    let proj0 = in.proj0;
    let proj1 = in.proj1;
    let proj2 = in.proj2;
    let proj3 = in.proj3;

    let p0 = proj0.xy;
    let p1 = proj1.xy;
    let p2 = proj2.xy;
    let p3 = proj3.xy;

    let p = in.p_screen;

    // Find closest segment, caching the winning segment endpoints to avoid
    // re-evaluating bezier_at after the loop.
    var min_d2 = 1e9;
    var min_k = 1u;
    var min_u = 0.0;
    var seg_prev = p0;
    var seg_curr = p0;
    var prev = p0;
    for (var k = 1u; k <= N_SEG; k++) {
        let curr = bezier_at(p0, p1, p2, p3, f32(k) / f32(N_SEG));
        let seg = curr - prev;
        let len2 = max(dot(seg, seg), 1e-8);
        let u = clamp(dot(p - prev, seg) / len2, 0.0, 1.0);
        let proj_pt = prev + u * seg;
        let diff = p - proj_pt;
        let d2 = dot(diff, diff);
        if (d2 < min_d2) {
            min_d2 = d2;
            min_k = k;
            min_u = u;
            seg_prev = prev;
            seg_curr = curr;
        }
        prev = curr;
    }
    let min_d = sqrt(min_d2);

    // Compute geometric alpha first — cheap, needed for early discard.
    let t = (f32(min_k - 1u) + min_u) / f32(N_SEG);
    let dt = t - 0.5;
    let pressure = 1.0 - 4.0 * dt * dt;

    let omt = 1.0 - t;
    let w = omt * omt * omt * proj0.z
        + 3.0 * omt * omt * t * proj1.z
        + 3.0 * omt * t * t * proj2.z
        + t * t * t * proj3.z;
    let inv_w = 1.0 / max(w, 0.001);

    let width = max(b.p0.w, 0.0001) * inv_w;
    let softness = max(b.p1.w, 0.0001) * inv_w;
    let local_width = width * pressure;
    let local_softness = softness * pressure;

    let inner = local_width - local_softness;
    let outer = local_width + local_softness;
    let a_geom = 1.0 - smoothstep(inner, outer, min_d);

    // Early discard: skip SH + texture sample for fragments outside the soft band.
    // brush_alpha ≤ 1 and local_opacity ≤ pressure, so a_geom * pressure is an
    // upper bound on the final alpha.
    if (a_geom * pressure < 0.001) { discard; }

    // Brush UV — uses cached segment endpoints, no extra bezier_at calls.
    let best_seg = seg_curr - seg_prev;
    let best_len = max(length(best_seg), 1e-4);
    let best_dir = best_seg / best_len;
    let best_proj = seg_prev + min_u * best_seg;
    let best_diff = p - best_proj;
    let min_signed_cross = best_diff.x * (-best_dir.y) + best_diff.y * best_dir.x;

    // SH direction + opacity (expensive: normalize, cross, dot).
    let pos_w = bezier_pos_world(b, t);
    let dl_b = bezier_dirs_sh(uniforms.cam_world.xyz, pos_w, bezier_deriv_world(b, t));
    let lx_b = dl_b.x;
    let ly_b = dl_b.y;
    let lz_b = dl_b.z;
    let o_lin = b.color.a + SH_C1_B * (ly_b * b.sh1_a.x + lz_b * b.sh1_a.y + lx_b * b.sh1_a.z);
    let opacity = clamp(o_lin, 0.0, 1.0);
    let local_opacity = opacity * pressure;

    let brush_u = t;
    let brush_v = clamp(min_signed_cross / max(local_width + local_softness, 1e-6) * 0.5 + 0.5, 0.0, 1.0);
    let brush_alpha = textureSample(brush_texture, brush_sampler, vec2f(brush_u, brush_v)).r;

    let a = clamp(a_geom * brush_alpha * local_opacity, 0.0, 0.999);
    if (a < 0.001) { discard; }

    let rr_lin = b.color.r + SH_C1_B * (ly_b * b.sh1_r.x + lz_b * b.sh1_r.y + lx_b * b.sh1_r.z);
    let gg_lin = b.color.g + SH_C1_B * (ly_b * b.sh1_g.x + lz_b * b.sh1_g.y + lx_b * b.sh1_g.z);
    let bb_lin = b.color.b + SH_C1_B * (ly_b * b.sh1_b.x + lz_b * b.sh1_b.y + lx_b * b.sh1_b.z);
    let rgb_vis = max(vec3f(rr_lin, gg_lin, bb_lin), vec3f(0.0));

    return vec4f(rgb_vis * a, a);
}
