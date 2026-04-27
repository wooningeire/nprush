// Differentiable rasterization of cubic bezier curves.
//
// V1 simplifications:
//   * Polyline approximation: each curve is sampled at N_SEG+1 points and
//     treated as N_SEG line segments. Distance-to-segment has a clean
//     analytical gradient.
//   * Hard argmin over segments (subgradient at ridges; in practice fine).
//   * Alpha = 1 - smoothstep(width-softness, width+softness, dist).
//   * Standard front-to-back alpha-compositing identical to the splat backward.

struct Bezier {
    p0_p1: vec4f,           // p0.xy, p1.xy
    p2_p3: vec4f,           // p2.xy, p3.xy
    color: vec4f,           // rgba
    width_soft_pad: vec4f,  // width, softness, _, _
}

struct BezierArray {
    items: array<Bezier, NUM_BEZIERS>,
}

struct GradArray {
    data: array<atomic<i32>, NUM_BEZIER_PARAMS>,
}

@group(0) @binding(0) var<storage, read> beziers: BezierArray;
@group(0) @binding(1) var<storage, read_write> grads: GradArray;
@group(0) @binding(2) var targetTex: texture_2d<f32>;
@group(0) @binding(3) var targetEdgeTex: texture_2d<f32>;

const N_SEG: u32 = 16u;

fn bezier_at(p0: vec2f, p1: vec2f, p2: vec2f, p3: vec2f, t: f32) -> vec2f {
    let omt = 1.0 - t;
    return omt*omt*omt * p0
         + 3.0 * omt*omt * t * p1
         + 3.0 * omt * t*t * p2
         + t*t*t * p3;
}

// Bernstein basis weights (B0..B3) at parameter t.
fn bernstein(t: f32) -> vec4f {
    let omt = 1.0 - t;
    return vec4f(omt*omt*omt, 3.0*omt*omt*t, 3.0*omt*t*t, t*t*t);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let dims = textureDimensions(targetTex);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    // Match the splat backward's pixel-to-curve coordinate mapping so both
    // layers share the same domain: p in [-aspect, aspect] x [-1, 1].
    let aspect = f32(dims.x) / f32(dims.y);
    let uv = (vec2f(global_id.xy) + vec2f(0.5)) / vec2f(dims.xy);
    var p = uv * 2.0 - 1.0;
    p.y = -p.y;
    p.x = p.x * aspect;

    let tgt_color = textureLoad(targetTex, global_id.xy, 0).rgb;
    let tgt_edge = textureLoad(targetEdgeTex, global_id.xy, 0).r;

    // ------------------------------------------------------------------
    // Forward pass: compute alphas, transmittances, and remember which
    // polyline segment was closest for each curve (needed for backward).
    // ------------------------------------------------------------------
    var alphas: array<f32, NUM_BEZIERS>;
    var min_seg: array<u32, NUM_BEZIERS>;
    var Ts: array<f32, NUM_BEZIERS_PLUS_ONE>;
    Ts[0] = 1.0;
    var C_pred = vec3f(0.0);

    for (var i = 0u; i < NUM_BEZIERS; i = i + 1u) {
        let b = beziers.items[i];
        let p0 = b.p0_p1.xy;
        let p1 = b.p0_p1.zw;
        let p2 = b.p2_p3.xy;
        let p3 = b.p2_p3.zw;
        let width = max(b.width_soft_pad.x, 0.001);
        let softness = max(b.width_soft_pad.y, 0.001);
        let opacity = b.color.a;

        var min_d = 1e9;
        var min_k = 1u;
        var prev = p0;
        for (var k = 1u; k <= N_SEG; k = k + 1u) {
            let curr = bezier_at(p0, p1, p2, p3, f32(k) / f32(N_SEG));
            let seg = curr - prev;
            let len2 = max(dot(seg, seg), 1e-8);
            let u = clamp(dot(p - prev, seg) / len2, 0.0, 1.0);
            let proj = prev + u * seg;
            let d = length(p - proj);
            if (d < min_d) {
                min_d = d;
                min_k = k;
            }
            prev = curr;
        }

        let inner = width - softness;
        let outer = width + softness;
        let a_geom = 1.0 - smoothstep(inner, outer, min_d);
        var a = clamp(a_geom * opacity, 0.0, 0.999);

        alphas[i] = a;
        min_seg[i] = min_k;
        C_pred += Ts[i] * a * b.color.rgb;
        Ts[i + 1u] = Ts[i] * (1.0 - a);
    }

    // Background is black for the edge layer (target is dark outside the
    // silhouette); keep the chain rule terms consistent.
    let background = vec3f(0.0);
    C_pred += Ts[NUM_BEZIERS] * background;

    // ------------------------------------------------------------------
    // Backward pass: chain rule from L = ||C_pred - tgt||^2 down to each
    // control point / color channel / width / softness.
    // ------------------------------------------------------------------
    // Plain MSE against the (already-thresholded) target edge image. Earlier
    // revisions multiplied dC by an `edge_weight = 1 + tgt_edge * k` boost
    // to "lock onto edges faster," but every nonzero k biases the
    // equilibrium toward fat strokes — matching an edge pixel (weight 1+k)
    // outvalues the off-edge penalty (weight 1) for being a bit too wide,
    // so curves prefer "definitely covers the edge" over "just covers the
    // edge." With k=0 the on-edge gain and off-edge cost balance, so the
    // optimum width is *exactly* the target edge thickness rather than
    // wider. The thresholded target image already has zero-loss outside
    // edges, so we don't lose much in finding rate.
    let dC = 2.0 * (C_pred - tgt_color);

    var dT = dot(dC, background);

    let FP_SCALE_POS = 10000.0;
    let FP_SCALE_COL = 100000.0;

    for (var j = 0u; j < NUM_BEZIERS; j = j + 1u) {
        let i = NUM_BEZIERS_MINUS_ONE - j;
        let b = beziers.items[i];
        let p0 = b.p0_p1.xy;
        let p1 = b.p0_p1.zw;
        let p2 = b.p2_p3.xy;
        let p3 = b.p2_p3.zw;
        let width = max(b.width_soft_pad.x, 0.001);
        let softness = max(b.width_soft_pad.y, 0.001);
        let opacity = b.color.a;
        let color = b.color.rgb;

        let a = alphas[i];
        let T_prev = Ts[i];

        let dColor = dC * (T_prev * a);
        let da = dT * (-T_prev) + dot(dC, T_prev * color);
        dT = dT * (1.0 - a) + dot(dC, a * color);

        // Reconstruct the closest segment from min_seg[i].
        let k = min_seg[i];
        let t_prev = f32(k - 1u) / f32(N_SEG);
        let t_curr = f32(k) / f32(N_SEG);
        let prev_pt = bezier_at(p0, p1, p2, p3, t_prev);
        let curr_pt = bezier_at(p0, p1, p2, p3, t_curr);
        let seg = curr_pt - prev_pt;
        let len2 = max(dot(seg, seg), 1e-8);
        let u_clamped = clamp(dot(p - prev_pt, seg) / len2, 0.0, 1.0);
        let proj = prev_pt + u_clamped * seg;
        let d_vec = p - proj;
        let d = max(length(d_vec), 1e-6);

        // a = (1 - smoothstep(inner, outer, d)) * opacity
        let inner = width - softness;
        let outer = width + softness;
        let denom = max(outer - inner, 1e-6);
        let x = clamp((d - inner) / denom, 0.0, 1.0);
        let smoothstep_deriv = 6.0 * x * (1.0 - x) / denom;
        let in_softband = (d > inner) && (d < outer);

        // Stray kill: MSE gradients flow through front-to-back compositing
        // (C_pred += T_prev * a * color, d/da uses T_prev). A curve buried
        // under many slightly-opaque layers can have T_prev ≈ 0 on most
        // pixels, so its da from the photometric loss vanishes even when
        // the stroke is still visible (earlier layers already put white in
        // the pixel). Those curves then never get a signal to reduce
        // opacity and sit around as a "grid" of unkillable garbage.
        //
        // Add an auxiliary loss (not composite-weighted):
        //   L_off = w * (1 - tgt_edge) * a
        // with a = a_geom * opacity, same as the forward pass.  dL/d(opacity)
        // = w * (1 - tgt_edge) * a_geom; on edge pixels tgt_edge ≈ 1, off;
        // in black, every visible stroke gets a direct opacity/geometry
        // gradient.  dL through a_geom to width, softness, and position uses
        // the same chain as the photometric d(width) terms with
        // (da + w*(1 - tgt_edge)) in place of da — but we must NOT add this
        // to the transmittance adjoint dT, which is only for the MSE.
        let a_geom = a / max(opacity, 1e-4);
        let OFF_EDGE_ALPHA = 0.35;
        let off_w = OFF_EDGE_ALPHA * (1.0 - tgt_edge);

        var d_opacity = da * (1.0 - smoothstep(inner, outer, d)) + off_w * a_geom;

        // Distance gradient flows only inside the soft band (outside it the
        // alpha is constant 0 or opacity, so the curve is locally insensitive).
        var dD = 0.0;
        var dWidth = 0.0;
        var dSoft = 0.0;
        if (in_softband) {
            dD = -(da + off_w) * opacity * smoothstep_deriv;
            // d(alpha)/d(width)    =  opacity * ss' / (2*softness)
            // d(alpha)/d(softness) = -opacity * ss' * (width - d) / (2*softness^2)
            dWidth = (da + off_w) * opacity * smoothstep_deriv / denom;
            dSoft = -(da + off_w) * opacity * smoothstep_deriv * (width - d)
                / max(2.0 * softness * softness, 1e-6);
        }

        // d = |p - proj|  =>  d(d)/d(proj) = -d_vec / d
        let dProj = -dD * d_vec / d;
        // proj = (1-u)*prev_pt + u*curr_pt, with u treated as constant
        // (envelope theorem: in the interior u minimizes d, on the clamp
        // boundaries the parameter is locked, so partials through u vanish).
        let dPrevPt = (1.0 - u_clamped) * dProj;
        let dCurrPt = u_clamped * dProj;

        // Bernstein-basis chain rule from segment endpoints to control points.
        let B_prev = bernstein(t_prev);
        let B_curr = bernstein(t_curr);
        let dP0 = B_prev.x * dPrevPt + B_curr.x * dCurrPt;
        let dP1 = B_prev.y * dPrevPt + B_curr.y * dCurrPt;
        let dP2 = B_prev.z * dPrevPt + B_curr.z * dCurrPt;
        let dP3 = B_prev.w * dPrevPt + B_curr.w * dCurrPt;

        let base = i * 14u;
        atomicAdd(&grads.data[base + 0u], i32(dP0.x * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 1u], i32(dP0.y * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 2u], i32(dP1.x * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 3u], i32(dP1.y * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 4u], i32(dP2.x * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 5u], i32(dP2.y * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 6u], i32(dP3.x * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 7u], i32(dP3.y * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 8u], i32(dColor.r * FP_SCALE_COL));
        atomicAdd(&grads.data[base + 9u], i32(dColor.g * FP_SCALE_COL));
        atomicAdd(&grads.data[base + 10u], i32(dColor.b * FP_SCALE_COL));
        atomicAdd(&grads.data[base + 11u], i32(d_opacity * FP_SCALE_COL));
        atomicAdd(&grads.data[base + 12u], i32(dWidth * FP_SCALE_POS));
        atomicAdd(&grads.data[base + 13u], i32(dSoft * FP_SCALE_POS));
    }
}
