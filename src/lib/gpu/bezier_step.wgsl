// Adam optimizer for cubic-bezier parameters.
// Param layout per curve (14 params):
//   0..1   P0.xy
//   2..3   P1.xy
//   4..5   P2.xy
//   6..7   P3.xy
//   8..10  color.rgb
//   11     opacity
//   12     width
//   13     softness

struct Bezier {
    p0_p1: vec4f,
    p2_p3: vec4f,
    color: vec4f,
    width_soft_pad: vec4f,
}

struct BezierArray {
    items: array<Bezier, NUM_BEZIERS>,
}

struct GradArray {
    data: array<atomic<i32>, NUM_BEZIER_PARAMS>,
}

struct AdamState {
    m: array<f32, NUM_BEZIER_PARAMS>,
    v: array<f32, NUM_BEZIER_PARAMS>,
    t: f32,
    pad: vec3f,
}

struct ADCArray {
    grad_accum: array<f32, NUM_BEZIERS>,
}

@group(0) @binding(0) var<storage, read_write> beziers: BezierArray;
@group(0) @binding(1) var<storage, read_write> grads: GradArray;
@group(0) @binding(2) var<storage, read_write> adam: AdamState;
@group(0) @binding(3) var<storage, read_write> adc: ADCArray;

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let bid = global_id.x;

    let current_t = adam.t;
    workgroupBarrier();
    if (bid == 0u) {
        adam.t = current_t + 1.0;
    }

    if (bid >= NUM_BEZIERS) {
        return;
    }

    var b = beziers.items[bid];
    let base_idx = bid * 14u;
    let t = current_t + 1.0;

    // A curve killed by the previous step has alpha == 0 (the kill check at
    // the bottom writes literal 0, below the 0.05 clamp floor). If we ran
    // the normal Adam update on it, the opacity update at lp == 11 would
    // clamp(0 - update, 0.05, 0.99) and resurrect the curve to alpha = 0.05,
    // making it briefly visible (~1px @ display res) before the kill check
    // killed it again next frame. Detect that state and skip param updates
    // entirely so dead curves stay dead until ADC explicitly re-spawns them.
    let is_dead = b.color.a < 1e-6;

    // Accumulate the L2 norm of the control-point gradient over an ADC
    // period. ADC uses this to decide which curves to clone/split.
    var pos_grad_norm2 = 0.0;

    for (var lp = 0u; lp < 14u; lp = lp + 1u) {
        let pidx = base_idx + lp;
        // atomicExchange both reads and zeros so we drain the buffer either
        // way; we just don't apply Adam state / param updates if dead.
        let raw_grad = atomicExchange(&grads.data[pidx], 0);
        if (is_dead) { continue; }
        // Match the FP scale used in bezier_backward.wgsl.
        var fp_scale = 100000.0;
        if (lp < 8u || lp == 12u || lp == 13u) {
            fp_scale = 10000.0;
        }
        let grad = f32(raw_grad) / fp_scale / 16384.0;

        let beta1 = 0.9;
        let beta2 = 0.999;
        let epsilon = 1e-8;

        // Per-parameter learning rates. Control points are intentionally
        // slow: the polyline closest-segment argmin in bezier_backward gives
        // discontinuous gradients (the closest segment for a pixel switches
        // as the curve moves), and Adam momentum amplifies those jumps into
        // visible oscillation. Halving lr from the previous 0.02 setting
        // damps the oscillation without preventing convergence — strong,
        // sustained gradient still moves a curve at lr per step under Adam,
        // and the canvas is only 2 norm units wide.
        var lr = 0.01;
        if (lp < 8u) { lr = 0.01; }                  // control points
        else if (lp <= 10u) { lr = 0.02; }           // color rgb
        else if (lp == 11u) { lr = 0.01; }           // opacity
        else if (lp == 12u) { lr = 0.005; }          // width
        else if (lp == 13u) { lr = 0.005; }          // softness

        var m = adam.m[pidx];
        var v = adam.v[pidx];
        m = beta1 * m + (1.0 - beta1) * grad;
        v = beta2 * v + (1.0 - beta2) * grad * grad;
        adam.m[pidx] = m;
        adam.v[pidx] = v;

        let m_hat = m / (1.0 - pow(beta1, t));
        let v_hat = v / (1.0 - pow(beta2, t));
        let raw_update = lr * m_hat / (sqrt(v_hat) + epsilon);

        // Per-parameter update clip (mirrors splat_step strategy: keeps the
        // optimizer from blowing through tight clamps in a single step).
        // Control points get a hard 0.003/step cap (down from 0.008): the
        // dominant failure mode at higher caps was visible oscillation as
        // a curve toggled which polyline segment was closest to each pixel.
        // 0.003/step still traverses the 2-unit canvas in ~700 steps under
        // sustained gradient. Opacity gets a relatively generous cap so a
        // curve pushed into empty space hits the kill threshold in roughly
        // one ADC period.
        var max_update = 0.01;
        if (lp < 8u) { max_update = 0.003; }
        else if (lp <= 10u) { max_update = 0.001; }
        else if (lp == 11u) { max_update = 0.005; }
        else if (lp == 12u) { max_update = 0.002; }
        else if (lp == 13u) { max_update = 0.001; }
        let update = clamp(raw_update, -max_update, max_update);

        if (lp < 8u) {
            pos_grad_norm2 = pos_grad_norm2 + grad * grad;
        }

        if (lp == 0u) { b.p0_p1.x -= update; }
        else if (lp == 1u) { b.p0_p1.y -= update; }
        else if (lp == 2u) { b.p0_p1.z -= update; }
        else if (lp == 3u) { b.p0_p1.w -= update; }
        else if (lp == 4u) { b.p2_p3.x -= update; }
        else if (lp == 5u) { b.p2_p3.y -= update; }
        else if (lp == 6u) { b.p2_p3.z -= update; }
        else if (lp == 7u) { b.p2_p3.w -= update; }
        else if (lp == 8u)  { b.color.r = clamp(b.color.r - update, 0.0, 1.0); }
        else if (lp == 9u)  { b.color.g = clamp(b.color.g - update, 0.0, 1.0); }
        else if (lp == 10u) { b.color.b = clamp(b.color.b - update, 0.0, 1.0); }
        // Lower clamp floors are intentionally near-zero. They are above 0
        // only to avoid degenerate divisions in the forward pass; they no
        // longer act as a soft "minimum-alive" floor. The kill condition
        // below is what actually controls when a curve goes away. Letting
        // both floors approach 0 means a stray's (width * opacity) product
        // can collapse fast enough to trip the kill in just a few steps,
        // instead of waiting ~90 steps for opacity to crawl from 0.6 down
        // past a relatively high old floor of 0.05.
        else if (lp == 11u) { b.color.a = clamp(b.color.a - update, 0.01, 0.99); }
        // Width/softness upper bounds are sized for the display resolution
        // (not the 128-px optim resolution). 1 norm unit is ~250 px on a
        // typical panel, so 0.02 norm = 5 px on display, which is already
        // visually thicker than a 1-2 px target edge. Lower bound of 0.001
        // (~0.25 px on display) lets productive curves go as thin as the
        // target's 1-2 px edges and lets strays collapse below the kill
        // threshold (0.001 * 0.6 = 0.0006 < 0.0008) immediately upon
        // hitting the width floor, instead of waiting on slow opacity decay.
        else if (lp == 12u) { b.width_soft_pad.x = clamp(b.width_soft_pad.x - update, 0.001, 0.02); }
        else if (lp == 13u) { b.width_soft_pad.y = clamp(b.width_soft_pad.y - update, 0.0002, 0.008); }
    }

    adc.grad_accum[bid] += sqrt(pos_grad_norm2);

    // Mark dead via "effective area" (width * opacity) — analogous to the
    // splat kill on (scale.x * scale.y). Initial product is 0.04 * 0.6 =
    // 0.024 so the 0.0008 threshold means a curve must drop ~30x in
    // effective coverage before being recycled. This is far more forgiving
    // than the previous either-factor-at-clamp-floor check, so transient
    // bad fits no longer trip an instant kill.
    if (b.width_soft_pad.x * b.color.a <= 0.0008) {
        b.color.a = 0.0;
    }

    beziers.items[bid] = b;
}
