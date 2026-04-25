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

    // Accumulate the L2 norm of the control-point gradient over an ADC
    // period. ADC uses this to decide which curves to clone/split.
    var pos_grad_norm2 = 0.0;

    for (var lp = 0u; lp < 14u; lp = lp + 1u) {
        let pidx = base_idx + lp;
        let raw_grad = atomicExchange(&grads.data[pidx], 0);
        // Match the FP scale used in bezier_backward.wgsl.
        var fp_scale = 100000.0;
        if (lp < 8u || lp == 12u || lp == 13u) {
            fp_scale = 10000.0;
        }
        let grad = f32(raw_grad) / fp_scale / 16384.0;

        let beta1 = 0.9;
        let beta2 = 0.999;
        let epsilon = 1e-8;

        // Per-parameter learning rates.
        var lr = 0.02;
        if (lp < 8u) { lr = 0.05; }                  // control points
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
        var max_update = 0.01;
        if (lp < 8u) { max_update = 0.02; }
        else if (lp <= 10u) { max_update = 0.001; }
        else if (lp == 11u) { max_update = 0.0005; }
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
        // Floor at 0.05 keeps a single bad step from tripping the kill below.
        // The kill criterion is on the (width * opacity) product, not on
        // either factor alone, so a curve that's only thin OR only faint
        // can still recover under future gradient updates.
        else if (lp == 11u) { b.color.a = clamp(b.color.a - update, 0.05, 0.99); }
        else if (lp == 12u) { b.width_soft_pad.x = clamp(b.width_soft_pad.x - update, 0.005, 0.5); }
        else if (lp == 13u) { b.width_soft_pad.y = clamp(b.width_soft_pad.y - update, 0.001, 0.1); }
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
