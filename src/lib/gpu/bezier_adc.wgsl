// Adaptive Density Control for the cubic-bezier edge layer.
//
// Mirrors splat_adc.wgsl: every ADC_PERIOD steps, curves whose accumulated
// positional gradient norm exceeds TAU_POS are duplicated into a "dead" slot
// (a curve previously zeroed out by the step shader's kill condition). Long
// curves are split at t=0.5 via de Casteljau (both halves share the original
// geometry exactly); short curves are cloned with a small jitter so the two
// copies can diverge under future gradient updates.
//
// Adam moments for both the source and destination curves are reset so
// neither inherits stale momentum from before the topology change.

struct Bezier {
    p0_p1: vec4f,
    p2_p3: vec4f,
    color: vec4f,
    width_soft_pad: vec4f,
}

struct BezierArray {
    items: array<Bezier, NUM_BEZIERS>,
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
@group(0) @binding(1) var<storage, read_write> adam: AdamState;
@group(0) @binding(2) var<storage, read_write> adc: ADCArray;

var<workgroup> dead_indices: array<u32, NUM_BEZIERS>;

@compute @workgroup_size(1, 1, 1)
fn main() {
    var dead_count = 0u;

    // Pass 1: collect dead curves (free list).
    for (var i = 0u; i < NUM_BEZIERS; i = i + 1u) {
        if (beziers.items[i].color.a < 0.05) {
            dead_indices[dead_count] = i;
            dead_count = dead_count + 1u;
        }
    }

    let ADC_PERIOD = 100.0;
    let TAU_POS = 0.001;
    let SPLIT_LEN_THRESHOLD = 0.25;

    // Pass 2: clone or split high-gradient curves into dead slots.
    for (var i = 0u; i < NUM_BEZIERS; i = i + 1u) {
        var b = beziers.items[i];
        if (b.color.a < 0.05) { continue; }

        let grad_norm = adc.grad_accum[i] / ADC_PERIOD;
        adc.grad_accum[i] = 0.0;

        if (grad_norm <= TAU_POS) { continue; }
        if (dead_count == 0u) { continue; }

        dead_count = dead_count - 1u;
        let new_idx = dead_indices[dead_count];

        let p0 = b.p0_p1.xy;
        let p1 = b.p0_p1.zw;
        let p2 = b.p2_p3.xy;
        let p3 = b.p2_p3.zw;

        // Cheap length proxy via midpoint B(0.5) = (p0 + 3p1 + 3p2 + p3) / 8.
        let mid = (p0 + 3.0 * p1 + 3.0 * p2 + p3) * 0.125;
        let len_approx = length(mid - p0) + length(p3 - mid);

        var new_b = b;

        if (len_approx > SPLIT_LEN_THRESHOLD) {
            // Split at t=0.5 via de Casteljau. Both halves cover exactly the
            // original curve geometry, just with their own control polygons.
            let q0 = (p0 + p1) * 0.5;
            let q1 = (p1 + p2) * 0.5;
            let q2 = (p2 + p3) * 0.5;
            let r0 = (q0 + q1) * 0.5;
            let r1 = (q1 + q2) * 0.5;
            let s = (r0 + r1) * 0.5;
            b.p0_p1 = vec4f(p0, q0);
            b.p2_p3 = vec4f(r0, s);
            new_b.p0_p1 = vec4f(s, r1);
            new_b.p2_p3 = vec4f(q2, p3);
        } else {
            // Clone with a small jitter on every control point so the copies
            // don't perfectly overlap and can diverge under future gradients.
            let seed = f32(i) * 3.14159 + adam.t;
            let jx = (fract(sin(seed * 12.9898) * 43758.5453) - 0.5) * 0.01;
            let jy = (fract(sin(seed * 78.233) * 43758.5453) - 0.5) * 0.01;
            new_b.p0_p1 = b.p0_p1 + vec4f(jx, jy, jx, jy);
            new_b.p2_p3 = b.p2_p3 + vec4f(jx, jy, jx, jy);
        }

        beziers.items[i] = b;
        beziers.items[new_idx] = new_b;

        // Reset Adam state for both copies so neither carries stale momentum.
        for (var p = 0u; p < 14u; p = p + 1u) {
            adam.m[i * 14u + p] = 0.0;
            adam.v[i * 14u + p] = 0.0;
            adam.m[new_idx * 14u + p] = 0.0;
            adam.v[new_idx * 14u + p] = 0.0;
        }
    }

    // Pass 3: smart fill. Any dead slots clone/split didn't consume get
    // populated by cloning the live curve with the highest effective
    // coverage (width * opacity), with jitter, into all remaining dead
    // slots.
    //
    // Why not random respawn? A previous version spawned new curves at
    // random positions across the field. Most landed in empty space where
    // the curve's alpha is non-zero only inside its narrow soft band, so
    // it picked up only a trickle of "wrong-place" gradient and took ~1000
    // steps to decay through the kill threshold. During that whole time
    // it was visible as a stray ghost blob, and each ADC tick spawned more
    // strays into empty space, so they accumulated.
    //
    // Cloning an established live curve instead seeds the new slot near
    // where the silhouette is already being learned, so the clone either
    // contributes (joins or extends an arc) or quickly dies and is re-
    // cloned next period — no long-lived strays in empty space.
    if (dead_count > 0u) {
        var best_score = -1.0;
        var best_idx = 0u;
        var found_live = false;
        for (var i = 0u; i < NUM_BEZIERS; i = i + 1u) {
            let bi = beziers.items[i];
            if (bi.color.a < 0.05) { continue; }
            let score = bi.width_soft_pad.x * bi.color.a;
            if (score > best_score) {
                best_score = score;
                best_idx = i;
                found_live = true;
            }
        }

        if (found_live) {
            let src = beziers.items[best_idx];
            for (var k = 0u; k < dead_count; k = k + 1u) {
                let idx = dead_indices[k];
                var nb = src;

                // Larger jitter than the in-pass-2 clone case so multiple
                // copies into multiple dead slots actually spread out.
                let seed = f32(idx + 1u) * 17.31 + adam.t * 0.97 + f32(k) * 4.13;
                let jx = (fract(sin(seed * 12.9898) * 43758.5453) - 0.5) * 0.06;
                let jy = (fract(sin(seed * 78.233) * 43758.5453) - 0.5) * 0.06;
                nb.p0_p1 = nb.p0_p1 + vec4f(jx, jy, jx, jy);
                nb.p2_p3 = nb.p2_p3 + vec4f(jx, jy, jx, jy);
                beziers.items[idx] = nb;

                adc.grad_accum[idx] = 0.0;
                for (var p = 0u; p < 14u; p = p + 1u) {
                    adam.m[idx * 14u + p] = 0.0;
                    adam.v[idx * 14u + p] = 0.0;
                }
            }
        }
        // If no live curves remain at all, leave dead slots dead. This is
        // an extreme failure mode (the optimizer killed everything) and
        // would require external re-seeding to recover; in practice the
        // conservative kill criterion in bezier_step.wgsl prevents it.
    }
}
