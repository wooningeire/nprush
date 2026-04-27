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
    pixel_count: f32,
    pad: vec2f,
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

    // Pass 1: collect dead curves (free list). The kill check in
    // bezier_step.wgsl writes literal 0.0 to alpha; the alive clamp floor
    // is 0.01. We treat anything below 0.005 as dead, which separates the
    // two cases without colliding with valid alive states.
    for (var i = 0u; i < NUM_BEZIERS; i = i + 1u) {
        if (beziers.items[i].color.a < 0.005) {
            dead_indices[dead_count] = i;
            dead_count = dead_count + 1u;
        }
    }

    // ADC_PERIOD must match the JS dispatch cadence in
    // GpuBezierOptimizerManager.ts. It is used here only as the divisor that
    // normalizes accumulated gradient into a per-step average; the actual
    // dispatch cadence is set in JS.
    let ADC_PERIOD = 100.0;
    let TAU_POS = 0.005;
    let SPLIT_LEN_THRESHOLD = 0.25;
    // Always preserve at least this fraction of dead slots. A productive
    // curve is never *perfectly* fit and so always has some residual
    // gradient; without this gate, ADC would keep splitting productive
    // curves even when every slot is alive, the resulting clone+parent
    // pairs would compete for the same edge, the loser would drift off as
    // a stray, and the system would never stabilize. Maintaining 30% dead
    // headroom means a freshly-killed stray's slot is *not* immediately
    // refilled, so the population gradually decays toward "as many curves
    // as the silhouette actually demands."
    let MIN_DEAD_FRACTION = 0.3;
    let MIN_DEAD_SLOTS = u32(f32(NUM_BEZIERS) * MIN_DEAD_FRACTION);

    // Pass 2: clone or split high-gradient curves into dead slots.
    for (var i = 0u; i < NUM_BEZIERS; i = i + 1u) {
        var b = beziers.items[i];
        if (b.color.a < 0.005) { continue; }

        let grad_norm = adc.grad_accum[i] / ADC_PERIOD;
        adc.grad_accum[i] = 0.0;

        if (grad_norm <= TAU_POS) {
            // Plateau check: if the curve has settled into a low-gradient state 
            // but the opacity optimizer is still trying to kill it (positive 
            // momentum in the alpha channel), it's a stray that only adds 
            // to loss. Kill it immediately to free the slot.
            if (adam.m[i * 14u + 11u] > 1e-5) {
                beziers.items[i].color.a = 0.0;
            }
            continue;
        }
        // Both checks are on the live dead_count: stop spawning as soon as
        // the headroom drops to MIN_DEAD_SLOTS so the next ADC tick still
        // has slack for whichever new high-grad curve emerges.
        if (dead_count <= MIN_DEAD_SLOTS) { continue; }

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
            let jx = (fract(sin(seed * 12.9898) * 43758.5453) - 0.5) * 0.003;
            let jy = (fract(sin(seed * 78.233) * 43758.5453) - 0.5) * 0.003;
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

    // No pass 3. Dead slots stay dead. Earlier revisions tried two ways of
    // refilling them — random respawn, then "smart fill" by cloning the
    // highest-coverage live curve into every remaining dead slot — and both
    // produced visible stray strokes in empty space:
    //
    //   - Random respawn: ghosts that decayed slowly in unused regions.
    //   - Smart fill: bright high-alpha clones planted near the silhouette
    //     each ADC tick; the fraction whose jitter pushed them off-edge
    //     drifted into empty space and accumulated faster than the step-
    //     shader kill could clean them up.
    //
    // Letting dead slots stay dead lets the population settle at "as many
    // curves as the target actually demands." Pass 2 still re-grows
    // capacity wherever there's real gradient signal (a high-grad live
    // curve gets cloned/split into a free slot), and that's the only
    // mechanism by which dead slots ever come back.
}
