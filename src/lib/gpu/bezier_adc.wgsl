struct Bezier {
    p0: vec4f,    // x, y, z, width
    p1: vec4f,    // x, y, z, softness
    p2: vec4f,    // x, y, z, _pad
    p3: vec4f,    // x, y, z, _pad
    color: vec4f, // r, g, b, a
}

struct BezierArray {
    items: array<Bezier, NUM_BEZIERS>,
}

struct AdamState {
    m: array<f32, NUM_BEZIER_PARAMS>,
    v: array<f32, NUM_BEZIER_PARAMS>,
    t: f32,
    pixel_count: f32,
    no_kill: f32, // 1.0 = disable loss-based killing in ADC
    pad: f32,
}

struct ADCArray {
    grad_accum: array<f32, NUM_BEZIERS>,
    loss_accum: array<f32, NUM_BEZIERS>,
}

struct BezierUniforms {
    vp: mat4x4f,
    mode: f32,
    max_width: f32,
    prune_alpha_thresh: f32,
    prune_width_thresh: f32,
    bg_penalty: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    vp_inv: mat4x4f,
}

@group(0) @binding(0) var<storage, read_write> beziers: BezierArray;
@group(0) @binding(1) var<storage, read_write> adam: AdamState;
@group(0) @binding(2) var<storage, read_write> adc: ADCArray;
@group(0) @binding(3) var<storage, read_write> pixel_loss: array<atomic<i32>, PIXEL_LOSS_SIZE>;
@group(0) @binding(4) var<uniform> uniforms: BezierUniforms;

var<workgroup> dead_indices: array<u32, NUM_BEZIERS>;

// Reconstruct a world-space point from a pixel index, using the same reciprocal
// depth encoding as mesh.wgsl. depth_enc = 1 - 0.1/w  =>  w = 0.1/(1-depth_enc).
// We use depth_enc = 0.5 (mid-range) as a neutral spawn depth when no depth info
// is available — the optimizer will pull the curve to the correct depth quickly.
fn pixel_to_world(px_idx: u32, spawn_depth: f32) -> vec3f {
    let px_x = px_idx % OPTIM_WIDTH;
    let px_y = px_idx / OPTIM_WIDTH;
    let uv = (vec2f(f32(px_x), f32(px_y)) + 0.5) / vec2f(f32(OPTIM_WIDTH), f32(OPTIM_HEIGHT));
    // NDC: y flipped (texture y=0 is top, NDC y=1 is top)
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let aspect = f32(OPTIM_WIDTH) / f32(OPTIM_HEIGHT);
    // Recover w from reciprocal depth encoding
    let w = 0.1 / max(1.0 - spawn_depth, 1e-5);
    // Approximate z_clip ≈ w (valid for zFar=100 >> zNear=0.01)
    let clip = vec4f(ndc.x * w, ndc.y * w, w, w);
    let world = uniforms.vp_inv * clip;
    return world.xyz / world.w;
}

@compute @workgroup_size(1, 1, 1)
fn main() {
    var dead_count = 0u;

    for (var i = 0u; i < NUM_BEZIERS; i = i + 1u) {
        if (beziers.items[i].color.a < 0.005) {
            dead_indices[dead_count] = i;
            dead_count = dead_count + 1u;
        }
    }

    let ADC_PERIOD = 50.0;
    let TAU_POS = 0.0002;       // must be moving to clone
    let TAU_LOSS = 0.001;       // kill if stuck AND contributing to loss
    let SPLIT_LEN_THRESHOLD = 0.25;

    adam.t = 0.0;

    // --- Pass 1: clone/split high-gradient live curves (existing behaviour) ---
    for (var i = 0u; i < NUM_BEZIERS; i = i + 1u) {
        var b = beziers.items[i];
        if (b.color.a < 0.005) { continue; }

        let grad_norm = adc.grad_accum[i] / ADC_PERIOD;
        adc.grad_accum[i] = 0.0;

        let loss_norm = adc.loss_accum[i] / ADC_PERIOD;
        adc.loss_accum[i] = 0.0;

        if (adam.no_kill < 0.5 && grad_norm <= TAU_POS && loss_norm > TAU_LOSS) {
            beziers.items[i].color.a = 0.0;
            // Re-add to dead list so pass 2 can use this slot
            dead_indices[dead_count] = i;
            dead_count = dead_count + 1u;
            continue;
        }

        if (grad_norm <= TAU_POS) { continue; }
        if (dead_count == 0u) { continue; }

        dead_count = dead_count - 1u;
        let new_idx = dead_indices[dead_count];

        let p0 = b.p0.xyz;
        let p1 = b.p1.xyz;
        let p2 = b.p2.xyz;
        let p3 = b.p3.xyz;

        let mid = (p0 + 3.0 * p1 + 3.0 * p2 + p3) * 0.125;
        let len_approx = length(mid - p0) + length(p3 - mid);

        var new_b = b;

        if (len_approx > SPLIT_LEN_THRESHOLD) {
            let q0 = (p0 + p1) * 0.5;
            let q1 = (p1 + p2) * 0.5;
            let q2 = (p2 + p3) * 0.5;
            let r0 = (q0 + q1) * 0.5;
            let r1 = (q1 + q2) * 0.5;
            let s  = (r0 + r1) * 0.5;

            b.p0 = vec4f(p0, b.p0.w);
            b.p1 = vec4f(q0, b.p1.w);
            b.p2 = vec4f(r0, 0.0);
            b.p3 = vec4f(s,  0.0);

            new_b.p0 = vec4f(s,  new_b.p0.w);
            new_b.p1 = vec4f(r1, new_b.p1.w);
            new_b.p2 = vec4f(q2, 0.0);
            new_b.p3 = vec4f(p3, 0.0);
        } else {
            let seed = f32(i) * 3.14159 + adam.t;
            let jx = (fract(sin(seed * 12.9898) * 43758.5453) - 0.5) * 0.001;
            let jy = (fract(sin(seed * 78.233)  * 43758.5453) - 0.5) * 0.001;
            let jz = (fract(sin(seed * 43.123)  * 43758.5453) - 0.5) * 0.001;
            let j = vec3f(jx, jy, jz);
            new_b.p0 = vec4f(b.p0.xyz + j, b.p0.w);
            new_b.p1 = vec4f(b.p1.xyz + j, b.p1.w);
            new_b.p2 = vec4f(b.p2.xyz + j, 0.0);
            new_b.p3 = vec4f(b.p3.xyz + j, 0.0);
        }

        beziers.items[i] = b;
        beziers.items[new_idx] = new_b;

        for (var p = 0u; p < 18u; p = p + 1u) {
            adam.m[i * 18u + p] = 0.0;
            adam.v[i * 18u + p] = 0.0;
            adam.m[new_idx * 18u + p] = 0.0;
            adam.v[new_idx * 18u + p] = 0.0;
        }
    }

    // --- Pass 2: seed new beziers at the highest-loss uncovered pixels ---
    // Single linear scan: collect the top-revive_count pixels by loss.
    // We keep a small sorted-insert buffer; for the typical revive_count
    // (tens to low hundreds) this is fast enough in a single-threaded workgroup.
    // Fill ALL remaining dead slots so population stays at capacity.
    let revive_count = dead_count;

    if (revive_count > 0u) {
        // One-pass scan: find the single best pixel, spawn there, zero it, repeat.
        // Cap at revive_count iterations; each iteration is O(PIXEL_LOSS_SIZE).
        // To keep GPU time bounded, limit to at most 64 spawns per ADC cycle.
        let max_spawns = min(revive_count, 64u);

        for (var pass = 0u; pass < max_spawns; pass = pass + 1u) {
            if (dead_count == 0u) { break; }

            // Find highest-loss pixel
            var best_px = 0u;
            var best_val = 0;
            for (var px = 0u; px < PIXEL_LOSS_SIZE; px = px + 1u) {
                let v = atomicLoad(&pixel_loss[px]);
                if (v > best_val) {
                    best_val = v;
                    best_px = px;
                }
            }
            // No meaningful loss anywhere — stop early
            if (best_val <= 0) { break; }

            // Claim this pixel
            atomicStore(&pixel_loss[best_px], 0);

            dead_count = dead_count - 1u;
            let slot = dead_indices[dead_count];

            let spawn_depth = 0.5;
            let center = pixel_to_world(best_px, spawn_depth);

            let seed = f32(best_px) * 1.61803 + f32(pass) * 2.71828;
            let angle = fract(sin(seed * 127.1) * 43758.5453) * 6.28318;
            let tx = cos(angle) * 0.025;
            let tz = sin(angle) * 0.025;
            let tangent = vec3f(tx, 0.0, tz);

            var nb: Bezier;
            nb.p0 = vec4f(center - tangent,        0.015);
            nb.p1 = vec4f(center - tangent * 0.33, 0.005);
            nb.p2 = vec4f(center + tangent * 0.33, 0.0);
            nb.p3 = vec4f(center + tangent,        0.0);
            nb.color = vec4f(0.5, 0.5, 0.5, 0.5);

            beziers.items[slot] = nb;
            for (var p = 0u; p < 18u; p = p + 1u) {
                adam.m[slot * 18u + p] = 0.0;
                adam.v[slot * 18u + p] = 0.0;
            }
        }
    }

    // Reset remaining pixel_loss entries for the next ADC period
    for (var px = 0u; px < PIXEL_LOSS_SIZE; px = px + 1u) {
        atomicStore(&pixel_loss[px], 0);
    }
}
