struct Bezier {
    p0: vec4f,    // x, y, z, width
    p1: vec4f,    // x, y, z, softness
    p2: vec4f,    // x, y, z, _pad
    p3: vec4f,    // x, y, z, _pad
    color: vec4f, // r, g, b, a
}

struct BezierArray {
    items: array<Bezier, {@NUM_BEZIERS}u>,
}

struct GradArray {
    data: array<atomic<i32>, {@NUM_BEZIER_PARAMS}u>,
}

struct AdamState {
    m: array<f32, {@NUM_BEZIER_PARAMS}u>,
    v: array<f32, {@NUM_BEZIER_PARAMS}u>,
    t: f32,
    pixel_count: f32,
    no_kill: f32, // 1.0 = disable loss-based killing in ADC
    pad: f32,
}

struct ADCArray {
    grad_accum: array<f32, {@NUM_BEZIERS}u>,
}

struct StepUniforms {
    vp: mat4x4f,             // offset 0,  size 64
    mode: f32,               // offset 64, size 4
    max_width: f32,          // offset 68, size 4
    prune_alpha_thresh: f32, // offset 72, size 4
    prune_width_thresh: f32, // offset 76, size 4
    bg_penalty: f32,         // offset 80, size 4  (unused in step, keeps layout aligned)
    _pad0: f32,              // offset 84
    _pad1: f32,              // offset 88
    _pad2: f32,              // offset 92
    // total: 96 bytes
}

@group(0) @binding(0) var<storage, read_write> beziers: BezierArray;
@group(0) @binding(1) var<storage, read_write> grads: GradArray;
@group(0) @binding(2) var<storage, read_write> adam: AdamState;
@group(0) @binding(3) var<storage, read_write> adc: ADCArray;
@group(0) @binding(4) var<uniform> uniforms: StepUniforms;

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let bezier_id = global_id.x;
    let current_t = adam.t;
    
    workgroupBarrier();
    if (bezier_id == 0u) {
        adam.t = current_t + 1.0;
    }
    
    if (bezier_id >= {@NUM_BEZIERS}u) {
        return;
    }

    var b = beziers.items[bezier_id];

    // Skip dead beziers entirely — don't apply Adam updates, which could
    // accidentally revive a killed curve via stale gradient momentum.
    if (b.color.a < f32({@BEZIER_KILL_ALPHA_THRESH})) {
        // Still drain any stale gradients so they don't accumulate.
        for (var lp = 0u; lp < 18u; lp++) {
            atomicExchange(&grads.data[bezier_id * 18u + lp], 0);
        }
        return;
    }

    let base_idx = bezier_id * 18u;
    let t = current_t + 1.0;
    var pos_grad_norm2 = 0.0;

    // lr per param: 0-11=positions, 12-14=color, 15=opacity, 16-17=width/softness
    // Gradients are summed over all pixels; divide by optim pixel count (~128*128=16384).
    const lr_table = array<f32, 18>(
        0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
        0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
        0.01,  0.01,  0.01,  0.005, 0.002, 0.002
    );
    // max_update per param
    const mu_table = array<f32, 18>(
        0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
        0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
        0.01,  0.01,  0.01,  0.005, 0.003, 0.003
    );
    // fp_scale: 10000 for positions/width/softness, 100000 for color
    const fps_table = array<f32, 18>(
        10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
        10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
        100000.0, 100000.0, 100000.0, 100000.0, 10000.0, 10000.0
    );

    for (var lp = 0u; lp < 18u; lp++) {
        let param_idx = base_idx + lp;
        let raw_grad = atomicExchange(&grads.data[param_idx], 0);
        
        let fp_scale = fps_table[lp];
        // Normalize by actual optim resolution so lr is resolution-independent.
        // pixel_count is written each frame by the TS dispatch call.
        let pixel_norm = 1.0 / max(adam.pixel_count, 1.0);
        let grad = f32(raw_grad) / fp_scale * pixel_norm;

        let lr = lr_table[lp];

        let beta1 = {@ADAM_BETA1};
        let beta2 = {@ADAM_BETA2};
        let epsilon = {@ADAM_EPS};
        
        var m = adam.m[param_idx];
        var v = adam.v[param_idx];
        
        m = beta1 * m + (1.0 - beta1) * grad;
        v = beta2 * v + (1.0 - beta2) * grad * grad;
        adam.m[param_idx] = m;
        adam.v[param_idx] = v;
        
        let m_hat = m / (1.0 - pow(beta1, t));
        let v_hat = v / (1.0 - pow(beta2, t));
        
        let raw_update = lr * m_hat / (sqrt(v_hat) + epsilon);
        
        let max_update = mu_table[lp];
        let update = clamp(raw_update, -max_update, max_update);

        // Unpack bezier fields into a flat array, apply update, repack
        var params_arr = array<f32, 18>(
            b.p0.x, b.p0.y, b.p0.z,
            b.p1.x, b.p1.y, b.p1.z,
            b.p2.x, b.p2.y, b.p2.z,
            b.p3.x, b.p3.y, b.p3.z,
            b.color.r, b.color.g, b.color.b, b.color.a,
            b.p0.w, b.p1.w
        );
        let lo = array<f32, 18>(
            -1e9, -1e9, -1e9, -1e9, -1e9, -1e9,
            -1e9, -1e9, -1e9, -1e9, -1e9, -1e9,
            0.0, 0.0, 0.0, 0.00, 0.001, 0.001
        );
        let width_hi = select(0.1, uniforms.max_width, uniforms.max_width > 0.0);
        let hi = array<f32, 18>(
            1e9, 1e9, 1e9, 1e9, 1e9, 1e9,
            1e9, 1e9, 1e9, 1e9, 1e9, 1e9,
            1.0, 1.0, 1.0, 0.99, width_hi, 0.03
        );
        params_arr[lp] = clamp(params_arr[lp] - update, lo[lp], hi[lp]);
        b.p0    = vec4f(params_arr[0],  params_arr[1],  params_arr[2],  params_arr[16]);
        b.p1    = vec4f(params_arr[3],  params_arr[4],  params_arr[5],  params_arr[17]);
        b.p2    = vec4f(params_arr[6],  params_arr[7],  params_arr[8],  b.p2.w);
        b.p3    = vec4f(params_arr[9],  params_arr[10], params_arr[11], b.p3.w);
        b.color = vec4f(params_arr[12], params_arr[13], params_arr[14], params_arr[15]);

        if (lp <= 11u) { pos_grad_norm2 += grad * grad; }
    }

    adc.grad_accum[bezier_id] += sqrt(pos_grad_norm2);

    // Prune very thin or transparent beziers — thresholds are configurable
    // per layer so fine color beziers can use a lower kill threshold.
    let alpha_thresh = select(f32({@BEZIER_PRUNE_ALPHA_DEFAULT}), uniforms.prune_alpha_thresh, uniforms.prune_alpha_thresh > 0.0);
    let width_thresh = select(f32({@BEZIER_PRUNE_WIDTH_DEFAULT}), uniforms.prune_width_thresh, uniforms.prune_width_thresh > 0.0);
    b.color.a = select(b.color.a, 0.0, b.color.a < alpha_thresh || b.p0.w <= width_thresh);

    // Kill beziers whose bounding hull is entirely outside the view frustum.
    // Skipped when no_kill is set (fine color layer) — those curves are allowed
    // to drift temporarily and will be pulled back by the loss gradient.
    if (b.color.a > 0.0 && adam.no_kill < 0.5) {
        let c0 = uniforms.vp * vec4f(b.p0.xyz, 1.0);
        let c1 = uniforms.vp * vec4f(b.p1.xyz, 1.0);
        let c2 = uniforms.vp * vec4f(b.p2.xyz, 1.0);
        let c3 = uniforms.vp * vec4f(b.p3.xyz, 1.0);
        // Use a small margin so curves near the edge aren't killed prematurely.
        let margin = f32({@BEZIER_OFFSCREEN_MARGIN});
        let all_left  = c0.x < -margin*c0.w && c1.x < -margin*c1.w && c2.x < -margin*c2.w && c3.x < -margin*c3.w;
        let all_right = c0.x >  margin*c0.w && c1.x >  margin*c1.w && c2.x >  margin*c2.w && c3.x >  margin*c3.w;
        let all_below = c0.y < -margin*c0.w && c1.y < -margin*c1.w && c2.y < -margin*c2.w && c3.y < -margin*c3.w;
        let all_above = c0.y >  margin*c0.w && c1.y >  margin*c1.w && c2.y >  margin*c2.w && c3.y >  margin*c3.w;
        let all_behind = c0.w < 0.0 && c1.w < 0.0 && c2.w < 0.0 && c3.w < 0.0;
        let offscreen = all_left || all_right || all_below || all_above || all_behind;
        b.color.a = select(b.color.a, 0.0, offscreen);
    }

    beziers.items[bezier_id] = b;
}
