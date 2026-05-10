struct Bezier {
    p0: vec4f,
    p1: vec4f,
    p2: vec4f,
    p3: vec4f,
    color: vec4f,
    sh1_r: vec4f,
    sh1_g: vec4f,
    sh1_b: vec4f,
    sh1_a: vec4f,
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
    no_kill: f32,
    pad: f32,
}

struct ADCArray {
    grad_accum: array<f32, {@NUM_BEZIERS}u>,
}

struct StepUniforms {
    vp: mat4x4f,
    mode: f32,
    max_width: f32,
    prune_alpha_thresh: f32,
    prune_width_thresh: f32,
    bg_penalty: f32,
    _pad0: f32,
    _pad1: f32,
    adc_period_steps: f32,
    optim_width: f32,
    optim_height: f32,
    _pad_res: vec2f,
    vp_inv: mat4x4f,
    cam_world: vec4f,
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

    if (bezier_id == 0u) {
        adam.t = current_t + 1.0;
    }

    if (bezier_id >= {@NUM_BEZIERS}u) {
        return;
    }

    var b = beziers.items[bezier_id];

    if (b.color.a < f32({@BEZIER_KILL_ALPHA_THRESH})) {
        for (var lp = 0u; lp < {@BEZIER_PARAMS_PER}u; lp++) {
            atomicExchange(&grads.data[bezier_id * {@BEZIER_PARAMS_PER}u + lp], 0);
        }
        return;
    }

    let base_idx = bezier_id * {@BEZIER_PARAMS_PER}u;
    let t_opt = current_t + 1.0;
    let beta1 = {@ADAM_BETA1};
    let beta2 = {@ADAM_BETA2};
    let epsilon = {@ADAM_EPS};
    let denom_m = 1.0 - pow(beta1, t_opt);
    let denom_v = 1.0 - pow(beta2, t_opt);
    let pixel_norm = 1.0 / max(adam.pixel_count, 1.0);
    var pos_grad_norm2 = 0.0;

    var params_arr = array<f32, {@BEZIER_PARAMS_PER}>(
        b.p0.x, b.p0.y, b.p0.z,
        b.p1.x, b.p1.y, b.p1.z,
        b.p2.x, b.p2.y, b.p2.z,
        b.p3.x, b.p3.y, b.p3.z,
        b.color.r, b.color.g, b.color.b, b.color.a,
        b.p0.w, b.p1.w,
        b.sh1_r.x, b.sh1_r.y, b.sh1_r.z,
        b.sh1_g.x, b.sh1_g.y, b.sh1_g.z,
        b.sh1_b.x, b.sh1_b.y, b.sh1_b.z,
        b.sh1_a.x, b.sh1_a.y, b.sh1_a.z
    );
    let lo = array<f32, {@BEZIER_PARAMS_PER}>(
        -1e9, -1e9, -1e9, -1e9, -1e9, -1e9,
        -1e9, -1e9, -1e9, -1e9, -1e9, -1e9,
        0.0, 0.0, 0.0, 0.00,
        0.001, 0.001,
        -2.5, -2.5, -2.5,
        -2.5, -2.5, -2.5,
        -2.5, -2.5, -2.5,
        -2.5, -2.5, -2.5
    );
    let width_hi = uniforms.max_width;
    let hi = array<f32, {@BEZIER_PARAMS_PER}>(
        1e9, 1e9, 1e9, 1e9, 1e9, 1e9,
        1e9, 1e9, 1e9, 1e9, 1e9, 1e9,
        1.0, 1.0, 1.0, 0.99,
        width_hi, 0.03,
        2.5, 2.5, 2.5,
        2.5, 2.5, 2.5,
        2.5, 2.5, 2.5,
        2.5, 2.5, 2.5
    );

    let lr_table = array<f32, {@BEZIER_PARAMS_PER}>(
        0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
        0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
        0.01,  0.01,  0.01,  0.005,
        0.002, 0.002,
        0.02, 0.02, 0.02,  0.02, 0.02, 0.02,
        0.02, 0.02, 0.02,
        0.02, 0.02, 0.02
    );
    let mu_table = array<f32, {@BEZIER_PARAMS_PER}>(
        0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
        0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
        0.01,  0.01,  0.01,  0.005,
        0.003, 0.003,
        0.005, 0.005, 0.005,  0.005, 0.005, 0.005,
        0.005, 0.005, 0.005,
        0.005, 0.005, 0.005
    );
    let fps_table = array<f32, {@BEZIER_PARAMS_PER}>(
        10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
        10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0,
        100000.0, 100000.0, 100000.0, 100000.0,
        10000.0, 10000.0,
        100000.0, 100000.0, 100000.0,
        100000.0, 100000.0, 100000.0,
        100000.0, 100000.0, 100000.0,
        100000.0, 100000.0, 100000.0
    );

    for (var lp = 0u; lp < {@BEZIER_PARAMS_PER}u; lp++) {
        let param_idx = base_idx + lp;
        let raw_grad = atomicExchange(&grads.data[param_idx], 0);

        let fp_scale = fps_table[lp];
        let grad = f32(raw_grad) / fp_scale * pixel_norm;

        let lr = lr_table[lp];

        var m = adam.m[param_idx];
        var v = adam.v[param_idx];

        m = beta1 * m + (1.0 - beta1) * grad;
        v = beta2 * v + (1.0 - beta2) * grad * grad;
        adam.m[param_idx] = m;
        adam.v[param_idx] = v;

        let m_hat = m / denom_m;
        let v_hat = v / denom_v;

        let raw_update = lr * m_hat / (sqrt(v_hat) + epsilon);

        let max_update = mu_table[lp];
        let update = clamp(raw_update, -max_update, max_update);

        params_arr[lp] = clamp(params_arr[lp] - update, lo[lp], hi[lp]);

        if (lp <= 11u) { pos_grad_norm2 += grad * grad; }
    }

    b.p0    = vec4f(params_arr[0],  params_arr[1],  params_arr[2],  params_arr[16]);
    b.p1    = vec4f(params_arr[3],  params_arr[4],  params_arr[5],  params_arr[17]);
    b.p2    = vec4f(params_arr[6],  params_arr[7],  params_arr[8],  b.p2.w);
    b.p3    = vec4f(params_arr[9],  params_arr[10], params_arr[11], b.p3.w);
    b.color = vec4f(params_arr[12], params_arr[13], params_arr[14], params_arr[15]);
    b.sh1_r = vec4f(params_arr[18], params_arr[19], params_arr[20], b.sh1_r.w);
    b.sh1_g = vec4f(params_arr[21], params_arr[22], params_arr[23], b.sh1_g.w);
    b.sh1_b = vec4f(params_arr[24], params_arr[25], params_arr[26], b.sh1_b.w);
    b.sh1_a = vec4f(params_arr[27], params_arr[28], params_arr[29], b.sh1_a.w);

    adc.grad_accum[bezier_id] += sqrt(pos_grad_norm2);

    let alpha_thresh = select(f32({@BEZIER_PRUNE_ALPHA_DEFAULT}), uniforms.prune_alpha_thresh, uniforms.prune_alpha_thresh > 0.0);
    let width_thresh = select(f32({@BEZIER_PRUNE_WIDTH_DEFAULT}), uniforms.prune_width_thresh, uniforms.prune_width_thresh > 0.0);
    b.color.a = select(b.color.a, 0.0, b.color.a < alpha_thresh || b.p0.w <= width_thresh);

    if (b.color.a > 0.0 && adam.no_kill < 0.5) {
        let c0 = uniforms.vp * vec4f(b.p0.xyz, 1.0);
        let c1 = uniforms.vp * vec4f(b.p1.xyz, 1.0);
        let c2 = uniforms.vp * vec4f(b.p2.xyz, 1.0);
        let c3 = uniforms.vp * vec4f(b.p3.xyz, 1.0);
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
