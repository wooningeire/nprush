struct Splat {
    pos_sx: vec4f,
    color: vec4f,
    quat: vec4f,
    sy_shape: vec4f,
    sh1_r: vec4f,
    sh1_g: vec4f,
    sh1_b: vec4f,
    sh1_a: vec4f,
}

struct SplatArray {
    splats: array<Splat, {@NUM_SPLATS}u>,
}

struct GradArray {
    data: array<atomic<i32>, {@NUM_PARAMS}u>,
}

struct AdamState {
    m: array<f32, {@NUM_PARAMS}u>,
    v: array<f32, {@NUM_PARAMS}u>,
    t: f32,
    pad: vec3f,
}

struct ADCArray {
    grad_accum: array<f32, {@NUM_SPLATS}u>,
}

@group(0) @binding(0) var<storage, read_write> splats: SplatArray;
@group(0) @binding(1) var<storage, read_write> grads: GradArray;
@group(0) @binding(2) var<storage, read_write> adam: AdamState;
@group(0) @binding(3) var<storage, read_write> adc: ADCArray;

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let splat_id = global_id.x;
    let current_t = adam.t;
    workgroupBarrier();
    if (splat_id == 0u) { adam.t = current_t + 1.0; }
    if (splat_id >= {@NUM_SPLATS}u) { return; }

    var s = splats.splats[splat_id];
    let base_idx = splat_id * {@SPLAT_PARAMS_PER_SPLAT}u;
    let t = current_t + 1.0;
    var pos_grad_norm2 = 0.0;

    // Param indices 16–31: RGB + opacity degree-1 SH (vec4-packed; backward writes xyz only).
    let lr_table = array<f32, {@SPLAT_PARAMS_PER_SPLAT}>(
        0.0005, 0.0005, 0.0005, 0.01,
        0.02, 0.02, 0.02, 0.01,
        0.005, 0.005, 0.005, 0.005,
        0.01, 0.01, 0.01, 0.01,
        0.02, 0.02, 0.02, 0.0,
        0.02, 0.02, 0.02, 0.0,
        0.02, 0.02, 0.02, 0.0,
        0.02, 0.02, 0.02, 0.0
    );
    let mu_table = array<f32, {@SPLAT_PARAMS_PER_SPLAT}>(
        0.005, 0.005, 0.005, 0.005,
        0.001, 0.001, 0.001, 0.0005,
        0.005, 0.005, 0.005, 0.005,
        0.005, 0.05, 0.05, 0.005,
        0.005, 0.005, 0.005, 0.0,
        0.005, 0.005, 0.005, 0.0,
        0.005, 0.005, 0.005, 0.0,
        0.005, 0.005, 0.005, 0.0
    );
    let fps_table = array<f32, {@SPLAT_PARAMS_PER_SPLAT}>(
        10000.0, 10000.0, 10000.0, 10000.0,
        100000.0, 100000.0, 100000.0, 100000.0,
        10000.0, 10000.0, 10000.0, 10000.0,
        10000.0, 10000.0, 10000.0, 10000.0,
        100000.0, 100000.0, 100000.0, 100000.0,
        100000.0, 100000.0, 100000.0, 100000.0,
        100000.0, 100000.0, 100000.0, 100000.0,
        100000.0, 100000.0, 100000.0, 100000.0
    );

    for (var lp = 0u; lp < {@SPLAT_PARAMS_PER_SPLAT}u; lp++) {
        let param_idx = base_idx + lp;
        let raw_grad = atomicExchange(&grads.data[param_idx], 0);
        let fp_scale = fps_table[lp];
        let grad = f32(raw_grad) / fp_scale / 16384.0;

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

        var params_arr = array<f32, {@SPLAT_PARAMS_PER_SPLAT}>(
            s.pos_sx.x, s.pos_sx.y, s.pos_sx.z, s.pos_sx.w,
            s.color.r,  s.color.g,  s.color.b,  s.color.a,
            s.quat.x,   s.quat.y,   s.quat.z,   s.quat.w,
            s.sy_shape.x, s.sy_shape.y, s.sy_shape.z, s.sy_shape.w,
            s.sh1_r.x, s.sh1_r.y, s.sh1_r.z, s.sh1_r.w,
            s.sh1_g.x, s.sh1_g.y, s.sh1_g.z, s.sh1_g.w,
            s.sh1_b.x, s.sh1_b.y, s.sh1_b.z, s.sh1_b.w,
            s.sh1_a.x, s.sh1_a.y, s.sh1_a.z, s.sh1_a.w
        );
        let lo = array<f32, {@SPLAT_PARAMS_PER_SPLAT}>(
            -1e9, -1e9, -1e9, 0.001,
            -5.0, -5.0, -5.0, 0.01,
            -1e9, -1e9, -1e9, -1e9,
            0.001, 0.1, 0.01, 0.001,
            -2.5, -2.5, -2.5, -1e9,
            -2.5, -2.5, -2.5, -1e9,
            -2.5, -2.5, -2.5, -1e9,
            -2.5, -2.5, -2.5, -1e9
        );
        let hi = array<f32, {@SPLAT_PARAMS_PER_SPLAT}>(
            1e9,  1e9,  1e9, 2.0,
            5.0,  5.0,  5.0, 0.99,
            1e9,  1e9,  1e9,  1e9,
            2.0, 10.0, 5.0,  2.0,
            2.5,  2.5,  2.5,  1e9,
            2.5,  2.5,  2.5,  1e9,
            2.5,  2.5,  2.5,  1e9,
            2.5,  2.5,  2.5,  1e9
        );
        params_arr[lp] = clamp(params_arr[lp] - update, lo[lp], hi[lp]);
        s.pos_sx = vec4f(params_arr[0], params_arr[1], params_arr[2], params_arr[3]);
        s.color = vec4f(params_arr[4], params_arr[5], params_arr[6], params_arr[7]);
        s.quat = vec4f(params_arr[8], params_arr[9], params_arr[10], params_arr[11]);
        s.sy_shape = vec4f(params_arr[12], params_arr[13], params_arr[14], params_arr[15]);
        s.sh1_r = vec4f(params_arr[16], params_arr[17], params_arr[18], params_arr[19]);
        s.sh1_g = vec4f(params_arr[20], params_arr[21], params_arr[22], params_arr[23]);
        s.sh1_b = vec4f(params_arr[24], params_arr[25], params_arr[26], params_arr[27]);
        s.sh1_a = vec4f(params_arr[28], params_arr[29], params_arr[30], params_arr[31]);

        if (lp <= 2u) { pos_grad_norm2 += grad * grad; }
    }

    let q_len = max(length(s.quat), 1e-8);
    s.quat = s.quat / q_len;

    adc.grad_accum[splat_id] += sqrt(pos_grad_norm2);

    let volume = s.pos_sx.w * s.sy_shape.x * s.sy_shape.w;
    s.color.a = select(s.color.a, 0.0, s.color.a < f32({@SPLAT_OPACITY_KILL_THRESH}) || volume < f32({@SPLAT_VOLUME_KILL_THRESH}));

    splats.splats[splat_id] = s;
}
