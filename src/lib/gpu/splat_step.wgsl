struct Splat {
    pos_sx: vec4f,
    color: vec4f,
    quat: vec4f,
    sy_shape: vec4f,
}

struct SplatArray {
    splats: array<Splat, NUM_SPLATS>,
}

struct GradArray {
    data: array<atomic<i32>, NUM_PARAMS>,
}

struct AdamState {
    m: array<f32, NUM_PARAMS>,
    v: array<f32, NUM_PARAMS>,
    t: f32,
    pad: vec3f,
}

struct ADCArray {
    grad_accum: array<f32, NUM_SPLATS>,
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
    if (splat_id >= NUM_SPLATS) { return; }

    var s = splats.splats[splat_id];
    let base_idx = splat_id * 15u;
    let t = current_t + 1.0;
    var pos_grad_norm2 = 0.0;

    // Param layout: 0=px, 1=py, 2=pz, 3=sx, 4=r, 5=g, 6=b, 7=opacity,
    // 8=qw, 9=qx, 10=qy, 11=qz, 12=sy, 13=shape_a, 14=shape_b
    // lr per param
    let lr_table    = array<f32, 15>(0.0005, 0.0005, 0.0005, 0.01, 0.02, 0.02, 0.02, 0.01, 0.005, 0.005, 0.005, 0.005, 0.01, 0.01, 0.01);
    // max_update per param
    let mu_table    = array<f32, 15>(0.005, 0.005, 0.005, 0.005, 0.001, 0.001, 0.001, 0.0005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.05, 0.05);
    // fp_scale: 10000 for pos/scale/quat/shape, 100000 for color
    let fps_table   = array<f32, 15>(10000.0, 10000.0, 10000.0, 10000.0, 100000.0, 100000.0, 100000.0, 100000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0);

    for (var lp = 0u; lp < 15u; lp++) {
        let param_idx = base_idx + lp;
        let raw_grad = atomicExchange(&grads.data[param_idx], 0);
        let fp_scale = fps_table[lp];
        let grad = f32(raw_grad) / fp_scale / 16384.0;

        let lr = lr_table[lp];

        let beta1 = 0.9;
        let beta2 = 0.999;
        let epsilon = 1e-8;
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

        // Unpack splat fields into a flat array, apply update, repack
        var params_arr = array<f32, 15>(
            s.pos_sx.x, s.pos_sx.y, s.pos_sx.z, s.pos_sx.w,
            s.color.r,  s.color.g,  s.color.b,  s.color.a,
            s.quat.x,   s.quat.y,   s.quat.z,   s.quat.w,
            s.sy_shape.x, s.sy_shape.y, s.sy_shape.z
        );
        let lo = array<f32, 15>(-1e9, -1e9, -1e9, 0.001, 0.05, 0.05, 0.05, 0.01, -1e9, -1e9, -1e9, -1e9, 0.001, 0.1, 0.01);
        let hi = array<f32, 15>( 1e9,  1e9,  1e9, 2.0,   1.0,  1.0,  1.0,  0.99,  1e9,  1e9,  1e9,  1e9,  2.0, 10.0, 5.0);
        params_arr[lp] = clamp(params_arr[lp] - update, lo[lp], hi[lp]);
        s.pos_sx   = vec4f(params_arr[0], params_arr[1], params_arr[2], params_arr[3]);
        s.color    = vec4f(params_arr[4], params_arr[5], params_arr[6], params_arr[7]);
        s.quat     = vec4f(params_arr[8], params_arr[9], params_arr[10], params_arr[11]);
        s.sy_shape = vec4f(params_arr[12], params_arr[13], params_arr[14], s.sy_shape.w);

        if (lp <= 2u) { pos_grad_norm2 += grad * grad; }
    }

    // Re-normalize quaternion
    let q_len = max(length(s.quat), 1e-8);
    s.quat = s.quat / q_len;

    adc.grad_accum[splat_id] += sqrt(pos_grad_norm2);

    let area = s.pos_sx.w * s.sy_shape.x;
    s.color.a = select(s.color.a, 0.0, s.color.a < 0.05 || area < 0.0001);

    splats.splats[splat_id] = s;
}
