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
    for (var lp = 0u; lp < 15u; lp++) {
        let param_idx = base_idx + lp;
        let raw_grad = atomicExchange(&grads.data[param_idx], 0);
        var fp_scale = 100000.0;
        if (lp <= 3u || lp >= 8u) { fp_scale = 10000.0; }
        let grad = f32(raw_grad) / fp_scale / 16384.0;

        var lr = 0.03;
        if (lp <= 2u) { lr = 0.01; }           // position xyz
        if (lp == 3u || lp == 12u) { lr = 0.01; } // scale sx, sy
        if (lp >= 4u && lp <= 6u) { lr = 0.02; }  // color
        if (lp == 7u) { lr = 0.01; }              // opacity
        if (lp >= 8u && lp <= 11u) { lr = 0.005; } // quaternion
        if (lp == 13u || lp == 14u) { lr = 0.01; } // shape

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

        var max_update = 0.01;
        if (lp <= 2u) { max_update = 0.005; }
        if (lp == 3u || lp == 12u) { max_update = 0.005; }
        if (lp >= 4u && lp <= 6u) { max_update = 0.001; }
        if (lp == 7u) { max_update = 0.0005; }
        if (lp >= 8u && lp <= 11u) { max_update = 0.005; }
        if (lp == 13u || lp == 14u) { max_update = 0.05; }
        let update = clamp(raw_update, -max_update, max_update);

        if (lp == 0u) { s.pos_sx.x -= update; pos_grad_norm2 += grad * grad; }
        else if (lp == 1u) { s.pos_sx.y -= update; pos_grad_norm2 += grad * grad; }
        else if (lp == 2u) { s.pos_sx.z -= update; pos_grad_norm2 += grad * grad; }
        else if (lp == 3u) { s.pos_sx.w = clamp(s.pos_sx.w - update, 0.001, 2.0); }
        else if (lp == 4u) { s.color.r = clamp(s.color.r - update, 0.05, 1.0); }
        else if (lp == 5u) { s.color.g = clamp(s.color.g - update, 0.05, 1.0); }
        else if (lp == 6u) { s.color.b = clamp(s.color.b - update, 0.05, 1.0); }
        else if (lp == 7u) { s.color.a = clamp(s.color.a - update, 0.01, 0.99); }
        else if (lp == 8u) { s.quat.x -= update; }
        else if (lp == 9u) { s.quat.y -= update; }
        else if (lp == 10u) { s.quat.z -= update; }
        else if (lp == 11u) { s.quat.w -= update; }
        else if (lp == 12u) { s.sy_shape.x = clamp(s.sy_shape.x - update, 0.001, 2.0); }
        else if (lp == 13u) { s.sy_shape.y = clamp(s.sy_shape.y - update, 0.1, 10.0); }
        else if (lp == 14u) { s.sy_shape.z = clamp(s.sy_shape.z - update, 0.01, 5.0); }
    }

    // Re-normalize quaternion
    let q_len = max(length(s.quat), 1e-8);
    s.quat = s.quat / q_len;

    adc.grad_accum[splat_id] += sqrt(pos_grad_norm2);

    let area = s.pos_sx.w * s.sy_shape.x;
    if (s.color.a < 0.05 || area < 0.0001) {
        s.color.a = 0.0;
    }

    splats.splats[splat_id] = s;
}
