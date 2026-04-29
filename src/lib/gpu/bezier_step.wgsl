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

struct StepUniforms {
    vp: mat4x4f,
    mode: f32,
    _pad: vec3f,
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
    
    if (bezier_id >= NUM_BEZIERS) {
        return;
    }

    var b = beziers.items[bezier_id];
    let base_idx = bezier_id * 18u;
    let t = current_t + 1.0;
    var pos_grad_norm2 = 0.0;

    for (var lp = 0u; lp < 18u; lp++) {
        let param_idx = base_idx + lp;
        let raw_grad = atomicExchange(&grads.data[param_idx], 0);
        
        var fp_scale = 100000.0;
        if (lp < 12u || lp >= 16u) {
            fp_scale = 10000.0;
        }
        let grad = f32(raw_grad) / fp_scale / 16384.0;

        var lr = 0.005; // Base LR for positions
        if (lp >= 12u && lp <= 14u) { lr = 0.02; }  // Color rgb
        if (lp == 15u) { lr = 0.01; }               // Opacity
        if (lp == 16u || lp == 17u) { lr = 0.002; } // Width/softness

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
        if (lp < 12u) { max_update = 0.005; }
        if (lp >= 12u && lp <= 14u) { max_update = 0.001; }
        if (lp == 15u) { max_update = 0.0005; }
        if (lp == 16u || lp == 17u) { max_update = 0.005; }
        
        let update = clamp(raw_update, -max_update, max_update);

        // Apply update to the corresponding parameter
        if (lp == 0u) { b.p0.x -= update; pos_grad_norm2 += grad * grad; }
        else if (lp == 1u) { b.p0.y -= update; pos_grad_norm2 += grad * grad; }
        else if (lp == 2u) { b.p0.z -= update; pos_grad_norm2 += grad * grad; }
        else if (lp == 3u) { b.p1.x -= update; pos_grad_norm2 += grad * grad; }
        else if (lp == 4u) { b.p1.y -= update; pos_grad_norm2 += grad * grad; }
        else if (lp == 5u) { b.p1.z -= update; pos_grad_norm2 += grad * grad; }
        else if (lp == 6u) { b.p2.x -= update; pos_grad_norm2 += grad * grad; }
        else if (lp == 7u) { b.p2.y -= update; pos_grad_norm2 += grad * grad; }
        else if (lp == 8u) { b.p2.z -= update; pos_grad_norm2 += grad * grad; }
        else if (lp == 9u) { b.p3.x -= update; pos_grad_norm2 += grad * grad; }
        else if (lp == 10u) { b.p3.y -= update; pos_grad_norm2 += grad * grad; }
        else if (lp == 11u) { b.p3.z -= update; pos_grad_norm2 += grad * grad; }
        else if (lp == 12u) { b.color.r = clamp(b.color.r - update, 0.05, 1.0); }
        else if (lp == 13u) { b.color.g = clamp(b.color.g - update, 0.05, 1.0); }
        else if (lp == 14u) { b.color.b = clamp(b.color.b - update, 0.05, 1.0); }
        else if (lp == 15u) { b.color.a = clamp(b.color.a - update, 0.00, 0.99); }
        else if (lp == 16u) { b.p0.w = clamp(b.p0.w - update, 0.001, 0.05); }
        else if (lp == 17u) { b.p1.w = clamp(b.p1.w - update, 0.001, 0.03); }
    }

    adc.grad_accum[bezier_id] += sqrt(pos_grad_norm2);

    // Prune very thin or transparent beziers
    if (b.color.a < 0.001 || b.p0.w <= 0.001) {
        b.color.a = 0.0;
    }

    beziers.items[bezier_id] = b;
}
