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
    max_width: f32, // 0 = use default cap (0.05)
    _pad: vec2f,
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

    // lr per param: 0-11=positions (0.005), 12-14=color (0.02), 15=opacity (0.01), 16-17=width/softness (0.002)
    const lr_table = array<f32, 18>(
        0.00001, 0.00001, 0.00001, 0.005, 0.005, 0.005,
        0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
        0.02,  0.02,  0.02,  0.01,  0.002, 0.002
    );
    // max_update per param
    const mu_table = array<f32, 18>(
        0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
        0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
        0.001, 0.001, 0.001, 0.0005, 0.005, 0.005
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
            0.05, 0.05, 0.05, 0.00, 0.001, 0.001
        );
        let width_hi = select(0.05, uniforms.max_width, uniforms.max_width > 0.0);
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

    // Prune very thin or transparent beziers
    b.color.a = select(b.color.a, 0.0, b.color.a < 0.001 || b.p0.w <= 0.001);

    beziers.items[bezier_id] = b;
}
