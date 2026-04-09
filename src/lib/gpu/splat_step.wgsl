struct Splat {
    transform: vec4f,
    color: vec4f,
    rot_pad: vec4f,
}

struct SplatArray {
    splats: array<Splat, 512>,
}

struct GradArray {
    data: array<atomic<i32>, 4608>,
}

struct AdamState {
    m: array<f32, 4608>,
    v: array<f32, 4608>,
    t: f32,
    pad: vec3f,
}

@group(0) @binding(0) var<storage, read_write> splats: SplatArray;
@group(0) @binding(1) var<storage, read_write> grads: GradArray;
@group(0) @binding(2) var<storage, read_write> adam: AdamState;

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let splat_id = global_id.x;
    
    let current_t = adam.t;
    workgroupBarrier();

    if (splat_id == 0u) {
        adam.t = current_t + 1.0;
    }
    
    if (splat_id >= 512u) { return; }

    var s = splats.splats[splat_id];
    let base_idx = splat_id * 9u;
    let t = current_t + 1.0;
    
    for (var local_param = 0u; local_param < 9u; local_param++) {
        let param_idx = base_idx + local_param;
        let raw_grad = atomicExchange(&grads.data[param_idx], 0);
        var fp_scale = 100000.0;
        if (local_param <= 3u || local_param == 8u) {
            fp_scale = 10000.0;
        }
        let grad = f32(raw_grad) / fp_scale / 16384.0;
        
        let beta1 = 0.9;
        let beta2 = 0.999;
        let epsilon = 1e-8;
        
        // Learning rates per parameter type
        var lr = 0.03;
        // position
        if (local_param == 0u || local_param == 1u) { lr = 0.05; }
        // scale
        if (local_param == 2u || local_param == 3u) { lr = 0.01; }
        // color
        if (local_param >= 4u && local_param <= 6u) { lr = 0.02; }
        // opacity
        if (local_param == 7u) { lr = 0.01; }
        // rotation
        if (local_param == 8u) { lr = 0.02; }
        
        var m = adam.m[param_idx];
        var v = adam.v[param_idx];
        m = beta1 * m + (1.0 - beta1) * grad;
        v = beta2 * v + (1.0 - beta2) * grad * grad;
        adam.m[param_idx] = m;
        adam.v[param_idx] = v;
        
        let m_hat = m / (1.0 - pow(beta1, t));
        let v_hat = v / (1.0 - pow(beta2, t));
        let raw_update = lr * m_hat / (sqrt(v_hat) + epsilon);
        
        // Per-parameter update clip to enforce position-first convergence
        var max_update = 0.01;
        if (local_param == 0u || local_param == 1u) { max_update = 0.02; }  // position
        if (local_param == 2u || local_param == 3u) { max_update = 0.005; }  // scale
        if (local_param >= 4u && local_param <= 6u) { max_update = 0.001; }  // color
        if (local_param == 7u) { max_update = 0.0005; }  // opacity
        if (local_param == 8u) { max_update = 0.01; }  // rotation
        let update = clamp(raw_update, -max_update, max_update);
        
        if (local_param == 0u) { s.transform.x -= update; }
        else if (local_param == 1u) { s.transform.y -= update; }
        else if (local_param == 2u) { s.transform.z = clamp(s.transform.z - update, 0.001, 2.0); }
        else if (local_param == 3u) { s.transform.w = clamp(s.transform.w - update, 0.001, 2.0); }
        else if (local_param == 4u) { s.color.r = clamp(s.color.r - update, 0.05, 1.0); }
        else if (local_param == 5u) { s.color.g = clamp(s.color.g - update, 0.05, 1.0); }
        else if (local_param == 6u) { s.color.b = clamp(s.color.b - update, 0.05, 1.0); }
        else if (local_param == 7u) { s.color.a = clamp(s.color.a - update, 0.01, 0.99); }
        else if (local_param == 8u) { s.rot_pad.x -= update; }
    }
    
    // Respawn splats that have shrunk below the px threshold or alpha threshold
    if (s.transform.z * s.transform.w < 0.0004 || s.color.a < 0.05) {
        let seed = f32(splat_id) * 3.14159 + t;
        s.transform.x = (fract(sin(seed * 12.9898) * 43758.5453) * 2.0 - 1.0) * 0.5;
        s.transform.y = (fract(sin(seed * 78.233) * 43758.5453) * 2.0 - 1.0) * 0.5;
        s.transform.z = 0.1 + fract(sin(seed * 39.346) * 43758.5453) * 0.15;
        s.transform.w = 0.1 + fract(sin(seed * 54.123) * 43758.5453) * 0.15;
        s.color.r = fract(sin(seed * 83.456) * 43758.5453);
        s.color.g = fract(sin(seed * 13.579) * 43758.5453);
        s.color.b = fract(sin(seed * 97.531) * 43758.5453);
        s.color.a = 0.3 + fract(sin(seed * 24.680) * 43758.5453) * 0.4;
        s.rot_pad.x = fract(sin(seed * 86.420) * 43758.5453) * 6.283185307;
        
        // Reset Adam state
        for (var local_param = 0u; local_param < 9u; local_param++) {
            let param_idx = base_idx + local_param;
            adam.m[param_idx] = 0.0;
            adam.v[param_idx] = 0.0;
        }
    }
    
    splats.splats[splat_id] = s;
}
