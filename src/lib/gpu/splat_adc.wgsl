struct Splat {
    transform: vec4f,
    color: vec4f,
    rot_pad: vec4f,
}

struct SplatArray {
    splats: array<Splat, NUM_SPLATS>,
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
@group(0) @binding(1) var<storage, read_write> adam: AdamState;
@group(0) @binding(2) var<storage, read_write> adc: ADCArray;

var<workgroup> dead_indices: array<u32, NUM_SPLATS>;

@compute @workgroup_size(1, 1, 1)
fn main() {
    var dead_count = 0u;
    
    // Pass 1: Collect dead splats (free list)
    for (var i = 0u; i < NUM_SPLATS; i++) {
        if (splats.splats[i].color.a < 0.05) {
            dead_indices[dead_count] = i;
            dead_count++;
        }
    }
    
    let ADC_PERIOD = 50.0;
    let TAU_POS = 0.00005;
    let SPLIT_SCALE_THRESHOLD = 0.01;
    
    // Pass 2: Clone or Split
    for (var i = 0u; i < NUM_SPLATS; i++) {
        var s = splats.splats[i];
        if (s.color.a < 0.05) { continue; } // ignore dead
        
        let grad_accum = adc.grad_accum[i] / ADC_PERIOD;
        adc.grad_accum[i] = 0.0; // Reset for next period
        
        if (grad_accum > TAU_POS) {
            let scale_norm = length(s.transform.zw);
            
            // Only split/clone if we have free slots
            if (dead_count > 0u) {
                dead_count--;
                let new_idx = dead_indices[dead_count];
                
                var new_s = s;
                
                if (scale_norm > SPLIT_SCALE_THRESHOLD) {
                    // Split: Divide scale by 1.6, perturb positions
                    let split_factor = 0.625; // 1.0 / 1.6
                    s.transform.z *= split_factor;
                    s.transform.w *= split_factor;
                    new_s.transform.z *= split_factor;
                    new_s.transform.w *= split_factor;
                    
                    // Simple perturbation: move them apart along X axis based on their scale
                    s.transform.x -= s.transform.z * 0.05;
                    new_s.transform.x += s.transform.z * 0.05;
                } else {
                    // Clone: Keep same scale, slightly perturb position to avoid perfect overlap
                    let seed = f32(i) * 3.14159 + adam.t;
                    new_s.transform.x += (fract(sin(seed * 12.9898) * 43758.5453) - 0.5) * 0.002;
                    new_s.transform.y += (fract(sin(seed * 78.233) * 43758.5453) - 0.5) * 0.002;
                }
                
                // Write back
                splats.splats[i] = s;
                splats.splats[new_idx] = new_s;
                
                // Reset momentum (Adam state) for both original and new splat
                for (var p = 0u; p < 11u; p++) {
                    adam.m[i * 11u + p] = 0.0;
                    adam.v[i * 11u + p] = 0.0;
                    adam.m[new_idx * 11u + p] = 0.0;
                    adam.v[new_idx * 11u + p] = 0.0;
                }
            }
        }
    }
}
