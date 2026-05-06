struct Splat {
    pos_sx: vec4f,
    color: vec4f,
    quat: vec4f,
    sy_shape: vec4f,
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

@compute @workgroup_size(1, 1, 1)
fn main() {
    var next_dead_search = 0u;
    let ADC_PERIOD = 25.0;
    let TAU_POS = 0.00005;
    let SPLIT_SCALE_THRESHOLD = 0.01;

    for (var i = 0u; i < NUM_SPLATS; i++) {
        var s = splats.splats[i];
        if (s.color.a < 0.05) { continue; }

        let grad_accum = adc.grad_accum[i] / ADC_PERIOD;
        adc.grad_accum[i] = 0.0;

        if (grad_accum > TAU_POS) {
            let scale_norm = length(vec3f(s.pos_sx.w, s.sy_shape.x, s.sy_shape.w));

            var found_dead = false;
            var new_idx = 0u;
            for (var d = next_dead_search; d < NUM_SPLATS; d++) {
                if (splats.splats[d].color.a < 0.05) {
                    new_idx = d;
                    found_dead = true;
                    next_dead_search = d + 1u;
                    break;
                }
            }
            if (!found_dead) { next_dead_search = NUM_SPLATS; }

            if (found_dead) {
                var new_s = s;

                if (scale_norm > SPLIT_SCALE_THRESHOLD) {
                    let split_factor = 0.625;
                    s.pos_sx.w *= split_factor;
                    s.sy_shape.x *= split_factor;
                    s.sy_shape.w *= split_factor;
                    new_s.pos_sx.w *= split_factor;
                    new_s.sy_shape.x *= split_factor;
                    new_s.sy_shape.w *= split_factor;
                    s.pos_sx.x -= s.pos_sx.w * 0.05;
                    new_s.pos_sx.x += s.pos_sx.w * 0.05;
                } else {
                    let seed = f32(i) * 3.14159 + adam.t;
                    new_s.pos_sx.x += (fract(sin(seed * 12.9898) * 43758.5453) - 0.5) * 0.002;
                    new_s.pos_sx.y += (fract(sin(seed * 78.233) * 43758.5453) - 0.5) * 0.002;
                }

                splats.splats[i] = s;
                splats.splats[new_idx] = new_s;

                for (var p = 0u; p < 16u; p++) {
                    adam.m[i * 16u + p] = 0.0;
                    adam.v[i * 16u + p] = 0.0;
                    adam.m[new_idx * 16u + p] = 0.0;
                    adam.v[new_idx * 16u + p] = 0.0;
                }
            }
        }
    }
}
