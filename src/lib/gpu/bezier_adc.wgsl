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

struct AdamState {
    m: array<f32, NUM_BEZIER_PARAMS>,
    v: array<f32, NUM_BEZIER_PARAMS>,
    t: f32,
    pixel_count: f32,
    pad: vec2f,
}

struct ADCArray {
    grad_accum: array<f32, NUM_BEZIERS>,
}

@group(0) @binding(0) var<storage, read_write> beziers: BezierArray;
@group(0) @binding(1) var<storage, read_write> adam: AdamState;
@group(0) @binding(2) var<storage, read_write> adc: ADCArray;

var<workgroup> dead_indices: array<u32, NUM_BEZIERS>;

@compute @workgroup_size(1, 1, 1)
fn main() {
    var dead_count = 0u;

    for (var i = 0u; i < NUM_BEZIERS; i = i + 1u) {
        if (beziers.items[i].color.a < 0.005) {
            dead_indices[dead_count] = i;
            dead_count = dead_count + 1u;
        }
    }

    let ADC_PERIOD = 25.0;
    let TAU_POS = 0.005;
    let SPLIT_LEN_THRESHOLD = 0.25;
    let MIN_DEAD_FRACTION = 0.3;
    let MIN_DEAD_SLOTS = u32(f32(NUM_BEZIERS) * MIN_DEAD_FRACTION);

    for (var i = 0u; i < NUM_BEZIERS; i = i + 1u) {
        var b = beziers.items[i];
        if (b.color.a < 0.005) { continue; }

        let grad_norm = adc.grad_accum[i] / ADC_PERIOD;
        adc.grad_accum[i] = 0.0;

        if (grad_norm <= TAU_POS) {
            if (adam.m[i * 18u + 15u] > 1e-5) {
                beziers.items[i].color.a = 0.0;
            }
            continue;
        }

        if (dead_count <= MIN_DEAD_SLOTS) { continue; }

        dead_count = dead_count - 1u;
        let new_idx = dead_indices[dead_count];

        let p0 = b.p0.xyz;
        let p1 = b.p1.xyz;
        let p2 = b.p2.xyz;
        let p3 = b.p3.xyz;

        let mid = (p0 + 3.0 * p1 + 3.0 * p2 + p3) * 0.125;
        let len_approx = length(mid - p0) + length(p3 - mid);

        var new_b = b;

        if (len_approx > SPLIT_LEN_THRESHOLD) {
            let q0 = (p0 + p1) * 0.5;
            let q1 = (p1 + p2) * 0.5;
            let q2 = (p2 + p3) * 0.5;
            let r0 = (q0 + q1) * 0.5;
            let r1 = (q1 + q2) * 0.5;
            let s = (r0 + r1) * 0.5;
            
            b.p0 = vec4f(p0, b.p0.w);
            b.p1 = vec4f(q0, b.p1.w);
            b.p2 = vec4f(r0, 0.0);
            b.p3 = vec4f(s, 0.0);
            
            new_b.p0 = vec4f(s, new_b.p0.w);
            new_b.p1 = vec4f(r1, new_b.p1.w);
            new_b.p2 = vec4f(q2, 0.0);
            new_b.p3 = vec4f(p3, 0.0);
        } else {
            let seed = f32(i) * 3.14159 + adam.t;
            let jx = (fract(sin(seed * 12.9898) * 43758.5453) - 0.5) * 0.003;
            let jy = (fract(sin(seed * 78.233) * 43758.5453) - 0.5) * 0.003;
            let jz = (fract(sin(seed * 43.123) * 43758.5453) - 0.5) * 0.003;
            let j = vec3f(jx, jy, jz);
            
            new_b.p0 = vec4f(b.p0.xyz + j, b.p0.w);
            new_b.p1 = vec4f(b.p1.xyz + j, b.p1.w);
            new_b.p2 = vec4f(b.p2.xyz + j, 0.0);
            new_b.p3 = vec4f(b.p3.xyz + j, 0.0);
        }

        beziers.items[i] = b;
        beziers.items[new_idx] = new_b;

        for (var p = 0u; p < 18u; p = p + 1u) {
            adam.m[i * 18u + p] = 0.0;
            adam.v[i * 18u + p] = 0.0;
            adam.m[new_idx * 18u + p] = 0.0;
            adam.v[new_idx * 18u + p] = 0.0;
        }
    }
}
