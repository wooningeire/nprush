struct Splat {
    transform: vec4f,
    color: vec4f,
    rot_pad: vec4f,
}

struct SplatArray {
    splats: array<Splat, NUM_SPLATS>,
}

struct GradArray {
    data: array<atomic<i32>, NUM_PARAMS>,
}

@group(0) @binding(0) var<storage, read> splats: SplatArray;
@group(0) @binding(1) var<storage, read_write> grads: GradArray;
@group(0) @binding(2) var targetTex: texture_2d<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3u) {
    let dims = textureDimensions(targetTex);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }
    
    let uv = (vec2f(global_id.xy) + vec2f(0.5)) / vec2f(dims.xy);
    var p = uv * 2.0 - 1.0;
    p.y = -p.y;
    
    let tgt_color = textureLoad(targetTex, global_id.xy, 0).rgb;

    var alphas = array<f32, NUM_SPLATS>();
    var Ts = array<f32, NUM_SPLATS_PLUS_ONE>();
    Ts[0] = 1.0;
    var C_pred = vec3f(0.0);
    
    for (var i = 0u; i < NUM_SPLATS; i++) {
        let s = splats.splats[i];
        let pos = s.transform.xy;
        let scale = s.transform.zw;
        let rot = s.rot_pad.x;
        let color = s.color.rgb;
        let opacity = s.color.a;
        
        let d = p - pos;
        let co = cos(rot);
        let si = sin(rot);
        let local_p = vec2f(co * d.x + si * d.y, -si * d.x + co * d.y);
        let safe_scale = max(vec2f(0.0001), scale);
        let scaled_p = local_p / safe_scale;
        let r = max(length(scaled_p), 0.0001);
        let power = -s.rot_pad.z * pow(r, s.rot_pad.y);
        
        var a = 0.0;
        if (power > -15.0) {
            a = exp(power) * opacity;
        }
        
        a = clamp(a, 0.0, 0.999);
        
        alphas[i] = a;
        C_pred += Ts[i] * a * color;
        Ts[i+1] = Ts[i] * (1.0 - a);
    }

    let background = vec3f(0.1);
    C_pred += Ts[NUM_SPLATS] * background;
    
    let dC = 2.0 * (C_pred - tgt_color);
    var dT = dot(dC, background);
    
    let FP_SCALE = 1000.0;

    for (var j = 0u; j < NUM_SPLATS; j++) {
        let i = NUM_SPLATS_MINUS_ONE - j;
        let s = splats.splats[i];
        let pos = s.transform.xy;
        let scale = s.transform.zw;
        let rot = s.rot_pad.x;
        let color = s.color.rgb;
        let opacity = s.color.a;

        let a = alphas[i];
        let T_prev = Ts[i];
        
        let dColor = dC * (T_prev * a);
        let da = dT * (-T_prev) + dot(dC, T_prev * color);
        dT = dT * (1.0 - a) + dot(dC, a * color);

        let d = p - pos;
        let co = cos(rot);
        let si = sin(rot);
        let local_p = vec2f(co * d.x + si * d.y, -si * d.x + co * d.y);
        let safe_scale = max(vec2f(0.0001), scale);
        let scaled_p = local_p / safe_scale;
        let r = max(length(scaled_p), 0.0001);
        let shape_a = s.rot_pad.y;
        let shape_b = s.rot_pad.z;
        let power = -shape_b * pow(r, shape_a);
        
        var d_opacity = 0.0;
        var d_pos = vec2f(0.0);
        var d_scale = vec2f(0.0);
        var d_rot = 0.0;
        var d_shape_a = 0.0;
        var d_shape_b = 0.0;
        
        if (power > -15.0) {
            let a_unscaled = exp(power);
            d_opacity = da * a_unscaled;
            let d_power = da * opacity * a_unscaled;
            
            let r_pow_a = pow(r, shape_a);
            let r_pow_a_minus_2 = pow(r, shape_a - 2.0);
            
            d_shape_a = d_power * (-shape_b * r_pow_a * log(r));
            d_shape_b = d_power * (-r_pow_a);
            
            let d_scaled_p = d_power * (-shape_b * shape_a * r_pow_a_minus_2) * scaled_p;
            d_scale = d_scaled_p * (-local_p / (safe_scale * safe_scale));
            let d_local_p = d_scaled_p / safe_scale;
            
            let d_co = d_local_p.x * d.x + d_local_p.y * d.y;
            let d_si = d_local_p.x * d.y - d_local_p.y * d.x;
            d_rot = d_co * (-si) + d_si * co;
            
            let d_d = vec2f(
                d_local_p.x * co - d_local_p.y * si,
                d_local_p.x * si + d_local_p.y * co
            );
            d_pos = -d_d;
        }

        let FP_SCALE_POS = 10000.0;
        let FP_SCALE_COL = 100000.0;
        
        let base_idx = i * 11u;
        atomicAdd(&grads.data[base_idx + 0u], i32(d_pos.x * FP_SCALE_POS));
        atomicAdd(&grads.data[base_idx + 1u], i32(d_pos.y * FP_SCALE_POS));
        atomicAdd(&grads.data[base_idx + 2u], i32(d_scale.x * FP_SCALE_POS));
        atomicAdd(&grads.data[base_idx + 3u], i32(d_scale.y * FP_SCALE_POS));
        atomicAdd(&grads.data[base_idx + 4u], i32(dColor.r * FP_SCALE_COL));
        atomicAdd(&grads.data[base_idx + 5u], i32(dColor.g * FP_SCALE_COL));
        atomicAdd(&grads.data[base_idx + 6u], i32(dColor.b * FP_SCALE_COL));
        atomicAdd(&grads.data[base_idx + 7u], i32(d_opacity * FP_SCALE_COL));
        atomicAdd(&grads.data[base_idx + 8u], i32(d_rot * FP_SCALE_POS));
        atomicAdd(&grads.data[base_idx + 9u], i32(d_shape_a * FP_SCALE_POS));
        atomicAdd(&grads.data[base_idx + 10u], i32(d_shape_b * FP_SCALE_POS));
    }
}
