struct Bezier {
    p0: vec4f,    // x, y, z, width
    p1: vec4f,    // x, y, z, softness
    p2: vec4f,    // x, y, z, _pad
    p3: vec4f,    // x, y, z, _pad
    color: vec4f, // r, g, b, a
    sh1_r: vec4f,
    sh1_g: vec4f,
    sh1_b: vec4f,
    sh1_a: vec4f,
}

struct BezierArray {
    items: array<Bezier, {@NUM_BEZIERS}u>,
}

struct AdamState {
    m: array<f32, {@NUM_BEZIER_PARAMS}u>,
    v: array<f32, {@NUM_BEZIER_PARAMS}u>,
    t: f32,
    pixel_count: f32,
    no_kill: f32, // 1.0 = disable loss-based killing in ADC
    pad: f32,
}

struct ADCArray {
    grad_accum: array<f32, {@NUM_BEZIERS}u>,
    loss_accum: array<f32, {@NUM_BEZIERS}u>,
}

struct BezierUniforms {
    vp: mat4x4f,
    mode: f32,
    max_width: f32,
    prune_alpha_thresh: f32,
    prune_width_thresh: f32,
    bg_penalty: f32,
    _pad0: f32,
    _pad1: f32,
    adc_period_steps: f32,
    optim_width: f32,
    optim_height: f32,
    vp_inv: mat4x4f,
}

@group(0) @binding(0) var<storage, read_write> beziers: BezierArray;
@group(0) @binding(1) var<storage, read_write> adam: AdamState;
@group(0) @binding(2) var<storage, read_write> adc: ADCArray;
@group(0) @binding(3) var<storage, read_write> pixel_loss: array<atomic<i32>, {@PIXEL_LOSS_SIZE}u>;
@group(0) @binding(4) var<uniform> uniforms: BezierUniforms;
@group(0) @binding(5) var<storage, read_write> dead_indices: array<u32, {@NUM_BEZIERS}u>;
@group(0) @binding(6) var targetDepthTex: texture_2d<f32>;

// Reconstruct a world-space point from a pixel index, using the same reciprocal
// depth encoding as mesh.wgsl. depth_enc = 1 - 0.1/w  =>  w = 0.1/(1-depth_enc).
// We use depth_enc = 0.5 (mid-range) as a neutral spawn depth when no depth info
// is available — the optimizer will pull the curve to the correct depth quickly.
fn pixel_to_world(px_idx: u32, spawn_depth: f32) -> vec3f {
    let ow = u32(uniforms.optim_width);
    let oh = u32(uniforms.optim_height);
    let px_x = px_idx % ow;
    let px_y = px_idx / ow;
    let uv = (vec2f(f32(px_x), f32(px_y)) + 0.5) / vec2f(f32(ow), f32(oh));
    // NDC: y flipped (texture y=0 is top, NDC y=1 is top)
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let aspect = f32(ow) / f32(oh);
    // Recover w from reciprocal depth encoding
    let w = 0.1 / max(1.0 - spawn_depth, 1e-5);
    // Approximate z_clip ≈ w (valid for zFar=100 >> zNear=0.01)
    let clip = vec4f(ndc.x * w, ndc.y * w, w, w);
    let world = uniforms.vp_inv * clip;
    return world.xyz / world.w;
}

fn pixel_center_coord(px_idx: u32) -> vec2f {
    let ow = u32(uniforms.optim_width);
    let px_x = px_idx % ow;
    let px_y = px_idx / ow;
    return vec2f(f32(px_x), f32(px_y)) + 0.5;
}

fn pixel_coord_to_world(pixel_coord: vec2f, spawn_depth: f32) -> vec3f {
    let ow = u32(uniforms.optim_width);
    let oh = u32(uniforms.optim_height);
    let uv = pixel_coord / vec2f(f32(ow), f32(oh));
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let w = 0.1 / max(1.0 - spawn_depth, 1e-5);
    let clip = vec4f(ndc.x * w, ndc.y * w, w, w);
    let world = uniforms.vp_inv * clip;
    return world.xyz / world.w;
}

fn pixel_loss_at(x: i32, y: i32) -> f32 {
    let ow = u32(uniforms.optim_width);
    let oh = u32(uniforms.optim_height);
    let cx = u32(clamp(x, 0, i32(ow) - 1));
    let cy = u32(clamp(y, 0, i32(oh) - 1));
    return f32(atomicLoad(&pixel_loss[cy * ow + cx]));
}

fn edge_spawn_depth(px_idx: u32) -> f32 {
    let ow = u32(uniforms.optim_width);
    let oh = u32(uniforms.optim_height);
    let cx = i32(px_idx % ow);
    let cy = i32(px_idx / ow);
    var best_depth = 1.0;

    for (var dy: i32 = -2; dy <= 2; dy = dy + 1) {
        for (var dx: i32 = -2; dx <= 2; dx = dx + 1) {
            let sx = u32(clamp(cx + dx, 0, i32(ow) - 1));
            let sy = u32(clamp(cy + dy, 0, i32(oh) - 1));
            let d = textureLoad(targetDepthTex, vec2u(sx, sy), 0).r;
            best_depth = min(best_depth, d);
        }
    }

    // Silhouette edge pixels can land on the background side; when no nearby
    // surface depth is available, spawn far enough away to avoid huge strokes.
    return select(0.95, best_depth, best_depth < 0.995);
}

fn edge_spawn_tangent_px(px_idx: u32, seed: f32) -> vec2f {
    let ow = u32(uniforms.optim_width);
    let cx = i32(px_idx % ow);
    let cy = i32(px_idx / ow);

    var sxx = 0.0;
    var sxy = 0.0;
    var syy = 0.0;
    for (var dy: i32 = -2; dy <= 2; dy = dy + 1) {
        for (var dx: i32 = -2; dx <= 2; dx = dx + 1) {
            let w = pixel_loss_at(cx + dx, cy + dy);
            let fx = f32(dx);
            let fy = f32(dy);
            sxx += w * fx * fx;
            sxy += w * fx * fy;
            syy += w * fy * fy;
        }
    }

    let lambda = 0.5 * (sxx + syy + sqrt((sxx - syy) * (sxx - syy) + 4.0 * sxy * sxy));
    var tangent = vec2f(sxy, lambda - sxx);
    if (dot(tangent, tangent) < 1e-5) {
        tangent = vec2f(lambda - syy, sxy);
    }

    if (dot(tangent, tangent) < 1e-5) {
        let angle = fract(sin(seed * 127.1) * 43758.5453) * 6.28318;
        tangent = vec2f(cos(angle), sin(angle));
    }

    return normalize(tangent);
}

fn check_offscreen(clip: vec4f) -> bool {
    let m = f32({@BEZIER_OFFSCREEN_MARGIN});
    const DEPTH_NEAR_CULL = 0.1;
    if (clip.w < DEPTH_NEAR_CULL) { return true; }
    let ndc = clip.xy / max(clip.w, 1e-5);
    return abs(ndc.x) > m || abs(ndc.y) > m;
}

@compute @workgroup_size(1, 1, 1)
fn main() {
    var dead_count = 0u;

    for (var i = 0u; i < {@NUM_BEZIERS}u; i = i + 1u) {
        if (beziers.items[i].color.a < f32({@BEZIER_KILL_ALPHA_THRESH})) {
            dead_indices[dead_count] = i;
            dead_count = dead_count + 1u;
        }
    }
    
    let ADC_PERIOD = max(uniforms.adc_period_steps, 1.0);
    let TAU_POS = f32({@BEZIER_TAU_POS});
    let TAU_LOSS = f32({@BEZIER_TAU_LOSS});
    let TAU_EDGE_SUPPORT = f32({@BEZIER_TAU_EDGE_SUPPORT});
    let SPLIT_LEN_THRESHOLD = f32({@BEZIER_SPLIT_LEN_THRESHOLD});
    let edge_mode = uniforms.mode > 1.5;

    // ... (adam.t comment) ...

    // --- Pass 1: Kill offscreen or stuck/high-loss curves ---
    for (var i = 0u; i < {@NUM_BEZIERS}u; i = i + 1u) {
        var b = beziers.items[i];
        if (b.color.a < f32({@BEZIER_KILL_ALPHA_THRESH})) { continue; }

        let grad_norm = adc.grad_accum[i] / ADC_PERIOD;
        let signal_norm = adc.loss_accum[i] / ADC_PERIOD;

        // Color modes: kill curves that are contributing to loss but not moving.
        // Edge mode: loss_accum stores target-edge support, so recycle curves
        // that received no support from actual edge pixels.
        // Skipped in multiview (no_kill): a curve sampled infrequently accumulates
        // low average gradient even if it's legitimately needed from its views.
        let stale_color_curve = !edge_mode && grad_norm <= TAU_POS && signal_norm > TAU_LOSS;
        let unsupported_edge_curve = edge_mode && signal_norm <= TAU_EDGE_SUPPORT;
        if (adam.no_kill < 0.5 && (stale_color_curve || unsupported_edge_curve)) {
            beziers.items[i].color.a = 0.0;
            dead_indices[dead_count] = i;
            dead_count = dead_count + 1u;
            continue;
        }

        // Offscreen culling — skipped when no_kill is set (e.g. turntable training)
        // so curves that are temporarily off-screen from the current random view
        // are not destroyed; they will be visible again from other angles.
        if (adam.no_kill < 0.5) {
            let p0_clip = uniforms.vp * vec4f(b.p0.xyz, 1.0);
            let p1_clip = uniforms.vp * vec4f(b.p1.xyz, 1.0);
            let p2_clip = uniforms.vp * vec4f(b.p2.xyz, 1.0);
            let p3_clip = uniforms.vp * vec4f(b.p3.xyz, 1.0);

            if (check_offscreen(p0_clip) && check_offscreen(p1_clip) && check_offscreen(p2_clip) && check_offscreen(p3_clip)) {
                beziers.items[i].color.a = 0.0;
                dead_indices[dead_count] = i;
                dead_count = dead_count + 1u;
                continue;
            }
        }
    }

    // --- Pass 1.5: Split or clone high-gradient curves into dead slots ---
    for (var i = 0u; i < {@NUM_BEZIERS}u; i = i + 1u) {
        var b = beziers.items[i];
        if (b.color.a < f32({@BEZIER_KILL_ALPHA_THRESH})) { continue; }
        if (edge_mode) { continue; }

        let grad_norm = adc.grad_accum[i] / ADC_PERIOD;

        if (grad_norm <= TAU_POS) { continue; }
        if (dead_count == 0u) { continue; }

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
            let s  = (r0 + r1) * 0.5;

            b.p0 = vec4f(p0, b.p0.w);
            b.p1 = vec4f(q0, b.p1.w);
            b.p2 = vec4f(r0, 0.0);
            b.p3 = vec4f(s,  0.0);

            new_b.p0 = vec4f(s,  new_b.p0.w);
            new_b.p1 = vec4f(r1, new_b.p1.w);
            new_b.p2 = vec4f(q2, 0.0);
            new_b.p3 = vec4f(p3, 0.0);
        } else {
            let seed = f32(i) * 3.14159 + adam.t;
            let jx = (fract(sin(seed * 12.9898) * 43758.5453) - 0.5) * 0.001;
            let jy = (fract(sin(seed * 78.233)  * 43758.5453) - 0.5) * 0.001;
            let jz = (fract(sin(seed * 43.123)  * 43758.5453) - 0.5) * 0.001;
            let j = vec3f(jx, jy, jz);
            new_b.p0 = vec4f(b.p0.xyz + j, b.p0.w);
            new_b.p1 = vec4f(b.p1.xyz + j, b.p1.w);
            new_b.p2 = vec4f(b.p2.xyz + j, 0.0);
            new_b.p3 = vec4f(b.p3.xyz + j, 0.0);
        }

        beziers.items[i] = b;
        beziers.items[new_idx] = new_b;

        for (var p = 0u; p < {@BEZIER_PARAMS_PER}u; p = p + 1u) {
            adam.m[i * {@BEZIER_PARAMS_PER}u + p] = 0.0;
            adam.v[i * {@BEZIER_PARAMS_PER}u + p] = 0.0;
            adam.m[new_idx * {@BEZIER_PARAMS_PER}u + p] = 0.0;
            adam.v[new_idx * {@BEZIER_PARAMS_PER}u + p] = 0.0;
        }
    }

    // --- Pass 2: seed new beziers at the highest-loss uncovered pixels ---
    // Color layers fall back to random pixels to keep capacity. Edge mode only
    // respawns from missing target-edge signal; otherwise dead slots stay dead.
    // Cap at 512 spawns per ADC cycle to keep GPU time bounded while allowing rapid revival.
    let max_spawns = min(dead_count, {@BEZIER_MAX_SPAWNS}u);
    let stride = max(1u, {@PIXEL_LOSS_SIZE}u / 4096u); // scan ~4096 pixels per pass to find local maxima efficiently

    for (var spawn_i = 0u; spawn_i < max_spawns; spawn_i = spawn_i + 1u) {
        if (dead_count == 0u) { break; }

        // Find a high-loss pixel efficiently using a strided scan
        var best_px = 0u;
        var best_val = 0;
        let start_offset = (spawn_i * 997u) % stride;
        
        for (var px = start_offset; px < {@PIXEL_LOSS_SIZE}u; px = px + stride) {
            let v = atomicLoad(&pixel_loss[px]);
            if (v > best_val) {
                best_val = v;
                best_px = px;
            }
        }

        // If no loss signal, fall back to a random onscreen pixel so color-layer
        // dead slots stay filled. Edge mode skips the fallback to avoid creating
        // fresh off-edge strokes after unsupported ones were just killed.
        var spawn_px = best_px;
        if (best_val <= 0) {
            if (edge_mode) {
                break;
            }
            let ow = u32(uniforms.optim_width);
            let oh = u32(uniforms.optim_height);
            let seed_fb = f32(spawn_i) * 1234.5678 + adam.t * 0.1;
            let rx = u32(fract(sin(seed_fb * 12.9898) * 43758.5453) * f32(ow));
            let ry = u32(fract(sin(seed_fb * 78.233)  * 43758.5453) * f32(oh));
            spawn_px = ry * ow + rx;
        }

        dead_count = dead_count - 1u;
        let slot = dead_indices[dead_count];

        let seed = f32(spawn_px) * 1.61803 + f32(spawn_i) * 2.71828;

        var nb: Bezier;
        if (edge_mode) {
            let spawn_depth = edge_spawn_depth(spawn_px);
            let center_px = pixel_center_coord(spawn_px);
            let tangent_px = edge_spawn_tangent_px(spawn_px, seed);
            let half_len_px = f32({@BEZIER_EDGE_SPAWN_HALF_LEN_PX});

            nb.p0 = vec4f(pixel_coord_to_world(center_px - tangent_px * half_len_px, spawn_depth),        f32({@BEZIER_SPAWN_WIDTH}));
            nb.p1 = vec4f(pixel_coord_to_world(center_px - tangent_px * half_len_px * 0.33, spawn_depth), f32({@BEZIER_SPAWN_SOFTNESS}));
            nb.p2 = vec4f(pixel_coord_to_world(center_px + tangent_px * half_len_px * 0.33, spawn_depth), 0.0);
            nb.p3 = vec4f(pixel_coord_to_world(center_px + tangent_px * half_len_px, spawn_depth),        0.0);
        } else {
            let spawn_depth = 0.5;
            let center = pixel_to_world(spawn_px, spawn_depth);
            let angle = fract(sin(seed * 127.1) * 43758.5453) * 6.28318;
            let tx = cos(angle) * f32({@BEZIER_SPAWN_TANGENT_LEN});
            let tz = sin(angle) * f32({@BEZIER_SPAWN_TANGENT_LEN});
            let tangent = vec3f(tx, 0.0, tz);

            nb.p0 = vec4f(center - tangent,        f32({@BEZIER_SPAWN_WIDTH}));
            nb.p1 = vec4f(center - tangent * 0.33, f32({@BEZIER_SPAWN_SOFTNESS}));
            nb.p2 = vec4f(center + tangent * 0.33, 0.0);
            nb.p3 = vec4f(center + tangent,        0.0);
        }
        if (best_val > 0) {
            // Claim this pixel so the next pass finds a different peak.
            atomicStore(&pixel_loss[best_px], 0);
        }
        nb.color = vec4f(0.5, 0.5, 0.5, 0.5);
        nb.sh1_r = vec4f(0.0);
        nb.sh1_g = vec4f(0.0);
        nb.sh1_b = vec4f(0.0);
        nb.sh1_a = vec4f(0.0);

        beziers.items[slot] = nb;
        for (var p = 0u; p < {@BEZIER_PARAMS_PER}u; p = p + 1u) {
            adam.m[slot * {@BEZIER_PARAMS_PER}u + p] = 0.0;
            adam.v[slot * {@BEZIER_PARAMS_PER}u + p] = 0.0;
        }
    }

    // --- Pass 3: clone/split high-gradient live curves ---
    for (var i = 0u; i < {@NUM_BEZIERS}u; i = i + 1u) {
        if (dead_count == 0u) { break; } // Optimization: skip if no dead slots available
        var b = beziers.items[i];
        if (b.color.a < f32({@BEZIER_KILL_ALPHA_THRESH})) { continue; }
        if (edge_mode) { continue; }

        let grad_norm = adc.grad_accum[i] / ADC_PERIOD;
        if (grad_norm <= TAU_POS) { continue; }

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
            let s  = (r0 + r1) * 0.5;

            b.p0 = vec4f(p0, b.p0.w);
            b.p1 = vec4f(q0, b.p1.w);
            b.p2 = vec4f(r0, 0.0);
            b.p3 = vec4f(s,  0.0);

            new_b.p0 = vec4f(s,  new_b.p0.w);
            new_b.p1 = vec4f(r1, new_b.p1.w);
            new_b.p2 = vec4f(q2, 0.0);
            new_b.p3 = vec4f(p3, 0.0);
        } else {
            let seed = f32(i) * 3.14159 + adam.t;
            let jx = (fract(sin(seed * 12.9898) * 43758.5453) - 0.5) * 0.001;
            let jy = (fract(sin(seed * 78.233)  * 43758.5453) - 0.5) * 0.001;
            let jz = (fract(sin(seed * 43.123)  * 43758.5453) - 0.5) * 0.001;
            let j = vec3f(jx, jy, jz);
            new_b.p0 = vec4f(b.p0.xyz + j, b.p0.w);
            new_b.p1 = vec4f(b.p1.xyz + j, b.p1.w);
            new_b.p2 = vec4f(b.p2.xyz + j, 0.0);
            new_b.p3 = vec4f(b.p3.xyz + j, 0.0);
        }

        beziers.items[i] = b;
        beziers.items[new_idx] = new_b;

        for (var p = 0u; p < {@BEZIER_PARAMS_PER}u; p = p + 1u) {
            adam.m[i * {@BEZIER_PARAMS_PER}u + p] = 0.0;
            adam.v[i * {@BEZIER_PARAMS_PER}u + p] = 0.0;
            adam.m[new_idx * {@BEZIER_PARAMS_PER}u + p] = 0.0;
            adam.v[new_idx * {@BEZIER_PARAMS_PER}u + p] = 0.0;
        }
    }

    // Reset remaining pixel_loss and accumulators for the next ADC period
    for (var px = 0u; px < {@PIXEL_LOSS_SIZE}u; px = px + 1u) {
        atomicStore(&pixel_loss[px], 0);
    }
    for (var i = 0u; i < {@NUM_BEZIERS}u; i = i + 1u) {
        adc.grad_accum[i] = 0.0;
        adc.loss_accum[i] = 0.0;
    }
}
