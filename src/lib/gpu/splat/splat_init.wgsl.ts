import { interpolateWgslTemplate } from "../wgsl-templates/interpolateWgslTemplate.ts";
import { Splat } from "./Splat.wgsl.ts";

export default interpolateWgslTemplate`
struct SplatArray {
    splats: array<${Splat}, {@NUM_SPLATS}u>,
}

@group(0) @binding(0) var<storage, read_write> splats: SplatArray;

fn hash(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= {@NUM_SPLATS}u) { return; }

    var seed = f32(i) * 1.61803398875 + 3.14159265359;
    
    let cx = (hash(seed) * 2.0 - 1.0) * 0.3;
    seed = seed + 1.0;
    let cy = (hash(seed) * 2.0 - 1.0) * 0.3;
    seed = seed + 1.0;
    let cz = (hash(seed) * 2.0 - 1.0) * 0.3;
    seed = seed + 1.0;
    let sx = 0.1 + hash(seed) * 0.15;
    seed = seed + 1.0;
    
    let cr = hash(seed);
    seed = seed + 1.0;
    let cg = hash(seed);
    seed = seed + 1.0;
    let cb = hash(seed);
    seed = seed + 1.0;
    
    // First 512 splats are "alive" with opacity
    var a = 0.0;
    if (i < 512u) {
        a = 0.3 + hash(seed) * 0.4;
    }
    seed = seed + 1.0;

    let sy = 0.1 + hash(seed) * 0.15;
    seed = seed + 1.0;
    let sz = 0.1 + hash(seed) * 0.15;
    seed = seed + 1.0;

    var s: ${Splat};
    s.pos_sx = vec4f(cx, cy, cz, sx);
    s.color = vec4f(cr, cg, cb, a);
    s.quat = vec4f(1.0, 0.0, 0.0, 0.0);
    s.sy_shape = vec4f(sy, 2.0, 0.5, sz);
    
    s.sh1_r = vec4f(0.0);
    s.sh1_g = vec4f(0.0);
    s.sh1_b = vec4f(0.0);
    s.sh1_a = vec4f(0.0);

    splats.splats[i] = s;
}
`;