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

@group(0) @binding(0) var<storage, read_write> beziers: BezierArray;

fn hash(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= {@NUM_BEZIERS}u) { return; }

    var seed = f32(i) * 1.61803398875 + 3.14159265359;
    
    let cx = hash(seed) * 2.0 - 1.0;
    seed = seed + 1.0;
    let cy = hash(seed) * 2.0 - 1.0;
    seed = seed + 1.0;
    let cz = (hash(seed) * 2.0 - 1.0) * 0.3;
    seed = seed + 1.0;
    
    let len = 0.2 + hash(seed) * 0.2;
    seed = seed + 1.0;
    let angle = hash(seed) * 6.28318530718;
    seed = seed + 1.0;
    
    let dx = cos(angle) * len;
    let dy = sin(angle) * len;
    
    let jx = (hash(seed) - 0.5) * 0.08;
    seed = seed + 1.0;
    let jy = (hash(seed) - 0.5) * 0.08;
    seed = seed + 1.0;
    let jz = (hash(seed) - 0.5) * 0.08;
    seed = seed + 1.0;

    let kx = (hash(seed) - 0.5) * 0.08;
    seed = seed + 1.0;
    let ky = (hash(seed) - 0.5) * 0.08;
    seed = seed + 1.0;
    let kz = (hash(seed) - 0.5) * 0.08;
    seed = seed + 1.0;

    var b: Bezier;
    
    // P0 (xyz, width)
    b.p0 = vec4f(cx - dx * 0.5, cy - dy * 0.5, cz, 0.02);
    
    // P1 (xyz, softness)
    b.p1 = vec4f(cx - dx * 0.15 + jx, cy - dy * 0.15 + jy, cz + jz, 0.005);
    
    // P2 (xyz, pad)
    b.p2 = vec4f(cx + dx * 0.15 + kx, cy + dy * 0.15 + ky, cz + kz, 0.0);
    
    // P3 (xyz, pad)
    b.p3 = vec4f(cx + dx * 0.5, cy + dy * 0.5, cz, 0.0);
    
    // color rgba
    let cr = hash(seed);
    seed = seed + 1.0;
    let cg = hash(seed);
    seed = seed + 1.0;
    let cb = hash(seed);
    seed = seed + 1.0;
    let initial_live_count = max(1u, {@NUM_BEZIERS}u / 8u);
    let alpha = select(0.0, 0.5, i < initial_live_count);
    b.color = vec4f(cr, cg, cb, alpha);
    
    // SH coefficients
    b.sh1_r = vec4f(0.0);
    b.sh1_g = vec4f(0.0);
    b.sh1_b = vec4f(0.0);
    b.sh1_a = vec4f(0.0);

    beziers.items[i] = b;
}
