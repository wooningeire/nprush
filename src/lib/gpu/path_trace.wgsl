// Progressive path tracer — diffuse-only, 1 sample per pixel per frame.
// Uses a flat BVH for O(log n) ray-scene intersection.
//
// Vertex layout (stride 10 f32): [px,py,pz, nx,ny,nz, r,g,b,a]
//
// BVH node layout (8 f32 = 32 bytes):
//   [min_x, min_y, min_z, data0,  max_x, max_y, max_z, data1]
//   Internal: data0 = left child index, data1 = right child index
//   Leaf:     data0 = first tri index in bvh_tris,
//             data1 = (count | 0x80000000)
//
// bvh_tris: reordered [i0, i1, i2] per triangle (u32)

const LEAF_FLAG: u32 = 0x80000000u;
const VSTRIDE:   u32 = 10u;
const MAX_STACK: u32 = 64u;

struct PTUniforms {
    invViewProjMat: mat4x4f, // offset  0
    frame:          u32,     // offset 64
    num_tris:       u32,     // offset 68  (unused — BVH knows its own count)
    out_w:          u32,     // offset 72
    out_h:          u32,     // offset 76
}

@group(0) @binding(0) var<uniform>             pt_uniforms: PTUniforms;
@group(0) @binding(1) var<storage, read>       vertices:    array<f32>;
@group(0) @binding(2) var<storage, read>       bvh_nodes:   array<f32>; // 8 f32 per node
@group(0) @binding(3) var<storage, read>       bvh_tris:    array<u32>; // reordered i0,i1,i2
@group(0) @binding(4) var<storage, read_write> accum:       array<f32>; // w*h*4 f32
@group(0) @binding(5) var                      env_tex:     texture_2d<f32>;
@group(0) @binding(6) var                      env_sampler: sampler;

// ── RNG ───────────────────────────────────────────────────────────────────────
fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
fn rand2(seed: ptr<function, u32>) -> vec2f {
    *seed = pcg(*seed); let a = *seed;
    *seed = pcg(*seed); let b = *seed;
    return vec2f(f32(a), f32(b)) / 4294967296.0;
}

// ── Geometry ──────────────────────────────────────────────────────────────────
struct Vertex { pos: vec3f, norm: vec3f, color: vec3f }

fn load_vert(idx: u32) -> Vertex {
    let b = idx * VSTRIDE;
    return Vertex(
        vec3f(vertices[b],   vertices[b+1u], vertices[b+2u]),
        vec3f(vertices[b+3u],vertices[b+4u], vertices[b+5u]),
        vec3f(vertices[b+6u],vertices[b+7u], vertices[b+8u]),
    );
}

struct Hit { hit: bool, t: f32, norm: vec3f, color: vec3f }

fn intersect_tri(ro: vec3f, rd: vec3f, i0: u32, i1: u32, i2: u32, t_max: f32) -> Hit {
    var res: Hit; res.hit = false;
    let v0 = load_vert(i0); let v1 = load_vert(i1); let v2 = load_vert(i2);
    let e1 = v1.pos - v0.pos;
    let e2 = v2.pos - v0.pos;
    let h  = cross(rd, e2);
    let a  = dot(e1, h);
    if (abs(a) < 1e-7) { return res; }
    let f = 1.0 / a;
    let s = ro - v0.pos;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) { return res; }
    let q = cross(s, e1);
    let v = f * dot(rd, q);
    if (v < 0.0 || u + v > 1.0) { return res; }
    let t = f * dot(e2, q);
    if (t < 1e-4 || t >= t_max) { return res; }
    let w = 1.0 - u - v;
    res.hit   = true;
    res.t     = t;
    res.norm  = normalize(w * v0.norm + u * v1.norm + v * v2.norm);
    res.color = w * v0.color + u * v1.color + v * v2.color;
    return res;
}

// ── AABB slab test ────────────────────────────────────────────────────────────
fn aabb_hit(ro: vec3f, inv_rd: vec3f, node_base: u32, t_max: f32) -> bool {
    let mn = vec3f(bvh_nodes[node_base],   bvh_nodes[node_base+1u], bvh_nodes[node_base+2u]);
    let mx = vec3f(bvh_nodes[node_base+4u],bvh_nodes[node_base+5u], bvh_nodes[node_base+6u]);
    let t0 = (mn - ro) * inv_rd;
    let t1 = (mx - ro) * inv_rd;
    let tmin = max(max(min(t0.x,t1.x), min(t0.y,t1.y)), min(t0.z,t1.z));
    let tmax = min(min(max(t0.x,t1.x), max(t0.y,t1.y)), max(t0.z,t1.z));
    return tmax >= max(tmin, 0.0) && tmin < t_max;
}

// ── BVH traversal ─────────────────────────────────────────────────────────────
fn scene_hit(ro: vec3f, rd: vec3f) -> Hit {
    let inv_rd = vec3f(1.0) / rd;
    var best: Hit; best.hit = false; best.t = 1e30;

    var stack: array<u32, 64>;
    var sp: u32 = 0u;
    stack[sp] = 0u; sp++;

    while (sp > 0u) {
        sp--;
        let node_idx = stack[sp];
        let nb = node_idx * 8u;

        if (!aabb_hit(ro, inv_rd, nb, best.t)) { continue; }

        let data1 = bitcast<u32>(bvh_nodes[nb + 7u]);

        if ((data1 & LEAF_FLAG) != 0u) {
            // Leaf: test triangles
            let tri_start = bitcast<u32>(bvh_nodes[nb + 3u]);
            let tri_count = data1 & ~LEAF_FLAG;
            for (var ti = 0u; ti < tri_count; ti++) {
                let base = (tri_start + ti) * 3u;
                let i0 = bvh_tris[base];
                let i1 = bvh_tris[base + 1u];
                let i2 = bvh_tris[base + 2u];
                let h = intersect_tri(ro, rd, i0, i1, i2, best.t);
                if (h.hit) { best = h; }
            }
        } else {
            // Internal: push children
            let left  = bitcast<u32>(bvh_nodes[nb + 3u]);
            let right = data1;
            if (sp + 1u < MAX_STACK) {
                stack[sp] = left;  sp++;
                stack[sp] = right; sp++;
            }
        }
    }
    return best;
}

// ── Environment ───────────────────────────────────────────────────────────────
const PI: f32 = 3.14159265358979;

fn sample_env(dir: vec3f) -> vec3f {
    let u = atan2(dir.z, dir.x) / (2.0 * PI) + 0.5;
    let v = 0.5 - asin(clamp(dir.y, -1.0, 1.0)) / PI;
    return textureSampleLevel(env_tex, env_sampler, vec2f(u, v), 0.0).rgb;
}

// ── Cosine-weighted hemisphere ────────────────────────────────────────────────
fn cosine_hemisphere(n: vec3f, seed: ptr<function, u32>) -> vec3f {
    let r = rand2(seed);
    let phi = 2.0 * PI * r.x;
    let sr  = sqrt(r.y);
    let x = cos(phi) * sr; let y = sin(phi) * sr; let z = sqrt(max(0.0, 1.0 - r.y));
    var up = vec3f(0.0, 1.0, 0.0);
    if (abs(n.y) > 0.99) { up = vec3f(1.0, 0.0, 0.0); }
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return normalize(t * x + b * y + n * z);
}

// ── Main ──────────────────────────────────────────────────────────────────────
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let out_w = pt_uniforms.out_w;
    let out_h = pt_uniforms.out_h;
    if (gid.x >= out_w || gid.y >= out_h) { return; }

    let pixel_idx = gid.y * out_w + gid.x;
    var seed = pcg(pixel_idx ^ (pt_uniforms.frame * 2654435761u));

    let jitter = rand2(&seed) - vec2f(0.5);
    let uv  = (vec2f(gid.xy) + vec2f(0.5) + jitter) / vec2f(f32(out_w), f32(out_h));
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

    let near_h = pt_uniforms.invViewProjMat * vec4f(ndc, 0.0, 1.0);
    let far_h  = pt_uniforms.invViewProjMat * vec4f(ndc, 1.0, 1.0);
    let ro = near_h.xyz / near_h.w;
    let rd = normalize(far_h.xyz / far_h.w - ro);

    var radiance   = vec3f(0.0);
    var throughput = vec3f(1.0);
    var ray_o = ro;
    var ray_d = rd;

    for (var bounce = 0; bounce < 3; bounce++) {
        let hit = scene_hit(ray_o, ray_d);
        if (!hit.hit) {
            radiance += throughput * sample_env(ray_d);
            break;
        }
        let n = select(hit.norm, -hit.norm, dot(hit.norm, ray_d) > 0.0);
        throughput *= hit.color; // diffuse: albedo (cos/pi and pdf cancel)
        ray_o = ray_o + ray_d * hit.t + n * 1e-4;
        ray_d = cosine_hemisphere(n, &seed);
    }

    let base = pixel_idx * 4u;
    accum[base]      += radiance.r;
    accum[base + 1u] += radiance.g;
    accum[base + 2u] += radiance.b;
    accum[base + 3u] += 1.0;
}
