// Progressive path tracer — diffuse-only, 1 sample per pixel per frame.
// Uses a flat BVH for O(log n) ray-scene intersection.
//
// Vertex layout (stride 12 f32): [px,py,pz, nx,ny,nz, r,g,b,a, u,v]
//
// BVH node layout (8 f32 = 32 bytes):
//   [min_x, min_y, min_z, data0,  max_x, max_y, max_z, data1]
//   Internal: data0 = left child index, data1 = right child index
//   Leaf:     data0 = first tri index in bvh_tris,
//             data1 = (count | 0x80000000)
//
// bvh_tris: reordered [i0, i1, i2] per triangle (u32)

const LEAF_FLAG: u32 = 0x80000000u;
const VSTRIDE:   u32 = 12u;
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

struct MeshSplat {
    pos: vec3f,
    radius: f32,
    color: vec3f,
    hardness: f32,
}

struct MeshUniforms {
    splatsEnabled: f32,
    numSplats: f32,
}

@group(0) @binding(7) var<storage, read>       textureSplats: array<MeshSplat, {@MESH_SPLAT_MAX_COUNT}>;
@group(0) @binding(8) var<uniform>             meshUniforms:  MeshUniforms;

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
// color.a encodes material type: 1.0 = diffuse, 0.0 = perfect specular mirror
struct Vertex { pos: vec3f, norm: vec3f, color: vec4f }

fn load_vert(idx: u32) -> Vertex {
    let b = idx * VSTRIDE;
    return Vertex(
        vec3f(vertices[b],   vertices[b+1u], vertices[b+2u]),
        vec3f(vertices[b+3u],vertices[b+4u], vertices[b+5u]),
        vec4f(vertices[b+6u],vertices[b+7u], vertices[b+8u], vertices[b+9u]),
    );
}

struct Hit { hit: bool, t: f32, norm: vec3f, gnorm: vec3f, color: vec4f }

fn intersect_tri(ro: vec3f, rd: vec3f, i0: u32, i1: u32, i2: u32, t_max: f32) -> Hit {
    var res: Hit; res.hit = false;
    let v0 = load_vert(i0); let v1 = load_vert(i1); let v2 = load_vert(i2);
    
    let edge1 = v1.pos - v0.pos;
    let edge2 = v2.pos - v0.pos;
    let h = cross(rd, edge2);
    let det = dot(edge1, h);
    
    // If determinant is near zero, ray is parallel to triangle
    if (abs(det) < 1e-12) { return res; }
    
    let inv_det = 1.0 / det;
    let s = ro - v0.pos;
    let u = dot(s, h) * inv_det;
    if (u < -1e-4 || u > 1.0 + 1e-4) { return res; }
    
    let q = cross(s, edge1);
    let v = dot(rd, q) * inv_det;
    if (v < -1e-4 || u + v > 1.0 + 1e-4) { return res; }
    
    let t = dot(edge2, q) * inv_det;
    if (t < 1e-4 || t >= t_max) { return res; }
    
    let w = 1.0 - u - v;
    let gnorm = normalize(cross(edge1, edge2));
    res.hit = true;
    res.t = t;
    res.gnorm = select(gnorm, -gnorm, dot(gnorm, rd) > 0.0);
    
    // Interpolate normal and handle potential zero-length result
    let n = u * v1.norm + v * v2.norm + w * v0.norm;
    let len_sq = dot(n, n);
    if (len_sq > 1e-18) {
        res.norm = n * inverseSqrt(len_sq);
    } else {
        // Fallback to geometric normal if vertex normals are degenerate
        res.norm = res.gnorm;
    }
    
    res.color = u * v1.color + v * v2.color + w * v0.color;
    return res;
}

// ── Robust Ray Offsetting ─────────────────────────────────────────────────────
// Prevents self-intersection by offsetting the ray origin along the normal.
// Scales with the magnitude of the position to maintain precision.
fn offset_ray(p: vec3f, n: vec3f) -> vec3f {
    let int_scale: f32 = 256.0;
    let float_scale: f32 = 1.0 / 65536.0;
    let origin: f32 = 1.0 / 32.0;

    let of_i = vec3i(i32(int_scale * n.x), i32(int_scale * n.y), i32(int_scale * n.z));

    let p_i = vec3f(
        bitcast<f32>(bitcast<i32>(p.x) + select(of_i.x, -of_i.x, p.x < 0.0)),
        bitcast<f32>(bitcast<i32>(p.y) + select(of_i.y, -of_i.y, p.y < 0.0)),
        bitcast<f32>(bitcast<i32>(p.z) + select(of_i.z, -of_i.z, p.z < 0.0))
    );

    return vec3f(
        select(p_i.x, p.x + float_scale * n.x, abs(p.x) < origin),
        select(p_i.y, p.y + float_scale * n.y, abs(p.y) < origin),
        select(p_i.z, p.z + float_scale * n.z, abs(p.z) < origin)
    );
}

// ── AABB slab test ────────────────────────────────────────────────────────────
fn aabb_hit(ro: vec3f, inv_rd: vec3f, node_base: u32, t_max: f32) -> bool {
    let mn = vec3f(bvh_nodes[node_base],    bvh_nodes[node_base+1u], bvh_nodes[node_base+2u]);
    let mx = vec3f(bvh_nodes[node_base+4u], bvh_nodes[node_base+5u], bvh_nodes[node_base+6u]);
    // Use min/max to handle ±Inf from zero ray components correctly
    let t0 = (mn - ro) * inv_rd;
    let t1 = (mx - ro) * inv_rd;
    let tmin = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let tmax = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    // tmax < 0: box behind ray; tmin > tmax: miss; tmin >= t_max: farther than best hit
    return tmax >= -1e-4 && tmax >= tmin && tmin < t_max;
}

// ── BVH traversal ─────────────────────────────────────────────────────────────
// Iterative stackless-style traversal using an explicit u32 stack.
fn scene_hit(ro: vec3f, rd: vec3f) -> Hit {
    var best: Hit; best.hit = false; best.t = 1e30;
    let inv_rd = vec3f(1.0 / rd.x, 1.0 / rd.y, 1.0 / rd.z);

    var stack: array<u32, MAX_STACK>;
    var stack_top: u32 = 0u;
    stack[stack_top] = 0u;
    stack_top += 1u;

    while (stack_top > 0u) {
        stack_top -= 1u;
        let node_idx = stack[stack_top];
        let node_base = node_idx * 8u; // 8 f32 per node

        if (!aabb_hit(ro, inv_rd, node_base, best.t)) { continue; }

        let data1_bits = bitcast<u32>(bvh_nodes[node_base + 7u]);
        if ((data1_bits & LEAF_FLAG) != 0u) {
            // Leaf node — test all triangles
            let first = bitcast<u32>(bvh_nodes[node_base + 3u]);
            let count = data1_bits & ~LEAF_FLAG;
            for (var k = 0u; k < count; k++) {
                let base = (first + k) * 3u;
                let i0 = bvh_tris[base];
                let i1 = bvh_tris[base + 1u];
                let i2 = bvh_tris[base + 2u];
                let h = intersect_tri(ro, rd, i0, i1, i2, best.t);
                if (h.hit) { best = h; }
            }
        } else {
            // Internal node — push both children (right first so left is popped first)
            let left  = bitcast<u32>(bvh_nodes[node_base + 3u]);
            let right = bitcast<u32>(bvh_nodes[node_base + 7u]);
            if (stack_top + 1u < MAX_STACK) {
                stack[stack_top] = right;
                stack_top += 1u;
            }
            if (stack_top < MAX_STACK) {
                stack[stack_top] = left;
                stack_top += 1u;
            }
        }
    }
    return best;
}

// ── Environment ───────────────────────────────────────────────────────────────
const PI: f32 = 3.14159265358979;

fn sample_env(dir: vec3f) -> vec3f {
    // Z-up equirectangular, matching envmap.wgsl
    let u = atan2(dir.y, dir.x) / (2.0 * PI) + 0.5;
    let v = 0.5 - asin(clamp(dir.z, -1.0, 1.0)) / PI;
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

    for (var bounce = 0; bounce < 8; bounce++) {
        let hit = scene_hit(ray_o, ray_d);
        if (!hit.hit) {
            radiance += throughput * sample_env(ray_d);
            break;
        }
        let gnorm = hit.gnorm;
        var n = select(hit.norm, -hit.norm, dot(hit.norm, ray_d) > 0.0);
        // Ensure shading normal is on the same side as geometric normal to prevent black spots
        if (dot(n, gnorm) < 0.0) { n = gnorm; }

        let is_specular = hit.color.a < 0.5;
        if (is_specular) {
            // Perfect mirror: reflect ray, throughput unchanged (no albedo tint)
            ray_o = offset_ray(ray_o + ray_d * hit.t, gnorm);
            ray_d = reflect(ray_d, n);
        } else {
            var albedo = hit.color.rgb;
            
            // Apply mesh texture splats if enabled
            if (meshUniforms.splatsEnabled > 0.5) {
                var splat_color = vec3f(0.0);
                var total_weight = 0.0;
                let num_splats = u32(meshUniforms.numSplats);
                let world_pos = ray_o + ray_d * hit.t;
                for (var i = 0u; i < num_splats; i++) {
                    let s = textureSplats[i];
                    let dist = length(world_pos - s.pos);
                    let w = exp(-(dist * dist) / (s.radius * s.radius));
                    splat_color += s.color * w;
                    total_weight += w;
                }
                let blend_factor = saturate(total_weight);
                if (blend_factor > 0.001) {
                    albedo = mix(albedo, splat_color / total_weight, blend_factor);
                }
            }
            
            throughput *= albedo; // diffuse: albedo (cos/pi and pdf cancel)
            ray_o = offset_ray(ray_o + ray_d * hit.t, gnorm);
            ray_d = cosine_hemisphere(n, &seed);
        }
    }

    let base = pixel_idx * 4u;
    accum[base]      += radiance.r;
    accum[base + 1u] += radiance.g;
    accum[base + 2u] += radiance.b;
    accum[base + 3u] += 1.0;
}
