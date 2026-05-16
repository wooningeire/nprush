struct Uniforms {
    viewMat: mat4x4f,
    viewInvMat: mat4x4f,
    resolution: vec2f,
    _pad0: vec2f,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    // Full-screen triangle
    const fullscreen_triangle_vertices = array<vec2f, 3>(
        vec2f(-1, -1),
        vec2f(-1, 3),
        vec2f(3, -1),
    );

    let vertex = fullscreen_triangle_vertices[vertex_index];
    var out: VertexOut;
    out.pos = vec4f(vertex, 0.0, 1.0);
    out.uv = vertex;
    return out;
}

@fragment
fn fs(inp: VertexOut) -> @location(0) vec4f {
    let aspect = u.resolution.x / u.resolution.y;

    // NDC → orthographic ray in view-space
    // Sphere radius = 0.85 of the shortest axis so it doesn't clip the edges
    const sphereRadius = 0.5;
    var rayOriginView = vec3f(inp.uv.x * aspect, inp.uv.y, 2.0) / sphereRadius;

    // Transform to world-space
    let rayOriginWorld = (u.viewInvMat * vec4f(rayOriginView, 1.0)).xyz;
    let rayDirWorld = normalize((u.viewInvMat * vec4f(0.0, 0.0, -1.0, 0.0)).xyz);

    // Sphere intersection: unit sphere at origin
    let oc = rayOriginWorld;
    let b = dot(oc, rayDirWorld);
    let c = dot(oc, oc) - 1.0;
    let disc = b * b - c;

    if (disc < 0.0) {
        // Background — subtle dark gradient
        let bg = mix(
            vec3f(0.06, 0.06, 0.08),
            vec3f(0.03, 0.03, 0.04),
            (inp.uv.y + 1.0) * 0.5,
        );
        return vec4f(bg, 1.0);
    }

    let t = -b - sqrt(disc);
    let hitPos = rayOriginWorld + t * rayDirWorld;
    let normal = normalize(hitPos); // Unit sphere → normal = position



    let lightDirWorld = vec3f(0, 0, 1);

    // Fill light from opposite side (fixed in world)
    let fillDir = normalize(vec3f(-0.3, 0.2, -0.5));

    let viewDir = -rayDirWorld;

    // Diffuse
    let ndotlKey = max(dot(normal, lightDirWorld), 0.0);
    let ndotlFill = max(dot(normal, fillDir), 0.0);

    // Blinn-Phong specular
    let halfVec = normalize(lightDirWorld + viewDir);
    let spec = pow(max(dot(normal, halfVec), 0.0), 64.0);

    // Fresnel rim
    let fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0);

    // Base color — warm clay-like material
    let baseColor = vec3f(0.72, 0.55, 0.45);

    let ambient = vec3f(0.08, 0.08, 0.12);
    let keyColor = vec3f(1.0, 0.95, 0.88);
    let fillColor = vec3f(0.25, 0.35, 0.55);
    let rimColor = vec3f(0.3, 0.4, 0.6);

    var color = ambient * baseColor
              + ndotlKey * keyColor * baseColor
              + ndotlFill * fillColor * baseColor * 0.4
              + spec * keyColor * 0.45
              + fresnel * rimColor * 0.25;

    // Tone-map (simple Reinhard)
    color = color / (color + vec3f(1.0));

    // Gamma
    color = pow(color, vec3f(1.0 / 2.2));

    return vec4f(color, 1.0);
}
