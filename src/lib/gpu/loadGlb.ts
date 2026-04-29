// Minimal GLB (binary glTF 2.0) loader. Walks every mesh primitive in the file,
// pulls out POSITION + NORMAL + indices, applies the node transform stack, and
// concatenates everything into a single interleaved [pos, normal, color] vertex
// buffer with a single uint32 index buffer. The result is auto-centered and
// uniformly scaled to fit a unit-radius bounding sphere so the existing camera
// (which orbits at radius ~3 around the origin) frames the model correctly.
//
// Supports: node TRS transforms, pbrMetallicRoughness baseColorFactor.
// Does not support: animations, skinning, texture maps, non-triangle primitives.

import { mat3, mat4, vec3 } from "wgpu-matrix";

export interface MeshData {
    // Interleaved [px, py, pz, nx, ny, nz, r, g, b, a, u, v] per vertex.
    vertices: Float32Array;
    indices: Uint32Array;
    /** True when TEXCOORD_0 was present in at least one primitive. */
    hasUvs: boolean;
}

const GLB_MAGIC = 0x46546c67; // "glTF"
const CHUNK_JSON = 0x4e4f534a; // "JSON"
const CHUNK_BIN = 0x004e4942; // "BIN\0"

const COMP_BYTE = 5120;
const COMP_UBYTE = 5121;
const COMP_SHORT = 5122;
const COMP_USHORT = 5123;
const COMP_UINT = 5125;
const COMP_FLOAT = 5126;

const COMP_SIZE: Record<number, number> = {
    [COMP_BYTE]: 1,
    [COMP_UBYTE]: 1,
    [COMP_SHORT]: 2,
    [COMP_USHORT]: 2,
    [COMP_UINT]: 4,
    [COMP_FLOAT]: 4,
};

const TYPE_NUM_COMPONENTS: Record<string, number> = {
    SCALAR: 1,
    VEC2: 2,
    VEC3: 3,
    VEC4: 4,
    MAT2: 4,
    MAT3: 9,
    MAT4: 16,
};

interface GlbAccessor {
    bufferView: number;
    byteOffset?: number;
    componentType: number;
    count: number;
    type: string;
}

interface GlbBufferView {
    buffer: number;
    byteOffset?: number;
    byteLength: number;
    byteStride?: number;
}

interface GlbPrimitive {
    attributes: Record<string, number>;
    indices?: number;
    mode?: number;
    material?: number;
}

interface GlbMaterial {
    pbrMetallicRoughness?: {
        baseColorFactor?: [number, number, number, number];
    };
}

interface GlbMesh {
    primitives: GlbPrimitive[];
}

interface GlbNode {
    name?: string;
    mesh?: number;
    children?: number[];
    matrix?: number[];
    translation?: [number, number, number];
    rotation?: [number, number, number, number];
    scale?: [number, number, number];
}

interface GlbScene {
    nodes: number[];
}

interface GlbJson {
    scenes?: GlbScene[];
    scene?: number;
    nodes?: GlbNode[];
    meshes?: GlbMesh[];
    accessors?: GlbAccessor[];
    bufferViews?: GlbBufferView[];
    materials?: GlbMaterial[];
}

export async function loadGlb(
    url: string,
    normalize = true,
    materialOverride?: [number, number, number, number],
    meshName?: string,
): Promise<MeshData> {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to fetch ${url}: ${response.status}`);
    }
    const buffer = await response.arrayBuffer();
    return parseGlb(buffer, normalize, materialOverride, meshName);
}

function parseGlb(
    buffer: ArrayBuffer,
    normalize = true,
    materialOverride?: [number, number, number, number],
    meshName?: string,
): MeshData {
    const dv = new DataView(buffer);
    if (dv.getUint32(0, true) !== GLB_MAGIC) {
        throw new Error("Not a GLB: bad magic");
    }

    let json: GlbJson | null = null;
    let bin: ArrayBuffer | null = null;

    let offset = 12;
    while (offset < buffer.byteLength) {
        const chunkLen = dv.getUint32(offset, true);
        const chunkType = dv.getUint32(offset + 4, true);
        const dataStart = offset + 8;
        if (chunkType === CHUNK_JSON) {
            const bytes = new Uint8Array(buffer, dataStart, chunkLen);
            json = JSON.parse(new TextDecoder().decode(bytes));
        } else if (chunkType === CHUNK_BIN) {
            bin = buffer.slice(dataStart, dataStart + chunkLen);
        }
        offset = dataStart + chunkLen;
    }

    if (!json) throw new Error("GLB missing JSON chunk");
    if (!bin) throw new Error("GLB missing BIN chunk");

    const positions: number[] = [];
    const normals: number[] = [];
    const colors: number[] = []; // rgba per vertex
    const uvs: number[] = []; // uv per vertex
    const indices: number[] = [];
    let hasUvs = false;

    const sceneIdx = json.scene ?? 0;
    const scene = json.scenes?.[sceneIdx];
    const rootNodes = scene?.nodes ?? json.nodes?.map((_, i) => i) ?? [];

    const identity = mat4.identity();
    for (const nodeIdx of rootNodes) {
        walkNode(json, bin, nodeIdx, identity, positions, normals, colors, uvs, indices, materialOverride, (found) => { if (found) hasUvs = true; }, meshName);
    }

    if (positions.length === 0) {
        throw new Error("GLB has no mesh geometry");
    }

    const vertCount = positions.length / 3;
    // Stride: pos(3) + normal(3) + color(4) + uv(2) = 12 floats
    const STRIDE = 12;
    const interleaved = new Float32Array(vertCount * STRIDE);
    for (let i = 0; i < vertCount; i++) {
        interleaved[i * STRIDE + 0]  = positions[i * 3 + 0];
        interleaved[i * STRIDE + 1]  = positions[i * 3 + 1];
        interleaved[i * STRIDE + 2]  = positions[i * 3 + 2];
        interleaved[i * STRIDE + 3]  = normals[i * 3 + 0];
        interleaved[i * STRIDE + 4]  = normals[i * 3 + 1];
        interleaved[i * STRIDE + 5]  = normals[i * 3 + 2];
        interleaved[i * STRIDE + 6]  = colors[i * 4 + 0];
        interleaved[i * STRIDE + 7]  = colors[i * 4 + 1];
        interleaved[i * STRIDE + 8]  = colors[i * 4 + 2];
        interleaved[i * STRIDE + 9]  = colors[i * 4 + 3];
        interleaved[i * STRIDE + 10] = uvs[i * 2 + 0] ?? 0;
        interleaved[i * STRIDE + 11] = uvs[i * 2 + 1] ?? 0;
    }

    if (normalize) {
        centerAndScaleToUnit(interleaved, STRIDE);
    }

    return {
        vertices: interleaved,
        indices: new Uint32Array(indices),
        hasUvs,
    };
}

function walkNode(
    json: GlbJson,
    bin: ArrayBuffer,
    nodeIdx: number,
    parentXform: Float32Array,
    outPositions: number[],
    outNormals: number[],
    outColors: number[],
    outUvs: number[],
    outIndices: number[],
    materialOverride?: [number, number, number, number],
    onHasUvs?: (found: boolean) => void,
    meshName?: string,
) {
    const node = json.nodes![nodeIdx];
    const local = nodeMatrix(node);
    const xform = mat4.mul(parentXform, local);

    // If a name filter is set, only process nodes whose name matches
    const nameMatches = !meshName || node.name === meshName;

    if (nameMatches && node.mesh !== undefined) {
        const mesh = json.meshes![node.mesh];
        for (const prim of mesh.primitives) {
            if ((prim.mode ?? 4) !== 4) continue;

            const posIdx = prim.attributes.POSITION;
            if (posIdx === undefined) continue;
            const positions = readAccessor(json, bin, posIdx) as Float32Array;

            let normals: Float32Array | null = null;
            if (prim.attributes.NORMAL !== undefined) {
                normals = readAccessor(json, bin, prim.attributes.NORMAL) as Float32Array;
            }

            let uvData: Float32Array | null = null;
            if (prim.attributes.TEXCOORD_0 !== undefined) {
                uvData = readAccessor(json, bin, prim.attributes.TEXCOORD_0) as Float32Array;
                onHasUvs?.(true);
            }

            // Read material base color factor (default white).
            // materialOverride takes precedence over the GLB material.
            let matColor: [number, number, number, number] = [1, 1, 1, 1];
            if (materialOverride) {
                matColor = materialOverride;
            } else if (prim.material !== undefined && json.materials) {
                const mat = json.materials[prim.material];
                const bc = mat?.pbrMetallicRoughness?.baseColorFactor;
                if (bc) matColor = bc;
            }

            const baseVertex = outPositions.length / 3;
            const numVerts = positions.length / 3;

            const normalXform = mat3.transpose(mat3.inverse(mat3.fromMat4(xform)));
            for (let i = 0; i < numVerts; i++) {
                const px = positions[i * 3 + 0];
                const py = positions[i * 3 + 1];
                const pz = positions[i * 3 + 2];
                const tp = vec3.transformMat4(vec3.fromValues(px, py, pz), xform);
                outPositions.push(tp[0], tp[1], tp[2]);

                if (normals) {
                    const nx = normals[i * 3 + 0];
                    const ny = normals[i * 3 + 1];
                    const nz = normals[i * 3 + 2];
                    const tn = vec3.transformMat3(vec3.fromValues(nx, ny, nz), normalXform);
                    const n = vec3.normalize(tn);
                    outNormals.push(n[0], n[1], n[2]);
                } else {
                    outNormals.push(0, 0, 0);
                }

                outColors.push(matColor[0], matColor[1], matColor[2], matColor[3]);

                if (uvData) {
                    outUvs.push(uvData[i * 2 + 0], uvData[i * 2 + 1]);
                } else {
                    outUvs.push(0, 0);
                }
            }

            if (prim.indices !== undefined) {
                const ix = readAccessor(json, bin, prim.indices) as
                    | Uint8Array | Uint16Array | Uint32Array;
                for (let i = 0; i < ix.length; i++) {
                    outIndices.push(baseVertex + ix[i]);
                }
            } else {
                for (let i = 0; i < numVerts; i++) outIndices.push(baseVertex + i);
            }

            if (!normals) {
                computeFlatNormals(outPositions, outIndices, baseVertex, numVerts, outNormals);
            }
        }
    }

    if (node.children) {
        for (const childIdx of node.children) {
            walkNode(json, bin, childIdx, xform, outPositions, outNormals, outColors, outUvs, outIndices, materialOverride, onHasUvs, meshName);
        }
    }
}

function nodeMatrix(node: GlbNode): Float32Array {
    if (node.matrix) {
        return new Float32Array(node.matrix);
    }
    const t = node.translation ?? [0, 0, 0];
    const r = node.rotation ?? [0, 0, 0, 1];
    const s = node.scale ?? [1, 1, 1];

    // Compose T * R * S manually (wgpu-matrix doesn't have a single-call helper
    // we can rely on for quaternion rotation in this version).
    const tm = mat4.translation(vec3.fromValues(t[0], t[1], t[2]));
    const rm = quatToMat4(r[0], r[1], r[2], r[3]);
    const sm = mat4.scaling(vec3.fromValues(s[0], s[1], s[2]));
    return mat4.mul(mat4.mul(tm, rm), sm);
}

function quatToMat4(x: number, y: number, z: number, w: number): Float32Array {
    const xx = x * x, yy = y * y, zz = z * z;
    const xy = x * y, xz = x * z, yz = y * z;
    const wx = w * x, wy = w * y, wz = w * z;
    const m = mat4.identity();
    m[0] = 1 - 2 * (yy + zz);
    m[1] = 2 * (xy + wz);
    m[2] = 2 * (xz - wy);
    m[4] = 2 * (xy - wz);
    m[5] = 1 - 2 * (xx + zz);
    m[6] = 2 * (yz + wx);
    m[8] = 2 * (xz + wy);
    m[9] = 2 * (yz - wx);
    m[10] = 1 - 2 * (xx + yy);
    return m;
}

function readAccessor(
    json: GlbJson,
    bin: ArrayBuffer,
    accessorIdx: number,
): Float32Array | Uint32Array | Uint16Array | Uint8Array {
    const acc = json.accessors![accessorIdx];
    const view = json.bufferViews![acc.bufferView];

    const numComp = TYPE_NUM_COMPONENTS[acc.type];
    if (!numComp) throw new Error(`Unsupported accessor type ${acc.type}`);
    const compSize = COMP_SIZE[acc.componentType];
    if (!compSize) throw new Error(`Unsupported component type ${acc.componentType}`);

    const elemSize = numComp * compSize;
    const stride = view.byteStride ?? elemSize;
    const baseOffset = (view.byteOffset ?? 0) + (acc.byteOffset ?? 0);

    // Slice out a tightly-packed copy. Doing this avoids alignment issues with
    // typed-array views (Float32Array needs 4-byte alignment relative to the
    // ArrayBuffer) and trivially handles strided source data.
    const total = acc.count * numComp;
    const dst = makeTyped(acc.componentType, total);
    const srcDv = new DataView(bin);
    for (let i = 0; i < acc.count; i++) {
        const elemOffset = baseOffset + i * stride;
        for (let c = 0; c < numComp; c++) {
            const at = elemOffset + c * compSize;
            const out = i * numComp + c;
            switch (acc.componentType) {
                case COMP_FLOAT: (dst as Float32Array)[out] = srcDv.getFloat32(at, true); break;
                case COMP_UINT: (dst as Uint32Array)[out] = srcDv.getUint32(at, true); break;
                case COMP_USHORT: (dst as Uint16Array)[out] = srcDv.getUint16(at, true); break;
                case COMP_UBYTE: (dst as Uint8Array)[out] = srcDv.getUint8(at); break;
                case COMP_SHORT: (dst as Int16Array)[out] = srcDv.getInt16(at, true); break;
                case COMP_BYTE: (dst as Int8Array)[out] = srcDv.getInt8(at); break;
            }
        }
    }
    return dst as Float32Array | Uint32Array | Uint16Array | Uint8Array;
}

function makeTyped(componentType: number, length: number) {
    switch (componentType) {
        case COMP_FLOAT: return new Float32Array(length);
        case COMP_UINT: return new Uint32Array(length);
        case COMP_USHORT: return new Uint16Array(length);
        case COMP_UBYTE: return new Uint8Array(length);
        case COMP_SHORT: return new Int16Array(length);
        case COMP_BYTE: return new Int8Array(length);
        default: throw new Error(`Unsupported component type ${componentType}`);
    }
}

function computeFlatNormals(
    positions: number[],
    indices: number[],
    baseVertex: number,
    numVerts: number,
    outNormals: number[],
) {
    // Sum face normals into the corresponding vertex slots, then renormalize.
    // This only touches the vertex range we just appended for this primitive.
    for (let i = 0; i < numVerts; i++) {
        outNormals[(baseVertex + i) * 3 + 0] = 0;
        outNormals[(baseVertex + i) * 3 + 1] = 0;
        outNormals[(baseVertex + i) * 3 + 2] = 0;
    }
    // Iterate triangles that reference this vertex range.
    for (let t = 0; t < indices.length; t += 3) {
        const a = indices[t];
        const b = indices[t + 1];
        const c = indices[t + 2];
        if (a < baseVertex || a >= baseVertex + numVerts) continue;
        const ax = positions[a * 3 + 0], ay = positions[a * 3 + 1], az = positions[a * 3 + 2];
        const bx = positions[b * 3 + 0], by = positions[b * 3 + 1], bz = positions[b * 3 + 2];
        const cx = positions[c * 3 + 0], cy = positions[c * 3 + 1], cz = positions[c * 3 + 2];
        const ex1 = bx - ax, ey1 = by - ay, ez1 = bz - az;
        const ex2 = cx - ax, ey2 = cy - ay, ez2 = cz - az;
        const nx = ey1 * ez2 - ez1 * ey2;
        const ny = ez1 * ex2 - ex1 * ez2;
        const nz = ex1 * ey2 - ey1 * ex2;
        outNormals[a * 3 + 0] += nx; outNormals[a * 3 + 1] += ny; outNormals[a * 3 + 2] += nz;
        outNormals[b * 3 + 0] += nx; outNormals[b * 3 + 1] += ny; outNormals[b * 3 + 2] += nz;
        outNormals[c * 3 + 0] += nx; outNormals[c * 3 + 1] += ny; outNormals[c * 3 + 2] += nz;
    }
    for (let i = 0; i < numVerts; i++) {
        const k = (baseVertex + i) * 3;
        const nx = outNormals[k], ny = outNormals[k + 1], nz = outNormals[k + 2];
        const len = Math.hypot(nx, ny, nz);
        if (len > 1e-8) {
            outNormals[k + 0] = nx / len;
            outNormals[k + 1] = ny / len;
            outNormals[k + 2] = nz / len;
        } else {
            outNormals[k + 0] = 0;
            outNormals[k + 1] = 1;
            outNormals[k + 2] = 0;
        }
    }
}

function centerAndScaleToUnit(interleaved: Float32Array, stride: number) {
    const numVerts = interleaved.length / stride;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < numVerts; i++) {
        const x = interleaved[i * stride + 0];
        const y = interleaved[i * stride + 1];
        const z = interleaved[i * stride + 2];
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
        if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    const cx = (minX + maxX) * 0.5;
    const cy = (minY + maxY) * 0.5;
    const cz = (minZ + maxZ) * 0.5;
    const halfExtent = Math.max(maxX - minX, maxY - minY, maxZ - minZ) * 0.5;
    if (halfExtent <= 1e-8) return;
    const scale = 1 / halfExtent;
    for (let i = 0; i < numVerts; i++) {
        interleaved[i * stride + 0] = (interleaved[i * stride + 0] - cx) * scale;
        interleaved[i * stride + 1] = (interleaved[i * stride + 1] - cy) * scale;
        interleaved[i * stride + 2] = (interleaved[i * stride + 2] - cz) * scale;
        // Normals are unaffected by translation; uniform scale doesn't change
        // them either, so leave the normal slots alone.
    }
}
