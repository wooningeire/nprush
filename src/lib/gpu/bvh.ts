// CPU-side SAH BVH builder.
// Produces a flat buffer of BvhNode structs for upload to the GPU.
//
// Node layout (8 x 4 bytes = 32 bytes):
//   bytes  0-11: min_x, min_y, min_z  (f32)
//   bytes 12-15: data0                (u32) — left child idx (internal) or first tri idx (leaf)
//   bytes 16-27: max_x, max_y, max_z  (f32)
//   bytes 28-31: data1                (u32) — right child idx (internal) or (count|LEAF_FLAG) (leaf)
//
// The GPU shader reads this as array<f32> and uses bitcast<u32> for data0/data1.
// Storing u32 values via DataView.setUint32 ensures the bit pattern is preserved
// when the GPU does bitcast<u32>(f32_array[slot]).

export const LEAF_FLAG = 0x80000000 >>> 0;
const MAX_LEAF_TRIS = 4;
const SAH_BINS = 8;

export interface BvhResult {
    // Raw bytes: 32 bytes per node, mixed f32/u32 fields
    nodes: ArrayBuffer;
    // Reordered triangle indices (3 u32 per triangle)
    triIndices: Uint32Array;
}

interface AABB {
    min: [number, number, number];
    max: [number, number, number];
}

function aabbUnion(a: AABB, b: AABB): AABB {
    return {
        min: [Math.min(a.min[0], b.min[0]), Math.min(a.min[1], b.min[1]), Math.min(a.min[2], b.min[2])],
        max: [Math.max(a.max[0], b.max[0]), Math.max(a.max[1], b.max[1]), Math.max(a.max[2], b.max[2])],
    };
}

function aabbSurfaceArea(a: AABB): number {
    const dx = a.max[0] - a.min[0];
    const dy = a.max[1] - a.min[1];
    const dz = a.max[2] - a.min[2];
    return 2 * (dx * dy + dy * dz + dz * dx);
}

function aabbCentroid(a: AABB): [number, number, number] {
    return [
        (a.min[0] + a.max[0]) * 0.5,
        (a.min[1] + a.max[1]) * 0.5,
        (a.min[2] + a.max[2]) * 0.5,
    ];
}

function emptyAABB(): AABB {
    return { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
}

function triAABB(verts: Float32Array, i0: number, i1: number, i2: number): AABB {
    const S = 10;
    const ax = verts[i0*S], ay = verts[i0*S+1], az = verts[i0*S+2];
    const bx = verts[i1*S], by = verts[i1*S+1], bz = verts[i1*S+2];
    const cx = verts[i2*S], cy = verts[i2*S+1], cz = verts[i2*S+2];
    return {
        min: [Math.min(ax,bx,cx), Math.min(ay,by,cy), Math.min(az,bz,cz)],
        max: [Math.max(ax,bx,cx), Math.max(ay,by,cy), Math.max(az,bz,cz)],
    };
}

interface BuildTri {
    i0: number; i1: number; i2: number;
    aabb: AABB;
    centroid: [number, number, number];
}

// ── Node storage using DataView for correct f32/u32 interleaving ──────────────
// Each node: 8 slots × 4 bytes = 32 bytes
// Slots: [f32 minX, f32 minY, f32 minZ, u32 data0, f32 maxX, f32 maxY, f32 maxZ, u32 data1]
const NODE_BYTES = 32;
let nodeBuf: ArrayBuffer = new ArrayBuffer(0);
let nodeDv: DataView = new DataView(nodeBuf);
let nodeCount = 0;

function growNodes(needed: number) {
    if (needed * NODE_BYTES <= nodeBuf.byteLength) return;
    const newSize = Math.max(needed * NODE_BYTES, nodeBuf.byteLength * 2, 1024);
    const newBuf = new ArrayBuffer(newSize);
    new Uint8Array(newBuf).set(new Uint8Array(nodeBuf));
    nodeBuf = newBuf;
    nodeDv = new DataView(nodeBuf);
}

function allocNode(): number {
    growNodes(nodeCount + 1);
    return nodeCount++;
}

function writeNodeBounds(idx: number, mn: [number,number,number], mx: [number,number,number]) {
    const off = idx * NODE_BYTES;
    nodeDv.setFloat32(off +  0, mn[0], true);
    nodeDv.setFloat32(off +  4, mn[1], true);
    nodeDv.setFloat32(off +  8, mn[2], true);
    nodeDv.setFloat32(off + 16, mx[0], true);
    nodeDv.setFloat32(off + 20, mx[1], true);
    nodeDv.setFloat32(off + 24, mx[2], true);
}

function writeNodeData(idx: number, data0: number, data1: number) {
    const off = idx * NODE_BYTES;
    nodeDv.setUint32(off + 12, data0 >>> 0, true);
    nodeDv.setUint32(off + 28, data1 >>> 0, true);
}

// ── Triangle output ───────────────────────────────────────────────────────────
let outTris: number[] = [];

// ── Recursive builder ─────────────────────────────────────────────────────────
function buildNode(tris: BuildTri[]): number {
    let bounds = emptyAABB();
    let centBounds = emptyAABB();
    for (const t of tris) {
        bounds = aabbUnion(bounds, t.aabb);
        centBounds = aabbUnion(centBounds, { min: t.centroid, max: t.centroid });
    }

    const nodeIdx = allocNode();
    writeNodeBounds(nodeIdx, bounds.min, bounds.max);

    if (tris.length <= MAX_LEAF_TRIS) {
        const start = outTris.length / 3;
        for (const t of tris) outTris.push(t.i0, t.i1, t.i2);
        writeNodeData(nodeIdx, start, (tris.length | LEAF_FLAG) >>> 0);
        return nodeIdx;
    }

    // SAH split
    let bestCost = Infinity;
    let bestAxis = 0;
    let bestSplit = 0;
    const parentSA = aabbSurfaceArea(bounds);

    for (let axis = 0; axis < 3; axis++) {
        const cMin = centBounds.min[axis];
        const cMax = centBounds.max[axis];
        if (cMax - cMin < 1e-8) continue;

        const binBounds: AABB[] = Array.from({ length: SAH_BINS }, emptyAABB);
        const binCount = new Int32Array(SAH_BINS);
        for (const t of tris) {
            let b = Math.floor(((t.centroid[axis] - cMin) / (cMax - cMin)) * SAH_BINS);
            b = Math.min(b, SAH_BINS - 1);
            binBounds[b] = aabbUnion(binBounds[b], t.aabb);
            binCount[b]++;
        }

        const leftBounds: AABB[] = new Array(SAH_BINS - 1);
        const leftCounts = new Int32Array(SAH_BINS - 1);
        let lb = emptyAABB(), lc = 0;
        for (let i = 0; i < SAH_BINS - 1; i++) {
            lb = aabbUnion(lb, binBounds[i]);
            lc += binCount[i];
            leftBounds[i] = lb;
            leftCounts[i] = lc;
        }

        let rb = emptyAABB(), rc = 0;
        for (let i = SAH_BINS - 2; i >= 0; i--) {
            rb = aabbUnion(rb, binBounds[i + 1]);
            rc += binCount[i + 1];
            const cost = (aabbSurfaceArea(leftBounds[i]) * leftCounts[i] +
                          aabbSurfaceArea(rb) * rc) / parentSA;
            if (cost < bestCost) { bestCost = cost; bestAxis = axis; bestSplit = i; }
        }
    }

    const cMin = centBounds.min[bestAxis];
    const cMax = centBounds.max[bestAxis];
    let left: BuildTri[], right: BuildTri[];

    if (cMax - cMin < 1e-8 || bestCost >= tris.length) {
        const mid = Math.floor(tris.length / 2);
        left = tris.slice(0, mid);
        right = tris.slice(mid);
    } else {
        left = []; right = [];
        for (const t of tris) {
            let b = Math.floor(((t.centroid[bestAxis] - cMin) / (cMax - cMin)) * SAH_BINS);
            b = Math.min(b, SAH_BINS - 1);
            (b <= bestSplit ? left : right).push(t);
        }
        if (left.length === 0 || right.length === 0) {
            const mid = Math.floor(tris.length / 2);
            left = tris.slice(0, mid);
            right = tris.slice(mid);
        }
    }

    const leftIdx  = buildNode(left);
    const rightIdx = buildNode(right);
    writeNodeData(nodeIdx, leftIdx, rightIdx);
    return nodeIdx;
}

export function buildBvh(verts: Float32Array, indices: Uint32Array): BvhResult {
    nodeCount = 0;
    outTris = [];

    const numTris = indices.length / 3;
    const tris: BuildTri[] = [];
    for (let ti = 0; ti < numTris; ti++) {
        const i0 = indices[ti * 3];
        const i1 = indices[ti * 3 + 1];
        const i2 = indices[ti * 3 + 2];
        const aabb = triAABB(verts, i0, i1, i2);
        tris.push({ i0, i1, i2, aabb, centroid: aabbCentroid(aabb) });
    }

    buildNode(tris);

    // Return only the used portion of the node buffer
    return {
        nodes: nodeBuf.slice(0, nodeCount * NODE_BYTES),
        triIndices: new Uint32Array(outTris),
    };
}
