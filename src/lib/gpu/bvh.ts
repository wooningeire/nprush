// CPU-side SAH BVH builder.
// Produces a flat array of BvhNode structs for upload to the GPU.
//
// Node layout (8 x f32 = 32 bytes):
//   [min_x, min_y, min_z, data0,  max_x, max_y, max_z, data1]
//
// Internal node: data0 = left child index, data1 = right child index
// Leaf node:     data0 = first triangle index in reordered tri_indices,
//                data1 = tri_count | LEAF_FLAG (bit 31 set)
//
// Triangle indices are stored in a separate reordered array so the GPU
// can look up triangles by BVH leaf range without indirection overhead.

export const LEAF_FLAG = 0x80000000;
const MAX_LEAF_TRIS = 4; // max triangles per leaf before splitting
const SAH_BINS = 8;      // number of SAH binning buckets

export interface BvhResult {
    // Flat node array: 8 f32 per node
    nodes: Float32Array;
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
    const VSTRIDE = 10;
    const p0 = [verts[i0 * VSTRIDE], verts[i0 * VSTRIDE + 1], verts[i0 * VSTRIDE + 2]] as [number, number, number];
    const p1 = [verts[i1 * VSTRIDE], verts[i1 * VSTRIDE + 1], verts[i1 * VSTRIDE + 2]] as [number, number, number];
    const p2 = [verts[i2 * VSTRIDE], verts[i2 * VSTRIDE + 1], verts[i2 * VSTRIDE + 2]] as [number, number, number];
    return {
        min: [Math.min(p0[0], p1[0], p2[0]), Math.min(p0[1], p1[1], p2[1]), Math.min(p0[2], p1[2], p2[2])],
        max: [Math.max(p0[0], p1[0], p2[0]), Math.max(p0[1], p1[1], p2[1]), Math.max(p0[2], p1[2], p2[2])],
    };
}

interface BuildTri {
    i0: number; i1: number; i2: number;
    aabb: AABB;
    centroid: [number, number, number];
}

// Flat node storage (built incrementally)
const nodeData: number[] = []; // 8 floats per node

function pushNode(
    minX: number, minY: number, minZ: number, data0: number,
    maxX: number, maxY: number, maxZ: number, data1: number,
): number {
    const idx = nodeData.length / 8;
    nodeData.push(minX, minY, minZ, data0, maxX, maxY, maxZ, data1);
    return idx;
}

function setNodeData(nodeIdx: number, data0: number, data1: number) {
    nodeData[nodeIdx * 8 + 3] = data0;
    nodeData[nodeIdx * 8 + 7] = data1;
}

// Reordered triangle output
const outTris: number[] = []; // i0, i1, i2 per tri

function buildNode(tris: BuildTri[]): number {
    // Compute bounds of all triangle AABBs and centroids
    let bounds = emptyAABB();
    let centBounds = emptyAABB();
    for (const t of tris) {
        bounds = aabbUnion(bounds, t.aabb);
        const cb: AABB = { min: t.centroid, max: t.centroid };
        centBounds = aabbUnion(centBounds, cb);
    }

    // Allocate node slot now (fill data later)
    const nodeIdx = pushNode(
        bounds.min[0], bounds.min[1], bounds.min[2], 0,
        bounds.max[0], bounds.max[1], bounds.max[2], 0,
    );

    if (tris.length <= MAX_LEAF_TRIS) {
        // Leaf
        const start = outTris.length / 3;
        for (const t of tris) outTris.push(t.i0, t.i1, t.i2);
        setNodeData(nodeIdx, start, (tris.length | LEAF_FLAG) >>> 0);
        return nodeIdx;
    }

    // Find best SAH split
    let bestCost = Infinity;
    let bestAxis = 0;
    let bestSplit = 0;

    const parentSA = aabbSurfaceArea(bounds);

    for (let axis = 0; axis < 3; axis++) {
        const cMin = centBounds.min[axis];
        const cMax = centBounds.max[axis];
        if (cMax - cMin < 1e-8) continue;

        // Bin triangles
        const binBounds: AABB[] = Array.from({ length: SAH_BINS }, emptyAABB);
        const binCount = new Int32Array(SAH_BINS);

        for (const t of tris) {
            const c = t.centroid[axis];
            let b = Math.floor(((c - cMin) / (cMax - cMin)) * SAH_BINS);
            b = Math.min(b, SAH_BINS - 1);
            binBounds[b] = aabbUnion(binBounds[b], t.aabb);
            binCount[b]++;
        }

        // Sweep left→right and right→left to find best split
        const leftBounds: AABB[] = new Array(SAH_BINS - 1);
        const leftCounts = new Int32Array(SAH_BINS - 1);
        let lb = emptyAABB();
        let lc = 0;
        for (let i = 0; i < SAH_BINS - 1; i++) {
            lb = aabbUnion(lb, binBounds[i]);
            lc += binCount[i];
            leftBounds[i] = lb;
            leftCounts[i] = lc;
        }

        let rb = emptyAABB();
        let rc = 0;
        for (let i = SAH_BINS - 2; i >= 0; i--) {
            rb = aabbUnion(rb, binBounds[i + 1]);
            rc += binCount[i + 1];
            const cost = (aabbSurfaceArea(leftBounds[i]) * leftCounts[i] +
                          aabbSurfaceArea(rb) * rc) / parentSA;
            if (cost < bestCost) {
                bestCost = cost;
                bestAxis = axis;
                bestSplit = i;
            }
        }
    }

    // Partition
    const cMin = centBounds.min[bestAxis];
    const cMax = centBounds.max[bestAxis];
    let left: BuildTri[];
    let right: BuildTri[];

    if (cMax - cMin < 1e-8 || bestCost >= tris.length) {
        // Degenerate: split in half
        const mid = Math.floor(tris.length / 2);
        left = tris.slice(0, mid);
        right = tris.slice(mid);
    } else {
        left = [];
        right = [];
        for (const t of tris) {
            let b = Math.floor(((t.centroid[bestAxis] - cMin) / (cMax - cMin)) * SAH_BINS);
            b = Math.min(b, SAH_BINS - 1);
            if (b <= bestSplit) left.push(t); else right.push(t);
        }
        if (left.length === 0 || right.length === 0) {
            const mid = Math.floor(tris.length / 2);
            left = tris.slice(0, mid);
            right = tris.slice(mid);
        }
    }

    const leftIdx  = buildNode(left);
    const rightIdx = buildNode(right);
    setNodeData(nodeIdx, leftIdx, rightIdx);
    return nodeIdx;
}

export function buildBvh(verts: Float32Array, indices: Uint32Array): BvhResult {
    // Reset global state
    nodeData.length = 0;
    outTris.length = 0;

    const numTris = indices.length / 3;
    const tris: BuildTri[] = [];
    for (let ti = 0; ti < numTris; ti++) {
        const i0 = indices[ti * 3 + 0];
        const i1 = indices[ti * 3 + 1];
        const i2 = indices[ti * 3 + 2];
        const aabb = triAABB(verts, i0, i1, i2);
        tris.push({ i0, i1, i2, aabb, centroid: aabbCentroid(aabb) });
    }

    buildNode(tris);

    return {
        nodes: new Float32Array(nodeData),
        triIndices: new Uint32Array(outTris),
    };
}
