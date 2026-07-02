import {
    DEFORMATION_SUPPORT_OFFSET,
    RIBBON_BASE_COLUMNS,
    RIBBON_EVALUATED_MIN_COLUMNS,
    RIBBON_EVALUATED_MIN_ROWS,
} from "./constants.ts";
import {
    defaultDepthForPaintView,
    makeViewRay,
    viewDepthForWorldPoint,
    viewPointToWorldAtDepth,
} from "./projection.ts";
import { distance2d, samplePaintStrokeSpline } from "./strokeSampling.ts";
import {
    add3,
    clamp,
    cross3,
    distance3,
    dot3,
    lerp,
    scale3,
    sub3,
} from "./vectorMath.ts";
import type {
    DeformationLine,
    PaintObject,
    PaintRibbonFace,
    PaintRibbonMesh,
    PaintRibbonVertex,
    PaintStroke,
    PaintView,
    RibbonUv,
    StrokeSurfaceHit,
    Vec2,
    Vec3,
} from "../types.ts";

export type RibbonBuildResult = {
    centerline: Vec3[],
    mesh: PaintRibbonMesh,
};

type Ray = {
    origin: Vec3,
    direction: Vec3,
};

type TriangleHit = {
    distance: number,
    barycentric: [number, number, number],
};

export const buildRibbonStrokeGeometry = (
    sourcePoints: Vec2[],
    sourceView: PaintView,
    width: number,
    columns: number[] = [...RIBBON_BASE_COLUMNS],
): RibbonBuildResult | null => {
    const closed = isClosedSourceStroke(sourcePoints);
    const points = closed ? sourcePoints.slice(0, -1) : sourcePoints;
    if (points.length < 2) return null;

    const depth = defaultDepthForPaintView(sourceView);
    const centerline = points
        .map(point => viewPointToWorldAtDepth(sourceView, point, depth))
        .filter((point): point is Vec3 => point !== null);
    if (centerline.length < 2) return null;

    const sortedColumns = normalizeColumns(columns);
    const uValues = normalizedPolylineU(centerline, closed);
    const vertices: PaintRibbonVertex[] = [];

    for (let row = 0; row < centerline.length; row++) {
        const sourcePoint = points[row];
        const sideOffset = ribbonSideOffsetAt(points, row, sourceView, width, closed);
        const center = centerline[row];
        const side = sideVectorAt(sourceView, sourcePoint, depth, sideOffset, center);

        for (const v of sortedColumns) {
            vertices.push({
                position: add3(center, scale3(side, v)),
                u: uValues[row],
                v,
            });
        }
    }

    return {
        centerline,
        mesh: {
            rows: centerline.length,
            columns: sortedColumns,
            closed,
            vertices,
            faces: buildRibbonFaces(centerline.length, sortedColumns.length, closed),
        },
    };
};

export const buildRibbonGeometryFromDraft = (
    draftStroke: Vec2[],
    sourceView: PaintView,
    width: number,
): RibbonBuildResult | null => buildRibbonStrokeGeometry(
    samplePaintStrokeSpline(draftStroke),
    sourceView,
    width,
);

export const addDeformationLineToStroke = (
    stroke: PaintStroke,
    deformationLine: DeformationLine,
): PaintStroke => {
    const supportColumns = deformationSupportColumns(deformationLine.points);
    const mesh = insertRibbonColumns(stroke.mesh, supportColumns);
    return {
        ...stroke,
        mesh,
        centerline: centerlineFromMesh(mesh),
        deformationLines: [
            ...stroke.deformationLines,
            {
                ...deformationLine,
                points: deformationLine.points.map(point => ({ ...point })),
            },
        ],
    };
};

export const evaluatedStrokeMesh = (stroke: PaintStroke): PaintRibbonMesh => {
    const columns = evaluatedRibbonColumns(stroke.mesh.columns);
    const rows = evaluatedRibbonRows(stroke.mesh);
    const vertices: PaintRibbonVertex[] = [];

    for (let row = 0; row < rows; row++) {
        const u = stroke.mesh.closed
            ? row / rows
            : row / Math.max(1, rows - 1);

        for (const v of columns) {
            vertices.push({
                position: strokeWorldPointAtUv(stroke, { u, v }),
                u,
                v,
            });
        }
    }

    return {
        rows,
        columns,
        closed: stroke.mesh.closed,
        vertices,
        faces: buildRibbonFaces(rows, columns.length, stroke.mesh.closed),
    };
};

export const strokeWorldPointAtUv = (stroke: PaintStroke, uv: RibbonUv): Vec3 => {
    return ribbonMeshPointAtUv(stroke.mesh, uv);
};

export const sculptStrokeMesh = (
    stroke: PaintStroke,
    center: RibbonUv,
    delta: Vec3,
    radius = 0.18,
): PaintStroke => {
    const radiusSafe = Math.max(radius, 1e-5);
    const mesh: PaintRibbonMesh = {
        ...stroke.mesh,
        columns: [...stroke.mesh.columns],
        vertices: stroke.mesh.vertices.map(vertex => {
            const du = uDistance(stroke.mesh, vertex.u, center.u);
            const dv = vertex.v - center.v;
            const distance = Math.hypot(du, dv);
            if (distance > radiusSafe) return cloneRibbonVertex(vertex);

            const t = distance / radiusSafe;
            const influence = (1 - t * t) ** 2;
            return {
                ...vertex,
                position: add3(vertex.position, scale3(delta, influence)),
            };
        }),
        faces: stroke.mesh.faces.map(face => [...face] as PaintRibbonFace),
    };

    return {
        ...stroke,
        mesh,
        centerline: centerlineFromMesh(mesh),
    };
};

export const raycastStrokeSurface = (
    objects: PaintObject[],
    strokes: PaintStroke[],
    view: PaintView,
    point: Vec2,
): StrokeSurfaceHit | null => {
    const ray = makeViewRay(view, point);
    if (!ray) return null;

    const objectById = new Map(objects.map(object => [object.id, object]));
    let best: StrokeSurfaceHit | null = null;
    let bestDistance = Number.POSITIVE_INFINITY;

    for (const stroke of strokes) {
        const object = objectById.get(stroke.objectId);
        if (!object?.visible || object.locked) continue;

        const surface = evaluatedStrokeMesh(stroke);
        for (let faceIndex = 0; faceIndex < surface.faces.length; faceIndex++) {
            const face = surface.faces[faceIndex];
            const vertices = face.map(index => surface.vertices[index]);
            const hitA = raycastTriangle(ray, vertices[0].position, vertices[1].position, vertices[2].position);
            const hitB = raycastTriangle(ray, vertices[0].position, vertices[2].position, vertices[3].position);
            const hit = hitA && (!hitB || hitA.distance <= hitB.distance)
                ? { hit: hitA, indices: [0, 1, 2] }
                : hitB
                    ? { hit: hitB, indices: [0, 2, 3] }
                    : null;
            if (!hit || hit.hit.distance >= bestDistance) continue;

            const triVertices = hit.indices.map(index => vertices[index]);
            const world = barycentricVec3(
                triVertices[0].position,
                triVertices[1].position,
                triVertices[2].position,
                hit.hit.barycentric,
            );
            bestDistance = hit.hit.distance;
            best = {
                objectId: stroke.objectId,
                strokeId: stroke.id,
                faceIndex,
                uv: {
                    u: barycentricNumber(triVertices[0].u, triVertices[1].u, triVertices[2].u, hit.hit.barycentric),
                    v: barycentricNumber(triVertices[0].v, triVertices[1].v, triVertices[2].v, hit.hit.barycentric),
                },
                world,
                viewDepth: viewDepthForWorldPoint(view, world),
            };
        }
    }

    return best;
};

export const meshConnectedComponentCount = (mesh: PaintRibbonMesh): number => {
    const adjacency = new Map<number, Set<number>>();
    for (let index = 0; index < mesh.vertices.length; index++) adjacency.set(index, new Set());
    for (const face of mesh.faces) {
        for (let index = 0; index < face.length; index++) {
            const a = face[index];
            const b = face[(index + 1) % face.length];
            adjacency.get(a)?.add(b);
            adjacency.get(b)?.add(a);
        }
    }

    const visited = new Set<number>();
    let count = 0;
    for (let index = 0; index < mesh.vertices.length; index++) {
        if (visited.has(index)) continue;
        count += 1;
        const stack = [index];
        visited.add(index);
        while (stack.length > 0) {
            const current = stack.pop()!;
            for (const next of adjacency.get(current) ?? []) {
                if (visited.has(next)) continue;
                visited.add(next);
                stack.push(next);
            }
        }
    }
    return count;
};

const evaluatedRibbonRows = (mesh: PaintRibbonMesh): number => {
    return Math.max(mesh.rows, RIBBON_EVALUATED_MIN_ROWS);
};

const evaluatedRibbonColumns = (columns: number[]): number[] => {
    const uniformColumns = Array.from({ length: RIBBON_EVALUATED_MIN_COLUMNS }, (_, index) => {
        return -1 + index * 2 / Math.max(1, RIBBON_EVALUATED_MIN_COLUMNS - 1);
    });
    return normalizeColumns([...columns, ...uniformColumns]);
};

const ribbonMeshPointAtUv = (mesh: PaintRibbonMesh, uv: RibbonUv): Vec3 => {
    const row = rowSampleAtU(mesh, uv.u);
    const column = columnSampleAtV(mesh.columns, uv.v);
    const a = lerp3(
        meshVertexAt(mesh, row.lower, column.lower).position,
        meshVertexAt(mesh, row.lower, column.upper).position,
        column.t,
    );
    const b = lerp3(
        meshVertexAt(mesh, row.upper, column.lower).position,
        meshVertexAt(mesh, row.upper, column.upper).position,
        column.t,
    );
    return lerp3(a, b, row.t);
};

type RibbonIndexSample = {
    lower: number,
    upper: number,
    t: number,
};

const rowSampleAtU = (mesh: PaintRibbonMesh, u: number): RibbonIndexSample => {
    if (mesh.rows <= 1) return { lower: 0, upper: 0, t: 0 };

    const target = mesh.closed ? wrapUnit(u) : clamp(u, 0, 1);
    const firstU = rowUAt(mesh, 0);

    for (let row = 0; row < mesh.rows - 1; row++) {
        const lowerU = rowUAt(mesh, row);
        const upperU = rowUAt(mesh, row + 1);
        if (target < lowerU || target > upperU) continue;
        return {
            lower: row,
            upper: row + 1,
            t: safeInverseLerp(lowerU, upperU, target),
        };
    }

    if (!mesh.closed) {
        const lastRow = mesh.rows - 1;
        return target <= firstU
            ? { lower: 0, upper: 0, t: 0 }
            : { lower: lastRow, upper: lastRow, t: 0 };
    }

    const lastRow = mesh.rows - 1;
    const lastU = rowUAt(mesh, lastRow);
    const wrappedTarget = target < firstU ? target + 1 : target;
    return {
        lower: lastRow,
        upper: 0,
        t: safeInverseLerp(lastU, firstU + 1, wrappedTarget),
    };
};

const columnSampleAtV = (columns: number[], v: number): RibbonIndexSample => {
    if (columns.length <= 1) return { lower: 0, upper: 0, t: 0 };

    const target = clamp(v, columns[0], columns.at(-1)!);
    for (let column = 0; column < columns.length - 1; column++) {
        const lower = columns[column];
        const upper = columns[column + 1];
        if (target < lower || target > upper) continue;
        return {
            lower: column,
            upper: column + 1,
            t: safeInverseLerp(lower, upper, target),
        };
    }

    const lastColumn = columns.length - 1;
    return target <= columns[0]
        ? { lower: 0, upper: 0, t: 0 }
        : { lower: lastColumn, upper: lastColumn, t: 0 };
};

const meshVertexAt = (mesh: PaintRibbonMesh, row: number, column: number): PaintRibbonVertex => {
    return mesh.vertices[row * mesh.columns.length + column];
};

const rowUAt = (mesh: PaintRibbonMesh, row: number): number => {
    return meshVertexAt(mesh, row, 0).u;
};

const uDistance = (mesh: PaintRibbonMesh, a: number, b: number): number => {
    return mesh.closed ? uvDelta(a, b) : Math.abs(clamp(a, 0, 1) - clamp(b, 0, 1));
};

const safeInverseLerp = (a: number, b: number, value: number): number => {
    return Math.abs(b - a) <= 1e-8 ? 0 : clamp((value - a) / (b - a), 0, 1);
};

const lerp3 = (a: Vec3, b: Vec3, t: number): Vec3 => [
    lerp(a[0], b[0], t),
    lerp(a[1], b[1], t),
    lerp(a[2], b[2], t),
];

const wrapUnit = (value: number): number => {
    const wrapped = value % 1;
    return wrapped < 0 ? wrapped + 1 : wrapped;
};

const insertRibbonColumns = (mesh: PaintRibbonMesh, columns: number[]): PaintRibbonMesh => {
    const nextColumns = normalizeColumns([...mesh.columns, ...columns]);
    if (nextColumns.length === mesh.columns.length) {
        return {
            ...mesh,
            columns: [...mesh.columns],
            vertices: mesh.vertices.map(cloneRibbonVertex),
            faces: mesh.faces.map(face => [...face] as PaintRibbonFace),
        };
    }

    const vertices: PaintRibbonVertex[] = [];
    for (let row = 0; row < mesh.rows; row++) {
        const rowVertices = rowSlice(mesh, row);
        for (const column of nextColumns) {
            vertices.push(interpolateRowVertex(rowVertices, column));
        }
    }

    return {
        rows: mesh.rows,
        columns: nextColumns,
        closed: mesh.closed,
        vertices,
        faces: buildRibbonFaces(mesh.rows, nextColumns.length, mesh.closed),
    };
};

const deformationSupportColumns = (points: RibbonUv[]): number[] => {
    if (points.length === 0) return [];
    const averageV = clamp(
        points.reduce((sum, point) => sum + point.v, 0) / points.length,
        -1,
        1,
    );
    return [
        averageV - DEFORMATION_SUPPORT_OFFSET,
        averageV,
        averageV + DEFORMATION_SUPPORT_OFFSET,
    ].map(value => clamp(value, -1, 1));
};

const normalizedPolylineU = (points: Vec3[], closed: boolean): number[] => {
    const distances = [0];
    let total = 0;
    for (let index = 1; index < points.length; index++) {
        total += distance3(points[index - 1], points[index]);
        distances.push(total);
    }
    if (closed && points.length > 2) {
        total += distance3(points.at(-1)!, points[0]);
    }
    if (total <= 1e-8) {
        return points.map((_, index) => points.length <= 1 ? 0 : index / (points.length - 1));
    }
    return distances.map(distance => distance / total);
};

const ribbonSideOffsetAt = (
    points: Vec2[],
    index: number,
    view: PaintView,
    width: number,
    closed: boolean,
): Vec2 => {
    const current = points[index];
    const previous = index > 0
        ? points[index - 1]
        : closed
            ? points.at(-1)!
            : current;
    const next = index < points.length - 1
        ? points[index + 1]
        : closed
            ? points[0]
            : current;
    const dxPx = (next.x - previous.x) * view.width * 0.5;
    const dyPx = (next.y - previous.y) * view.height * 0.5;
    const lengthPx = Math.hypot(dxPx, dyPx);
    if (lengthPx <= 1e-6) return { x: 0, y: Math.max(width, 1) / view.height };

    const halfWidthPx = Math.max(width, 1) * 0.5;
    return {
        x: -dyPx / lengthPx * halfWidthPx * 2 / view.width,
        y: dxPx / lengthPx * halfWidthPx * 2 / view.height,
    };
};

const sideVectorAt = (
    sourceView: PaintView,
    sourcePoint: Vec2,
    depth: number,
    sideOffset: Vec2,
    center: Vec3,
): Vec3 => {
    const sideWorld = viewPointToWorldAtDepth(
        sourceView,
        {
            x: sourcePoint.x + sideOffset.x,
            y: sourcePoint.y + sideOffset.y,
        },
        depth,
    );
    if (!sideWorld) return [0, 0, 0];
    return sub3(sideWorld, center);
};

const buildRibbonFaces = (
    rows: number,
    columns: number,
    closed: boolean,
): PaintRibbonFace[] => {
    const faces: PaintRibbonFace[] = [];
    const rowLimit = closed ? rows : rows - 1;
    for (let row = 0; row < rowLimit; row++) {
        const nextRow = (row + 1) % rows;
        for (let column = 0; column < columns - 1; column++) {
            faces.push([
                row * columns + column,
                nextRow * columns + column,
                nextRow * columns + column + 1,
                row * columns + column + 1,
            ]);
        }
    }
    return faces;
};

const normalizeColumns = (columns: number[]): number[] => {
    return [...new Set(columns.map(value => Number(clamp(value, -1, 1).toFixed(5))))]
        .sort((a, b) => a - b);
};

const centerlineFromMesh = (mesh: PaintRibbonMesh): Vec3[] => {
    const centerColumn = nearestColumnIndex(mesh.columns, 0);
    const centerline: Vec3[] = [];
    for (let row = 0; row < mesh.rows; row++) {
        centerline.push(mesh.vertices[row * mesh.columns.length + centerColumn].position);
    }
    return centerline;
};

const nearestColumnIndex = (columns: number[], target: number): number => {
    let best = 0;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (let index = 0; index < columns.length; index++) {
        const distance = Math.abs(columns[index] - target);
        if (distance < bestDistance) {
            best = index;
            bestDistance = distance;
        }
    }
    return best;
};

const rowSlice = (mesh: PaintRibbonMesh, row: number): PaintRibbonVertex[] => {
    const start = row * mesh.columns.length;
    return mesh.vertices.slice(start, start + mesh.columns.length);
};

const interpolateRowVertex = (
    rowVertices: PaintRibbonVertex[],
    column: number,
): PaintRibbonVertex => {
    const exact = rowVertices.find(vertex => Math.abs(vertex.v - column) <= 1e-5);
    if (exact) return cloneRibbonVertex(exact);

    const rightIndex = rowVertices.findIndex(vertex => vertex.v > column);
    const left = rightIndex <= 0 ? rowVertices[0] : rowVertices[rightIndex - 1];
    const right = rightIndex < 0 ? rowVertices.at(-1)! : rowVertices[rightIndex];
    const t = Math.abs(right.v - left.v) <= 1e-6
        ? 0
        : (column - left.v) / (right.v - left.v);

    return {
        position: [
            lerp(left.position[0], right.position[0], t),
            lerp(left.position[1], right.position[1], t),
            lerp(left.position[2], right.position[2], t),
        ],
        u: lerp(left.u, right.u, t),
        v: column,
    };
};

const isClosedSourceStroke = (points: Vec2[]): boolean => (
    points.length > 3 && distance2d(points[0], points.at(-1)!) <= 0.04
);

const uvDelta = (a: number, b: number): number => {
    const direct = Math.abs(a - b);
    return Math.min(direct, 1 - direct);
};

const raycastTriangle = (ray: Ray, a: Vec3, b: Vec3, c: Vec3): TriangleHit | null => {
    const edge1 = sub3(b, a);
    const edge2 = sub3(c, a);
    const p = cross3(ray.direction, edge2);
    const determinant = dot3(edge1, p);
    if (Math.abs(determinant) <= 1e-8) return null;

    const invDeterminant = 1 / determinant;
    const t = sub3(ray.origin, a);
    const u = dot3(t, p) * invDeterminant;
    if (u < 0 || u > 1) return null;

    const q = cross3(t, edge1);
    const v = dot3(ray.direction, q) * invDeterminant;
    if (v < 0 || u + v > 1) return null;

    const distance = dot3(edge2, q) * invDeterminant;
    if (distance <= 1e-6) return null;

    return {
        distance,
        barycentric: [1 - u - v, u, v],
    };
};

const barycentricVec3 = (
    a: Vec3,
    b: Vec3,
    c: Vec3,
    weights: [number, number, number],
): Vec3 => [
    barycentricNumber(a[0], b[0], c[0], weights),
    barycentricNumber(a[1], b[1], c[1], weights),
    barycentricNumber(a[2], b[2], c[2], weights),
];

const barycentricNumber = (
    a: number,
    b: number,
    c: number,
    weights: [number, number, number],
): number => a * weights[0] + b * weights[1] + c * weights[2];

const cloneRibbonVertex = (vertex: PaintRibbonVertex): PaintRibbonVertex => ({
    ...vertex,
    position: [...vertex.position] as Vec3,
});
