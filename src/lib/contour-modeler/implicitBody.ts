import type { FitSample, FitView, ImplicitBodyParams, Vec2, Vec3 } from "./types.ts";

export const DEFAULT_IMPLICIT_BODY_PARAMS: ImplicitBodyParams = {
    center: [0, 0, 0],
    axisX: [1, 0, 0],
    axisY: [0, 1, 0],
    axisZ: [0, 0, 1],
    height: 1,
    radiusBottom: 0.42,
    radiusTop: 0.16,
    bulge: 0.04,
    ovalX: 1,
    ovalZ: 1,
    boxiness: 0,
};

export function cloneImplicitBodyParams(params: ImplicitBodyParams): ImplicitBodyParams {
    return {
        center: [...params.center] as Vec3,
        axisX: [...(params.axisX ?? [1, 0, 0])] as Vec3,
        axisY: [...(params.axisY ?? [0, 1, 0])] as Vec3,
        axisZ: [...(params.axisZ ?? [0, 0, 1])] as Vec3,
        height: params.height,
        radiusBottom: params.radiusBottom,
        radiusTop: params.radiusTop,
        bulge: params.bulge,
        ovalX: params.ovalX,
        ovalZ: params.ovalZ,
        boxiness: params.boxiness ?? 0,
    };
}

export function sanitizeImplicitBodyParams(params: ImplicitBodyParams): ImplicitBodyParams {
    const axes = bodyAxes(params);
    return {
        center: [
            clamp(params.center[0], -1.5, 1.5),
            clamp(params.center[1], -1.5, 1.5),
            clamp(params.center[2], -1.5, 1.5),
        ],
        axisX: axes.x,
        axisY: axes.y,
        axisZ: axes.z,
        height: clamp(params.height, 0.12, 2.5),
        radiusBottom: clamp(params.radiusBottom, 0.03, 1.5),
        radiusTop: clamp(params.radiusTop, 0.015, 1.5),
        bulge: clamp(params.bulge, 0, 0.7),
        ovalX: clamp(params.ovalX, 0.45, 2.25),
        ovalZ: clamp(params.ovalZ, 0.45, 2.25),
        boxiness: clamp(params.boxiness ?? 0, 0, 1),
    };
}

export function implicitRadiusAt(params: ImplicitBodyParams, t: number): number {
    const tt = clamp(t, 0, 1);
    const taper = params.radiusBottom + (params.radiusTop - params.radiusBottom) * tt;
    const bulge = params.bulge * Math.sin(Math.PI * tt);
    return Math.max(0.01, taper + bulge);
}

export function implicitBodyPoint(params: ImplicitBodyParams, t: number, theta: number): Vec3 {
    const radius = implicitRadiusAt(params, t);
    const axes = bodyAxes(params);
    const p = crossSectionExponent(params);
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    const localX = signedPow(c, 2 / p) * radius * params.ovalX;
    const localY = (t - 0.5) * params.height;
    const localZ = signedPow(s, 2 / p) * radius * params.ovalZ;
    return [
        params.center[0] + axes.x[0] * localX + axes.y[0] * localY + axes.z[0] * localZ,
        params.center[1] + axes.x[1] * localX + axes.y[1] * localY + axes.z[1] * localZ,
        params.center[2] + axes.x[2] * localX + axes.y[2] * localY + axes.z[2] * localZ,
    ];
}

export function implicitBodySdf(params: ImplicitBodyParams, p: Vec3): number {
    const axes = bodyAxes(params);
    const dx = p[0] - params.center[0];
    const dy = p[1] - params.center[1];
    const dz = p[2] - params.center[2];
    const localX = dot3([dx, dy, dz], axes.x);
    const localY = dot3([dx, dy, dz], axes.y);
    const localZ = dot3([dx, dy, dz], axes.z);
    const halfHeight = params.height * 0.5;
    const t = clamp((localY + halfHeight) / params.height, 0, 1);
    const radius = implicitRadiusAt(params, t);
    const rx = Math.max(0.005, radius * params.ovalX);
    const rz = Math.max(0.005, radius * params.ovalZ);
    const pExp = crossSectionExponent(params);
    const radialNorm = Math.pow(
        Math.pow(Math.abs(localX / rx), pExp) + Math.pow(Math.abs(localZ / rz), pExp),
        1 / pExp,
    );
    const radial = (radialNorm - 1) * Math.min(rx, rz);
    const cap = Math.abs(localY) - halfHeight;

    if (cap <= 0) return radial;
    if (radial <= 0) return cap;
    return Math.hypot(radial, cap);
}

export function implicitBodyNormal(params: ImplicitBodyParams, p: Vec3): Vec3 {
    const eps = 0.002;
    const dx = implicitBodySdf(params, [p[0] + eps, p[1], p[2]]) - implicitBodySdf(params, [p[0] - eps, p[1], p[2]]);
    const dy = implicitBodySdf(params, [p[0], p[1] + eps, p[2]]) - implicitBodySdf(params, [p[0], p[1] - eps, p[2]]);
    const dz = implicitBodySdf(params, [p[0], p[1], p[2] + eps]) - implicitBodySdf(params, [p[0], p[1], p[2] - eps]);
    const len = Math.hypot(dx, dy, dz);
    if (len <= 1e-8) return [0, 1, 0];
    return [dx / len, dy / len, dz / len];
}

export function projectPoint(viewProjMat: number[] | Float32Array, p: Vec3): Vec2 | null {
    const x = p[0];
    const y = p[1];
    const z = p[2];
    const w = 1;
    const clipX = viewProjMat[0] * x + viewProjMat[4] * y + viewProjMat[8] * z + viewProjMat[12] * w;
    const clipY = viewProjMat[1] * x + viewProjMat[5] * y + viewProjMat[9] * z + viewProjMat[13] * w;
    const clipW = viewProjMat[3] * x + viewProjMat[7] * y + viewProjMat[11] * z + viewProjMat[15] * w;
    if (!Number.isFinite(clipW) || Math.abs(clipW) <= 1e-6) return null;
    return { x: clipX / clipW, y: clipY / clipW };
}

export function evaluateImplicitBodyLoss(
    params: ImplicitBodyParams,
    samples: FitSample[],
    views: FitView[],
): number {
    if (samples.length === 0) return 0;

    let weightedLoss = 0;
    let totalWeight = 0;

    for (const sample of samples) {
        const view = views[sample.viewIndex];
        if (!view) continue;
        const d2 = nearestFeatureDistance2(params, sample, view);
        weightedLoss += d2 * sample.weight;
        totalWeight += sample.weight;
    }

    return totalWeight <= 1e-8 ? 0 : weightedLoss / totalWeight;
}

export function paramsToCandidateArray(params: ImplicitBodyParams): Float32Array {
    return new Float32Array([
        params.center[0], params.center[1], params.center[2], params.height,
        params.radiusBottom, params.radiusTop, params.bulge, params.ovalX,
        params.ovalZ, params.boxiness ?? 0, 0, 0,
    ]);
}

export function candidateArrayToParams(values: ArrayLike<number>): ImplicitBodyParams {
    return sanitizeImplicitBodyParams({
        center: [values[0], values[1], values[2]],
        height: values[3],
        radiusBottom: values[4],
        radiusTop: values[5],
        bulge: values[6],
        ovalX: values[7],
        ovalZ: values[8],
        boxiness: values[9] ?? 0,
    });
}

export function bodyAxes(params: ImplicitBodyParams): { x: Vec3; y: Vec3; z: Vec3 } {
    const y = normalize3(params.axisY ?? [0, 1, 0], [0, 1, 0]);
    const zSeed = normalize3(params.axisZ ?? [0, 0, 1], [0, 0, 1]);
    let x = normalize3(cross3(y, zSeed), [1, 0, 0]);
    let z = normalize3(cross3(x, y), [0, 0, 1]);

    if (params.axisX) {
        const preferredX = normalize3(params.axisX, x);
        z = normalize3(cross3(preferredX, y), z);
        x = normalize3(cross3(y, z), x);
    }

    return { x, y, z };
}

function nearestFeatureDistance2(
    params: ImplicitBodyParams,
    sample: FitSample,
    view: FitView,
): number {
    let best = 1e6;
    const thetaSteps = sample.kind === "edge" ? 30 : 36;

    if (sample.kind === "edge") {
        const ySteps = 12;
        for (let yi = 0; yi <= ySteps; yi++) {
            const t = yi / ySteps;
            for (let ti = 0; ti < thetaSteps; ti++) {
                const d2 = projectedBodyPointDistance2(params, view, sample.point, t, ti / thetaSteps * Math.PI * 2);
                if (d2 < best) best = d2;
            }
        }
    } else {
        const rings = [0.18, 0.38, 0.62, 0.82];
        for (const t of rings) {
            for (let ti = 0; ti < thetaSteps; ti++) {
                const d2 = projectedBodyPointDistance2(params, view, sample.point, t, ti / thetaSteps * Math.PI * 2);
                if (d2 < best) best = d2;
            }
        }
    }

    return best;
}

function projectedBodyPointDistance2(
    params: ImplicitBodyParams,
    view: FitView,
    target: Vec2,
    t: number,
    theta: number,
): number {
    const projected = projectPoint(view.viewProjMat, implicitBodyPoint(params, t, theta));
    if (!projected) return 1e6;
    const dx = projected.x - target.x;
    const dy = projected.y - target.y;
    return dx * dx + dy * dy;
}

function clamp(v: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, v));
}

function dot3(a: Vec3, b: Vec3): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross3(a: Vec3, b: Vec3): Vec3 {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
}

function normalize3(v: Vec3, fallback: Vec3): Vec3 {
    const len = Math.hypot(v[0], v[1], v[2]);
    if (!Number.isFinite(len) || len <= 1e-8) return [...fallback] as Vec3;
    return [v[0] / len, v[1] / len, v[2] / len];
}

function crossSectionExponent(params: ImplicitBodyParams): number {
    return 2 + (params.boxiness ?? 0) * 10;
}

function signedPow(v: number, exponent: number): number {
    return Math.sign(v) * Math.pow(Math.abs(v), exponent);
}
