import type { ChartProjectionMode, PaintView, Vec2 } from "../types.ts";
import { MIN_DEPTH } from "./constants.ts";
import { depthForProjectionAtPoint } from "./projection.ts";
import { distance2d } from "./strokeSampling.ts";
import { lerp } from "./vectorMath.ts";

export type SnapCarryDepth = {
    rayDepth: number,
};

export function snapCarryDepthAtPoint(rayDistance: number): SnapCarryDepth {
    return {
        rayDepth: Math.max(MIN_DEPTH, rayDistance),
    };
}

export function carryStrokeDepths(
    directDepths: Array<SnapCarryDepth | null>,
    points: Vec2[],
): Array<SnapCarryDepth | null> {
    const carriedDepths = new Array<SnapCarryDepth | null>(directDepths.length).fill(null);
    let previousHitIndex: number | null = null;

    for (let index = 0; index < directDepths.length; index++) {
        const directDepth = directDepths[index];
        if (!directDepth) continue;

        if (previousHitIndex === null) {
            for (let fillIndex = 0; fillIndex <= index; fillIndex++) {
                carriedDepths[fillIndex] = directDepth;
            }
        } else {
            const previousDepth = directDepths[previousHitIndex]!;
            for (let fillIndex = previousHitIndex + 1; fillIndex < index; fillIndex++) {
                carriedDepths[fillIndex] = mixSnapCarryDepths(
                    previousDepth,
                    directDepth,
                    exitEntryDepthMix(points[fillIndex], points[previousHitIndex], points[index]),
                );
            }
            carriedDepths[index] = directDepth;
        }

        previousHitIndex = index;
    }

    if (previousHitIndex !== null) {
        const finalDepth = directDepths[previousHitIndex]!;
        for (let index = previousHitIndex; index < directDepths.length; index++) {
            carriedDepths[index] ??= finalDepth;
        }
    }

    return carriedDepths;
}

export function depthForCarriedSnapAtPoint(
    view: PaintView,
    point: Vec2,
    depth: SnapCarryDepth | null,
    projectionMode: ChartProjectionMode,
): number | null {
    if (!depth) return null;
    // Carry ray distance, then convert at the fallback point. Reusing the hit point's
    // view-plane scalar changes the actual ray depth as the cursor moves.
    return depthForProjectionAtPoint(view, point, depth.rayDepth, projectionMode);
}

function mixSnapCarryDepths(a: SnapCarryDepth, b: SnapCarryDepth, t: number): SnapCarryDepth {
    return {
        rayDepth: lerp(a.rayDepth, b.rayDepth, t),
    };
}

function exitEntryDepthMix(point: Vec2, exitPoint: Vec2, entryPoint: Vec2): number {
    // extended distance/fac function;
    //  1. 0 at the exit point
    //  2. 1 at the entry point
    //  3. naturally tends to 0.5 when far away from both
    // https://www.desmos.com/calculator/fndo5xyztv
    // https://www.desmos.com/3d/ksz46dpbr8
    const exitDistance = distance2d(point, exitPoint);
    const entryDistance = distance2d(point, entryPoint);
    const denominator = exitDistance + entryDistance;
    if (denominator <= 1e-8) return 0.5;
    // Equivalent to 1 - 1 / (1 + exitDistance / entryDistance), without a zero-entry divide.
    return exitDistance / denominator;
}
