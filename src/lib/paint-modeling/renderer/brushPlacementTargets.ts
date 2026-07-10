import type { RenderRibbon } from "../types.ts";
import { FLOATS_PER_RIBBON_VERTEX } from "./constants.ts";

export type BrushSurfaceTargetData = {
    count: number,
    infos: Uint32Array,
    vertices: Float32Array,
};

export const createBrushSurfaceTargetData = (
    ribbons: RenderRibbon[],
): BrushSurfaceTargetData | null => {
    const targetInfos: number[] = [];
    const targetVertices: number[] = [];

    for (const ribbon of ribbons) {
        if (ribbonSegmentCount(ribbon) === 0) continue;

        targetInfos.push(
            targetVertices.length / FLOATS_PER_RIBBON_VERTEX,
            ribbon.vertices.length,
            ribbon.closed ? 1 : 0,
            0,
        );

        for (const vertex of ribbon.vertices) {
            targetVertices.push(
                vertex.position[0],
                vertex.position[1],
                vertex.position[2],
                vertex.u,
                vertex.side[0],
                vertex.side[1],
                vertex.side[2],
                0,
            );
        }
    }

    if (targetInfos.length === 0) return null;
    return {
        count: targetInfos.length / 4,
        infos: new Uint32Array(targetInfos),
        vertices: new Float32Array(targetVertices),
    };
};

const ribbonSegmentCount = (ribbon: RenderRibbon): number => {
    if (ribbon.vertices.length < 2) return 0;
    return ribbon.closed ? ribbon.vertices.length : ribbon.vertices.length - 1;
};