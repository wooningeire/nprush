import { RIBBON_RENDER_COLUMNS } from "../state/constants.ts";
import type { RenderRibbon, Vec4 } from "../types.ts";
import {
    FLOATS_PER_RIBBON_VERTEX,
    RIBBON_UNIFORM_FLOATS,
} from "./constants.ts";

export type RibbonDrawState = {
    vertexBuffer: GPUBuffer,
    uniformBuffer: GPUBuffer,
    bindGroup: GPUBindGroup,
    vertexCount: number,
    rows: number,
    closed: boolean,
    color: Vec4,
    shade: number,
};

export const createRibbonDraws = (
    device: GPUDevice,
    bindGroupLayout: GPUBindGroupLayout,
    ribbons: RenderRibbon[],
    label: string,
): RibbonDrawState[] => {
    const draws: RibbonDrawState[] = [];
    for (let index = 0; index < ribbons.length; index++) {
        const ribbon = ribbons[index];
        const segmentCount = ribbonSegmentCount(ribbon);
        if (segmentCount === 0) continue;

        const vertexBuffer = device.createBuffer({
            label: `${label} ${index} oriented vertices`,
            size: Math.max(
                Float32Array.BYTES_PER_ELEMENT,
                ribbon.vertices.length * FLOATS_PER_RIBBON_VERTEX * Float32Array.BYTES_PER_ELEMENT,
            ),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(vertexBuffer, 0, createRibbonVertexData(ribbon));

        const uniformBuffer = device.createBuffer({
            label: `${label} ${index} uniforms`,
            size: RIBBON_UNIFORM_FLOATS * Float32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const bindGroup = device.createBindGroup({
            label: `${label} ${index} bind group`,
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: vertexBuffer } },
            ],
        });

        draws.push({
            vertexBuffer,
            uniformBuffer,
            bindGroup,
            vertexCount: segmentCount * (RIBBON_RENDER_COLUMNS - 1) * 6,
            rows: ribbon.vertices.length,
            closed: ribbon.closed,
            color: ribbon.color,
            shade: ribbon.shade ?? 0,
        });
    }
    return draws;
};

export const destroyRibbonDraws = (draws: RibbonDrawState[]) => {
    for (const draw of draws) {
        draw.vertexBuffer.destroy();
        draw.uniformBuffer.destroy();
    }
};

export const writeRibbonDrawUniforms = (
    device: GPUDevice,
    draws: RibbonDrawState[],
    viewProjMat: number[] | Float32Array,
    viewMat: number[] | Float32Array,
) => {
    for (const draw of draws) {
        const data = new Float32Array(RIBBON_UNIFORM_FLOATS);
        data.set(viewProjMat, 0);
        data.set(viewMat, 16);
        data.set(draw.color, 32);
        data[36] = draw.rows;
        data[37] = draw.closed ? 1 : 0;
        data[38] = draw.shade;
        data[39] = RIBBON_RENDER_COLUMNS;
        device.queue.writeBuffer(draw.uniformBuffer, 0, data);
    }
};

const createRibbonVertexData = (ribbon: RenderRibbon): Float32Array => {
    const data = new Float32Array(ribbon.vertices.length * FLOATS_PER_RIBBON_VERTEX);
    let offset = 0;
    for (const vertex of ribbon.vertices) {
        data[offset++] = vertex.position[0];
        data[offset++] = vertex.position[1];
        data[offset++] = vertex.position[2];
        data[offset++] = vertex.u;
        data[offset++] = vertex.side[0];
        data[offset++] = vertex.side[1];
        data[offset++] = vertex.side[2];
        data[offset++] = 0;
    }
    return data;
};

const ribbonSegmentCount = (ribbon: RenderRibbon): number => {
    if (ribbon.vertices.length < 2) return 0;
    return ribbon.closed ? ribbon.vertices.length : ribbon.vertices.length - 1;
};
