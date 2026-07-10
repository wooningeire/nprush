import type { Vec2 } from "../types.ts";
import { DEPTH_FORMAT } from "./constants.ts";

const SOURCE_POINT_FLOATS = 4;

export const createBrushGuidePipeline = (
    device: GPUDevice,
    layout: GPUPipelineLayout,
    module: GPUShaderModule,
    format: GPUTextureFormat,
    fragmentEntryPoint: string,
    depthCompare: GPUCompareFunction,
    label: string,
): GPURenderPipeline => device.createRenderPipeline({
    label,
    layout,
    vertex: {
        module,
        entryPoint: "guide_vertex",
    },
    fragment: {
        module,
        entryPoint: fragmentEntryPoint,
        targets: [{
            format,
            blend: {
                color: {
                    operation: "add",
                    srcFactor: "src-alpha",
                    dstFactor: "one-minus-src-alpha",
                },
                alpha: {
                    operation: "add",
                    srcFactor: "src-alpha",
                    dstFactor: "one-minus-src-alpha",
                },
            },
        }],
    },
    primitive: {
        topology: "line-list",
        cullMode: "none",
    },
    depthStencil: {
        format: DEPTH_FORMAT,
        depthCompare,
        depthWriteEnabled: false,
    },
});

export const createSourcePointBuffer = (
    device: GPUDevice,
    sourcePoints: Vec2[],
): GPUBuffer => {
    const data = new Float32Array(Math.max(1, sourcePoints.length) * SOURCE_POINT_FLOATS);
    for (let index = 0; index < sourcePoints.length; index++) {
        const offset = index * SOURCE_POINT_FLOATS;
        data[offset] = sourcePoints[index].x;
        data[offset + 1] = sourcePoints[index].y;
    }

    const buffer = createBuffer(
        device,
        data.byteLength,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        "paint brush source points",
    );
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
};

export const createBuffer = (
    device: GPUDevice,
    size: number,
    usage: GPUBufferUsageFlags,
    label: string,
): GPUBuffer => device.createBuffer({
    label,
    size: Math.max(16, alignTo(size, 16)),
    usage,
});

export const destroyBuffers = (buffers: GPUBuffer[]) => {
    for (const buffer of buffers) {
        buffer.destroy();
    }
};

export const createLoggedShaderModule = (
    device: GPUDevice,
    label: string,
    code: string,
): GPUShaderModule => {
    const module = device.createShaderModule({ label, code });
    void module.getCompilationInfo().then(info => {
        for (const message of info.messages) {
            console.warn(
                "[" + label + "] " + message.type + ": " + message.message
                + " (line " + message.lineNum + ")",
            );
        }
    });
    return module;
};

const alignTo = (value: number, alignment: number): number => (
    Math.ceil(value / alignment) * alignment
);