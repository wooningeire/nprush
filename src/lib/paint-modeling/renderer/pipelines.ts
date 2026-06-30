import {
    DEPTH_FORMAT,
    FLOATS_PER_TRIANGLE_VERTEX,
    FLOATS_PER_VERTEX,
} from "./constants.ts";

const createColorTarget = (format: GPUTextureFormat): GPUColorTargetState => ({
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
});

export const createGridPipeline = (
    device: GPUDevice,
    layout: GPUPipelineLayout,
    module: GPUShaderModule,
    format: GPUTextureFormat,
): GPURenderPipeline => device.createRenderPipeline({
    label: "paint modeler grid pipeline",
    layout,
    vertex: {
        module,
        entryPoint: "grid_vertex",
    },
    fragment: {
        module,
        entryPoint: "grid_fragment",
        targets: [createColorTarget(format)],
    },
    primitive: {
        topology: "triangle-list",
        cullMode: "none",
    },
    depthStencil: {
        format: DEPTH_FORMAT,
        depthCompare: "always",
        depthWriteEnabled: false,
    },
});

export const createSegmentPipeline = (
    device: GPUDevice,
    layout: GPUPipelineLayout,
    module: GPUShaderModule,
    format: GPUTextureFormat,
    topology: GPUPrimitiveTopology,
): GPURenderPipeline => device.createRenderPipeline({
    label: `paint modeler segment ${topology} pipeline`,
    layout,
    vertex: {
        module,
        entryPoint: "segment_vertex",
        buffers: [{
            arrayStride: FLOATS_PER_VERTEX * Float32Array.BYTES_PER_ELEMENT,
            attributes: [
                { shaderLocation: 0, offset: 0, format: "float32x3" },
                { shaderLocation: 1, offset: 3 * Float32Array.BYTES_PER_ELEMENT, format: "float32x3" },
                { shaderLocation: 2, offset: 6 * Float32Array.BYTES_PER_ELEMENT, format: "float32x3" },
                { shaderLocation: 3, offset: 9 * Float32Array.BYTES_PER_ELEMENT, format: "float32x4" },
                { shaderLocation: 4, offset: 13 * Float32Array.BYTES_PER_ELEMENT, format: "float32" },
                { shaderLocation: 5, offset: 14 * Float32Array.BYTES_PER_ELEMENT, format: "float32" },
                { shaderLocation: 6, offset: 15 * Float32Array.BYTES_PER_ELEMENT, format: "float32" },
            ],
        }],
    },
    fragment: {
        module,
        entryPoint: "segment_fragment",
        targets: [createColorTarget(format)],
    },
    primitive: {
        topology,
        cullMode: "none",
    },
    depthStencil: {
        format: DEPTH_FORMAT,
        depthCompare: "less-equal",
        depthWriteEnabled: false,
    },
});

export const createTrianglePipeline = (
    device: GPUDevice,
    layout: GPUPipelineLayout,
    module: GPUShaderModule,
    format: GPUTextureFormat,
    depthWriteEnabled: boolean,
): GPURenderPipeline => device.createRenderPipeline({
    label: depthWriteEnabled
        ? "paint modeler depth triangle pipeline"
        : "paint modeler overlay triangle pipeline",
    layout,
    vertex: {
        module,
        entryPoint: "triangle_vertex",
        buffers: [{
            arrayStride: FLOATS_PER_TRIANGLE_VERTEX * Float32Array.BYTES_PER_ELEMENT,
            attributes: [
                { shaderLocation: 0, offset: 0, format: "float32x3" },
                { shaderLocation: 1, offset: 3 * Float32Array.BYTES_PER_ELEMENT, format: "float32x4" },
                { shaderLocation: 2, offset: 7 * Float32Array.BYTES_PER_ELEMENT, format: "float32x3" },
                { shaderLocation: 3, offset: 10 * Float32Array.BYTES_PER_ELEMENT, format: "float32" },
            ],
        }],
    },
    fragment: {
        module,
        entryPoint: "triangle_fragment",
        targets: [createColorTarget(format)],
    },
    primitive: {
        topology: "triangle-list",
        cullMode: "none",
    },
    depthStencil: {
        format: DEPTH_FORMAT,
        depthCompare: "less-equal",
        depthWriteEnabled,
    },
});
