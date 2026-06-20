export type VertexBufferState = {
    buffer: GPUBuffer | null,
    capacityVertices: number,
};

export const createVertexBufferState = (): VertexBufferState => ({
    buffer: null,
    capacityVertices: 0,
});

export const createUniformBuffer = (device: GPUDevice, floatCount: number, label: string): GPUBuffer => device.createBuffer({
    label,
    size: floatCount * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

export const createUniformBindGroup = (
    device: GPUDevice,
    layout: GPUBindGroupLayout,
    buffer: GPUBuffer,
    label: string,
): GPUBindGroup => device.createBindGroup({
    label,
    layout,
    entries: [{ binding: 0, resource: { buffer } }],
});

export const uploadVertexData = (
    device: GPUDevice,
    state: VertexBufferState,
    data: Float32Array,
    vertexCount: number,
    label: string,
) => {
    if (vertexCount === 0) return;
    if (vertexCount > state.capacityVertices || !state.buffer) {
        state.buffer?.destroy();
        state.buffer = device.createBuffer({
            label,
            size: data.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        state.capacityVertices = vertexCount;
    }
    device.queue.writeBuffer(state.buffer, 0, data);
};

export const destroyVertexBuffer = (state: VertexBufferState) => {
    state.buffer?.destroy();
    state.buffer = null;
    state.capacityVertices = 0;
};