import type { GpuUniformsBufferManager } from "./GpuUniformsBufferManager";
import sphereModuleSrc from "./sphere.wgsl?raw";

export class GpuSphereRenderPipelineManager {
    readonly renderPipeline: GPURenderPipeline;
    readonly uniformsManager: GpuUniformsBufferManager;
    
    readonly vertexBuffer: GPUBuffer;
    readonly indexBuffer: GPUBuffer;
    readonly indexCount: number;

    constructor({
        device,
        format,
        uniformsManager,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        uniformsManager: GpuUniformsBufferManager,
    }) {
        this.uniformsManager = uniformsManager;

        const module = device.createShaderModule({
            label: "sphere module",
            code: sphereModuleSrc,
        });

        const pipelineLayout = device.createPipelineLayout({
            label: "sphere render pipeline layout",
            bindGroupLayouts: [
                uniformsManager.bindGroupLayout,
            ],
        });

        this.renderPipeline = device.createRenderPipeline({
            label: "sphere render pipeline",
            layout: pipelineLayout,

            vertex: {
                module,
                entryPoint: "vert",
                buffers: [
                    {
                        // positions and normals interleaved: [x, y, z, nx, ny, nz]
                        arrayStride: 24, // 6 floats * 4 bytes
                        attributes: [
                            {
                                shaderLocation: 0,
                                offset: 0,
                                format: "float32x3",
                            },
                            {
                                shaderLocation: 1,
                                offset: 12, // 3 floats * 4 bytes
                                format: "float32x3",
                            },
                        ],
                    },
                ],
            },

            fragment: {
                module,
                entryPoint: "frag",
                targets: [
                    {
                        format,
                    },
                    {
                        format,
                    },
                ],
            },

            primitive: {
                topology: "triangle-list",
                cullMode: "back",
                frontFace: "ccw",
            },
        });

        // Generate geometry
        const { vertices, indices } = GpuSphereRenderPipelineManager.generateUvSphere(32, 32);
        this.indexCount = indices.length;

        // Upload to buffers
        this.vertexBuffer = device.createBuffer({
            label: "sphere vertex buffer",
            size: vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.vertexBuffer, 0, vertices);

        this.indexBuffer = device.createBuffer({
            label: "sphere index buffer",
            size: indices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.indexBuffer, 0, indices);
    }

    addDraw(renderPassEncoder: GPURenderPassEncoder) {
        renderPassEncoder.setPipeline(this.renderPipeline);
        renderPassEncoder.setBindGroup(0, this.uniformsManager.bindGroup);
        renderPassEncoder.setVertexBuffer(0, this.vertexBuffer);
        renderPassEncoder.setIndexBuffer(this.indexBuffer, "uint16");
        renderPassEncoder.drawIndexed(this.indexCount);
    }

    private static generateUvSphere(latBands: number, lonBands: number): { vertices: Float32Array, indices: Uint16Array } {
        const vertices: number[] = [];
        const indices: number[] = [];

        for (let lat = 0; lat <= latBands; lat++) {
            const theta = lat * Math.PI / latBands;
            const sinTheta = Math.sin(theta);
            const cosTheta = Math.cos(theta);

            for (let lon = 0; lon <= lonBands; lon++) {
                const phi = lon * 2 * Math.PI / lonBands;
                const sinPhi = Math.sin(phi);
                const cosPhi = Math.cos(phi);

                const x = cosPhi * sinTheta;
                const y = cosTheta;
                const z = sinPhi * sinTheta;

                // Position
                vertices.push(x, y, z);
                // Normal
                vertices.push(x, y, z);
            }
        }

        for (let lat = 0; lat < latBands; lat++) {
            for (let lon = 0; lon < lonBands; lon++) {
                const first = (lat * (lonBands + 1)) + lon;
                const second = first + lonBands + 1;

                indices.push(first, second, first + 1);
                indices.push(second, second + 1, first + 1);
            }
        }

        return {
            vertices: new Float32Array(vertices),
            indices: new Uint16Array(indices),
        };
    }
}
