import type { Mat4 } from "wgpu-matrix";

export class GpuUniformsBufferManager {
    readonly device: GPUDevice;
    
    readonly bindGroupLayout: GPUBindGroupLayout;
    readonly bindGroup: GPUBindGroup;

    readonly uniformsBuffer: GPUBuffer;
    
    private readonly viewProjMatOffset = 0;
    private readonly viewMatOffset = 64;
    private readonly shadingModeOffset = 128;
    private readonly invViewProjMatOffset = 144;

    constructor({
        device,
    }: {
        device: GPUDevice,
    }) {
        this.device = device;
        
        // 64 bytes for viewProjMat, 64 for viewMat, 16 for shadingMode + padding, 64 for invViewProjMat
        const bufferSize = 208;

        this.uniformsBuffer = device.createBuffer({
            label: "uniforms buffer",
            size: bufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.bindGroupLayout = device.createBindGroupLayout({
            label: "uniforms bind group layout",
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: "uniform",
                    },
                },
            ],
        });

        this.bindGroup = device.createBindGroup({
            label: "uniforms bind group",
            layout: this.bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.uniformsBuffer,
                    },
                },
            ],
        });
    }

    writeViewProjMat(mat: Mat4) {
        this.device.queue.writeBuffer(
            this.uniformsBuffer, 
            this.viewProjMatOffset, 
            (mat as Float32Array).buffer, 
            (mat as Float32Array).byteOffset, 
            (mat as Float32Array).byteLength
        );
    }

    writeViewMat(mat: Mat4) {
        this.device.queue.writeBuffer(
            this.uniformsBuffer, 
            this.viewMatOffset, 
            (mat as Float32Array).buffer, 
            (mat as Float32Array).byteOffset, 
            (mat as Float32Array).byteLength
        );
    }

    writeShadingMode(mode: 'normals' | 'shaded') {
        this.device.queue.writeBuffer(
            this.uniformsBuffer,
            this.shadingModeOffset,
            new Float32Array([mode === 'normals' ? 0 : 1])
        );
    }

    writeInvViewProjMat(mat: Mat4) {
        this.device.queue.writeBuffer(
            this.uniformsBuffer,
            this.invViewProjMatOffset,
            (mat as Float32Array).buffer,
            (mat as Float32Array).byteOffset,
            (mat as Float32Array).byteLength
        );
    }
}
