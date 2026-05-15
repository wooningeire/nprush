import edgeModuleSrc from "./splat_edge.wgsl?raw";
import { injectWgslConstants, constants } from "../constants.ts";

export class GpuSplatEdgeManager {
    private readonly device: GPUDevice;
    private readonly pipeline: GPUComputePipeline;
    private readonly bindGroupLayout: GPUBindGroupLayout;
    private bindGroup: GPUBindGroup | null = null;

    constructor({
        device,
        numSplats,
        numParams,
    }: {
        device: GPUDevice,
        numSplats: number,
        numParams: number,
    }) {
        this.device = device;

        this.bindGroupLayout = device.createBindGroupLayout({
            label: "splat edge bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba8unorm" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
            ],
        });

        const edgeModule = device.createShaderModule({
            label: "splat edge",
            code: injectWgslConstants(edgeModuleSrc, {
                ...constants,
                NUM_SPLATS: numSplats,
                NUM_PARAMS: numParams,
            }),
        });

        this.pipeline = device.createComputePipeline({
            label: "splat edge pipeline",
            layout: device.createPipelineLayout({ 
                label: "splat edge pipeline layout",
                bindGroupLayouts: [this.bindGroupLayout] 
            }),
            compute: { module: edgeModule, entryPoint: "main" },
        });
    }

    setTarget(depthTextureView: GPUTextureView, edgeTextureView: GPUTextureView, normalTextureView?: GPUTextureView) {
        this.bindGroup = this.device.createBindGroup({
            label: "splat edge bind group",
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: depthTextureView },
                { binding: 1, resource: edgeTextureView },
                { binding: 2, resource: normalTextureView ?? depthTextureView },
            ],
        });
    }

    addDispatches(pass: GPUComputePassEncoder, width: number, height: number) {
        if (!this.bindGroup) return;
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
    }
}
