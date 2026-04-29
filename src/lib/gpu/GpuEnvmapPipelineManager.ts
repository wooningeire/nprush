import type { GpuUniformsBufferManager } from "./GpuUniformsBufferManager";
import envmapSrc from "./envmap.wgsl?raw";

// Renders the equirectangular environment map as a fullscreen background.
// Must be drawn before the mesh pass (or with depth test = always + no depth write)
// so the mesh always renders on top.
export class GpuEnvmapPipelineManager {
    private readonly device: GPUDevice;
    private readonly pipeline: GPURenderPipeline;
    private readonly bindGroup: GPUBindGroup;

    constructor({
        device,
        format,
        uniformsManager,
        envTexture,
    }: {
        device: GPUDevice;
        format: GPUTextureFormat;
        uniformsManager: GpuUniformsBufferManager;
        envTexture: GPUTexture;
    }) {
        this.device = device;

        const module = device.createShaderModule({
            label: "envmap module",
            code: envmapSrc,
        });
        module.getCompilationInfo().then(info => {
            for (const m of info.messages) console.warn(`[envmap] ${m.type}: ${m.message} (line ${m.lineNum})`);
        });

        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
            ],
        });

        this.pipeline = device.createRenderPipeline({
            label: "envmap pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: { module, entryPoint: "vert" },
            fragment: {
                module,
                entryPoint: "frag",
                targets: [
                    { format },
                    // Second color target (depth visualization) — write black/transparent
                    { format },
                ],
            },
            primitive: { topology: "triangle-list" },
            // No depth write; depth test always passes so the mesh Z-buffer is unaffected.
            depthStencil: {
                format: "depth24plus",
                depthWriteEnabled: false,
                depthCompare: "always",
            },
        });

        this.bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: uniformsManager.uniformsBuffer } },
                { binding: 1, resource: envTexture.createView() },
                { binding: 2, resource: device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "repeat", addressModeV: "clamp-to-edge" }) },
            ],
        });
    }

    addDraw(pass: GPURenderPassEncoder) {
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.draw(6);
    }
}
