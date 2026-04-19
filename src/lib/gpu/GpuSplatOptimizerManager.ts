import backwardModuleSrc from "./splat_backward.wgsl?raw";
import stepModuleSrc from "./splat_step.wgsl?raw";
import renderModuleSrc from "./splat_render.wgsl?raw";

export class GpuSplatOptimizerManager {
    private readonly device: GPUDevice;
    
    readonly numSplats: number;
    readonly numParams: number;

    readonly splatBuffer: GPUBuffer;
    readonly gradBuffer: GPUBuffer;
    readonly adamBuffer: GPUBuffer;

    private readonly backwardPipeline: GPUComputePipeline;
    private readonly stepPipeline: GPUComputePipeline;
    private readonly renderPipeline: GPURenderPipeline;

    private backwardBindGroupLayout: GPUBindGroupLayout;
    private stepBindGroupLayout: GPUBindGroupLayout;
    private renderBindGroupLayout: GPUBindGroupLayout;

    private backwardBindGroup!: GPUBindGroup;
    private stepBindGroup: GPUBindGroup;
    private renderBindGroup!: GPUBindGroup;

    private dims: { width: number, height: number } = { width: 0, height: 0 };

    constructor({
        device,
        format,
        numSplats = 512,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        numSplats?: number,
    }) {
        this.device = device;
        this.numSplats = numSplats;
        this.numParams = numSplats * 11;
        
        // Init Buffers
        const splatData = new Float32Array(this.numSplats * 12);
        for (let i = 0; i < this.numSplats; i++) {
            const o = i * 12;
            // position: cluster near center
            splatData[o + 0] = (Math.random() * 2 - 1) * 0.5;
            splatData[o + 1] = (Math.random() * 2 - 1) * 0.5;
            // scale: start small
            splatData[o + 2] = 0.1 + Math.random() * 0.15;
            splatData[o + 3] = 0.1 + Math.random() * 0.15;
            
            // color
            splatData[o + 4] = Math.random();
            splatData[o + 5] = Math.random();
            splatData[o + 6] = Math.random();
            // opacity: moderate
            splatData[o + 7] = 0.3 + Math.random() * 0.4;
            
            // rotation
            splatData[o + 8] = Math.random() * Math.PI * 2;
            // shape_a (power)
            splatData[o + 9] = 2.0;
            // shape_b (multiplier)
            splatData[o + 10] = 0.5;
        }

        this.splatBuffer = device.createBuffer({
            label: "splat buffer",
            size: splatData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.splatBuffer, 0, splatData);

        this.gradBuffer = device.createBuffer({
            label: "splat grad buffer",
            size: this.numParams * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // m (numParams * 4) + v (numParams * 4) + t (4) + pad (12) + extra padding (16)
        this.adamBuffer = device.createBuffer({
            label: "splat adam buffer",
            size: this.numParams * 8 + 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const injectConstants = (src: string) => src
            .replace(/NUM_SPLATS_PLUS_ONE/g, `${this.numSplats + 1}u`)
            .replace(/NUM_SPLATS_MINUS_ONE/g, `${this.numSplats - 1}u`)
            .replace(/NUM_SPLATS/g, `${this.numSplats}u`)
            .replace(/NUM_PARAMS/g, `${this.numParams}u`);

        // Backward Pipeline
        this.backwardBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
            ],
        });
        const backwardModule = device.createShaderModule({ label: "splat backward", code: injectConstants(backwardModuleSrc) });
        backwardModule.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[splat_backward] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
        });
        this.backwardPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.backwardBindGroupLayout] }),
            compute: { module: backwardModule, entryPoint: "main" },
        });

        // Step Pipeline
        this.stepBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ],
        });
        const stepModule = device.createShaderModule({ label: "splat step", code: injectConstants(stepModuleSrc) });
        stepModule.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[splat_step] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
        });
        this.stepPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.stepBindGroupLayout] }),
            compute: { module: stepModule, entryPoint: "main" },
        });

        this.stepBindGroup = device.createBindGroup({
            layout: this.stepBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.splatBuffer } },
                { binding: 1, resource: { buffer: this.gradBuffer } },
                { binding: 2, resource: { buffer: this.adamBuffer } },
            ],
        });

        const renderModule = device.createShaderModule({ label: "splat render", code: injectConstants(renderModuleSrc) });
        renderModule.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[splat_render] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
        });
        
        this.renderPipeline = device.createRenderPipeline({
            label: "splat render pipeline",
            layout: "auto",
            vertex: { module: renderModule, entryPoint: "vert" },
            fragment: { module: renderModule, entryPoint: "frag", targets: [{ format }] },
            primitive: { topology: "triangle-list" },
        });
        this.renderBindGroupLayout = this.renderPipeline.getBindGroupLayout(0);
    }

    setBackwardTarget(targetTextureView: GPUTextureView, width: number, height: number) {
        this.dims = { width, height };
        
        this.backwardBindGroup = this.device.createBindGroup({
            layout: this.backwardBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.splatBuffer } },
                { binding: 1, resource: { buffer: this.gradBuffer } },
                { binding: 2, resource: targetTextureView },
            ],
        });
    }

    setRenderTarget(targetTextureView: GPUTextureView) {
        this.renderBindGroup = this.device.createBindGroup({
            layout: this.renderBindGroupLayout,
            entries: [
                { binding: 0, resource: targetTextureView },
                { binding: 1, resource: { buffer: this.splatBuffer } },
            ],
        });
    }

    dispatch(commandEncoder: GPUCommandEncoder) {
        if (!this.backwardBindGroup) return;

        const pass = commandEncoder.beginComputePass();
        
        // Run optimization step
        pass.setPipeline(this.backwardPipeline);
        pass.setBindGroup(0, this.backwardBindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.dims.width / 16), Math.ceil(this.dims.height / 16));
        
        pass.setPipeline(this.stepPipeline);
        pass.setBindGroup(0, this.stepBindGroup);
        pass.dispatchWorkgroups(8);
        
        pass.end();
    }

    addDraw(renderPassEncoder: GPURenderPassEncoder) {
        if (!this.renderBindGroup) return;
        renderPassEncoder.setPipeline(this.renderPipeline);
        renderPassEncoder.setBindGroup(0, this.renderBindGroup);
        renderPassEncoder.draw(6);
    }
}
