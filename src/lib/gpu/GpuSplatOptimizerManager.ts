import backwardModuleSrc from "./splat_backward.wgsl?raw";
import stepModuleSrc from "./splat_step.wgsl?raw";
import renderModuleSrc from "./splat_render.wgsl?raw";
import adcModuleSrc from "./splat_adc.wgsl?raw";
import edgeModuleSrc from "./splat_edge.wgsl?raw";
import type { Mat4 } from "wgpu-matrix";

export class GpuSplatOptimizerManager {
    private readonly device: GPUDevice;
    
    readonly numSplats: number;
    readonly numParams: number;

    readonly splatBuffer: GPUBuffer;
    readonly gradBuffer: GPUBuffer;
    readonly adamBuffer: GPUBuffer;
    readonly adcBuffer: GPUBuffer;
    readonly renderUniformsBuffer: GPUBuffer;
    readonly splatUniformsBuffer: GPUBuffer;

    private readonly backwardPipeline: GPUComputePipeline;
    private readonly stepPipeline: GPUComputePipeline;
    private readonly adcPipeline: GPUComputePipeline;
    private readonly edgePipeline: GPUComputePipeline;
    private readonly renderPipeline: GPURenderPipeline;

    private backwardBindGroupLayout: GPUBindGroupLayout;
    private stepBindGroupLayout: GPUBindGroupLayout;
    private edgeBindGroupLayout: GPUBindGroupLayout;
    private renderBindGroupLayout: GPUBindGroupLayout;

    private backwardBindGroup!: GPUBindGroup;
    private stepBindGroup: GPUBindGroup;
    private adcBindGroup: GPUBindGroup;
    private edgeBindGroup!: GPUBindGroup;
    private renderBindGroup!: GPUBindGroup;
    
    private stepCount: number = 0;

    private dims: { width: number, height: number } = { width: 0, height: 0 };

    constructor({
        device,
        format,
        numSplats = 512,
        numBeziers,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        numSplats?: number,
        numBeziers?: number,
    }) {
        this.device = device;
        this.numSplats = numSplats;
        this.numParams = numSplats * 15;
        
        // Init Buffers — 4 × vec4f = 16 floats per splat
        const splatData = new Float32Array(this.numSplats * 16);
        for (let i = 0; i < this.numSplats; i++) {
            const o = i * 16;
            // pos_sx: x, y, z, sx
            splatData[o + 0] = (Math.random() * 2 - 1) * 0.3;
            splatData[o + 1] = (Math.random() * 2 - 1) * 0.3;
            splatData[o + 2] = (Math.random() * 2 - 1) * 0.3;
            splatData[o + 3] = 0.1 + Math.random() * 0.15;  // sx
            // color: r, g, b, opacity
            splatData[o + 4] = Math.random();
            splatData[o + 5] = Math.random();
            splatData[o + 6] = Math.random();
            if (i < 512) {
                splatData[o + 7] = 0.3 + Math.random() * 0.4;
            } else {
                splatData[o + 7] = 0.0;
            }
            // quat: qw, qx, qy, qz — identity
            splatData[o + 8] = 1.0;
            splatData[o + 9] = 0.0;
            splatData[o + 10] = 0.0;
            splatData[o + 11] = 0.0;
            // sy_shape: sy, shape_a, shape_b, pad
            splatData[o + 12] = 0.1 + Math.random() * 0.15; // sy
            splatData[o + 13] = 2.0;  // shape_a
            splatData[o + 14] = 0.5;  // shape_b
            splatData[o + 15] = 0.0;  // pad
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

        this.adcBuffer = device.createBuffer({
            label: "splat adc buffer",
            size: this.numSplats * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.renderUniformsBuffer = device.createBuffer({
            label: "splat render uniforms buffer",
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // VP matrix for backward and forward shaders (mat4x4f = 64 bytes) + inv VP (64 bytes) + blur_enabled (4 bytes) + padding
        this.splatUniformsBuffer = device.createBuffer({
            label: "splat VP uniforms buffer",
            size: 160,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const injectConstants = (src: string) => src
            .replace(/NUM_SPLATS_PLUS_ONE/g, `${this.numSplats + 1}u`)
            .replace(/NUM_SPLATS_MINUS_ONE/g, `${this.numSplats - 1}u`)
            .replace(/NUM_SPLATS_DIV_32/g, `${Math.ceil(this.numSplats / 32)}u`)
            .replace(/NUM_SPLATS/g, `${this.numSplats}u`)
            .replace(/NUM_PARAMS/g, `${this.numParams}u`);

        // Backward Pipeline — now has 5 bindings (splats, grads, target, edge, VP uniform)
        this.backwardBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
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
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
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
                { binding: 3, resource: { buffer: this.adcBuffer } },
            ],
        });

        // ADC Pipeline
        const adcBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ],
        });
        const adcModule = device.createShaderModule({ label: "splat adc", code: injectConstants(adcModuleSrc) });
        adcModule.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[splat_adc] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
        });
        this.adcPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [adcBindGroupLayout] }),
            compute: { module: adcModule, entryPoint: "main" },
        });

        this.adcBindGroup = device.createBindGroup({
            layout: adcBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.splatBuffer } },
                { binding: 1, resource: { buffer: this.adamBuffer } },
                { binding: 2, resource: { buffer: this.adcBuffer } },
            ],
        });

        // Edge Detection Pipeline
        this.edgeBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba8unorm" } },
            ],
        });
        const edgeModule = device.createShaderModule({ label: "splat edge", code: edgeModuleSrc });
        edgeModule.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[splat_edge] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
        });
        this.edgePipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.edgeBindGroupLayout] }),
            compute: { module: edgeModule, entryPoint: "main" },
        });

        const renderModule = device.createShaderModule({ label: "splat render", code: injectConstants(renderModuleSrc) });
        renderModule.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[splat_render] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
        });
        
        this.renderBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
                { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
            ],
        });

        this.renderPipeline = device.createRenderPipeline({
            label: "splat render pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.renderBindGroupLayout] }),
            vertex: { module: renderModule, entryPoint: "vert" },
            fragment: { module: renderModule, entryPoint: "frag", targets: [{ format }] },
            primitive: { topology: "triangle-list" },
        });
    }

    writeSplatVPMatrix(mat: Mat4, invMat: Mat4, blurEnabled: boolean = false) {
        this.device.queue.writeBuffer(
            this.splatUniformsBuffer,
            0,
            (mat as Float32Array).buffer,
            (mat as Float32Array).byteOffset,
            (mat as Float32Array).byteLength
        );
        this.device.queue.writeBuffer(
            this.splatUniformsBuffer,
            64,
            (invMat as Float32Array).buffer,
            (invMat as Float32Array).byteOffset,
            (invMat as Float32Array).byteLength
        );
        this.device.queue.writeBuffer(
            this.splatUniformsBuffer,
            128,
            new Float32Array([blurEnabled ? 1 : 0])
        );
    }

    setBackwardTarget(targetTextureView: GPUTextureView, targetDepthTextureView: GPUTextureView, width: number, height: number) {
        this.dims = { width, height };
        
        this.backwardBindGroup = this.device.createBindGroup({
            layout: this.backwardBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.splatBuffer } },
                { binding: 1, resource: { buffer: this.gradBuffer } },
                { binding: 2, resource: targetTextureView },
                { binding: 3, resource: targetDepthTextureView }, // actual depth, not edge
                { binding: 4, resource: { buffer: this.splatUniformsBuffer } },
            ],
        });
    }

    setEdgeTarget(depthTextureView: GPUTextureView, edgeTextureView: GPUTextureView) {
        this.edgeBindGroup = this.device.createBindGroup({
            layout: this.edgeBindGroupLayout,
            entries: [
                { binding: 0, resource: depthTextureView },
                { binding: 1, resource: edgeTextureView },
            ],
        });
    }

    setRenderTarget(
        targetTextureView: GPUTextureView,
        splatViewTextureView: GPUTextureView,
        depthTextureView: GPUTextureView,
        edgeTextureView: GPUTextureView,
        bezierViewTextureView: GPUTextureView,
        baseColorBezierViewTextureView: GPUTextureView,
        colorBezierViewTextureView: GPUTextureView,
        ptTextureView: GPUTextureView,
    ) {
        this.renderBindGroup = this.device.createBindGroup({
            layout: this.renderBindGroupLayout,
            entries: [
                { binding: 0, resource: targetTextureView },
                { binding: 1, resource: splatViewTextureView },
                { binding: 2, resource: depthTextureView },
                { binding: 3, resource: edgeTextureView },
                { binding: 4, resource: bezierViewTextureView },
                { binding: 5, resource: baseColorBezierViewTextureView },
                { binding: 6, resource: colorBezierViewTextureView },
                { binding: 7, resource: { buffer: this.renderUniformsBuffer } },
                { binding: 8, resource: ptTextureView },
            ],
        });
    }

    writeRenderUniforms(edgeEnabled: boolean, baseColorEnabled: boolean, colorEnabled: boolean, posterizationEnabled: boolean) {
        this.device.queue.writeBuffer(
            this.renderUniformsBuffer,
            0,
            new Float32Array([edgeEnabled ? 1 : 0, baseColorEnabled ? 1 : 0, colorEnabled ? 1 : 0, posterizationEnabled ? 1 : 0])
        );
    }

    dispatch(commandEncoder: GPUCommandEncoder) {
        if (!this.backwardBindGroup) return;

        const pass = commandEncoder.beginComputePass();
        
        pass.setPipeline(this.backwardPipeline);
        pass.setBindGroup(0, this.backwardBindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.dims.width / 16), Math.ceil(this.dims.height / 16));
        
        pass.setPipeline(this.stepPipeline);
        pass.setBindGroup(0, this.stepBindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.numSplats / 64));
        
        this.stepCount++;
        if (this.stepCount % 25 === 0) {
            pass.setPipeline(this.adcPipeline);
            pass.setBindGroup(0, this.adcBindGroup);
            pass.dispatchWorkgroups(1);
        }
        
        pass.end();
    }

    dispatchEdge(commandEncoder: GPUCommandEncoder, width: number, height: number) {
        if (!this.edgeBindGroup) return;
        
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(this.edgePipeline);
        pass.setBindGroup(0, this.edgeBindGroup);
        pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
        pass.end();
    }

    addDraw(renderPassEncoder: GPURenderPassEncoder) {
        if (!this.renderBindGroup) return;
        renderPassEncoder.setPipeline(this.renderPipeline);
        renderPassEncoder.setBindGroup(0, this.renderBindGroup);
        renderPassEncoder.draw(6);
    }
}
