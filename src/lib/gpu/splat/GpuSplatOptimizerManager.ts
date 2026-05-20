import backwardModuleSrc from "./splat_backward.wgsl.ts";
import stepModuleSrc from "./splat_step.wgsl.ts";
import renderBlitModuleSrc from "./render_blit.wgsl?raw";
import renderCompositeModuleSrc from "./render_composite.wgsl?raw";
import adcModuleSrc from "./splat_adc.wgsl.ts";
import initModuleSrc from "./splat_init.wgsl.ts";
import type { Mat4 } from "wgpu-matrix";
import { constants, injectWgslConstants } from "../constants.ts";
import { GpuSplatDepthSortManager } from "./GpuSplatDepthSortManager.ts";
import { GpuSplatBinningManager } from "./GpuSplatBinningManager.ts";
import { GpuSplatEdgeManager } from "./GpuSplatEdgeManager.ts";

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

    private readonly depthSortManager: GpuSplatDepthSortManager;
    private readonly binningManager: GpuSplatBinningManager;
    private readonly edgeManager: GpuSplatEdgeManager;

    get sortIndicesBuffer() { return this.depthSortManager.sortIndicesBuffer; }
    get tileStartsBuffer() { return this.binningManager.tileStartsBuffer; }
    get tileEndsBuffer() { return this.binningManager.tileEndsBuffer; }
    get binningAtomicBuffer() { return this.binningManager.binningAtomicBuffer; }

    

    private readonly backwardPipeline: GPUComputePipeline;
    private readonly stepPipeline: GPUComputePipeline;
    private readonly adcPipeline: GPUComputePipeline;
    private readonly targetPipeline: GPURenderPipeline;
    private readonly compositePipeline: GPURenderPipeline;
    private readonly blitPipeline: GPURenderPipeline;
    private readonly blitRPipeline: GPURenderPipeline;
    private readonly blitAPipeline: GPURenderPipeline;
    private readonly initPipeline: GPUComputePipeline;
    private initBindGroup: GPUBindGroup;

    private backwardBindGroupLayout: GPUBindGroupLayout;
    private stepBindGroupLayout: GPUBindGroupLayout;
    private renderBindGroupLayout: GPUBindGroupLayout;
    private blitBindGroupLayout: GPUBindGroupLayout;

    private backwardBindGroup!: GPUBindGroup;
    private stepBindGroup: GPUBindGroup;
    private adcBindGroup: GPUBindGroup;
    private renderBindGroup!: GPUBindGroup;
    private blitBindGroups: Record<number, GPUBindGroup> = {};
    
    private stepCount: number = 0;

    private dims: { width: number, height: number } = { width: 0, height: 0 };
    private cachedAdamPixelCount: number | null = null;

    constructor({
        device,
        format,
        numSplats,
    }: {
        device: GPUDevice,
        format: GPUTextureFormat,
        numSplats: number,
    }) {
        this.device = device;
        this.numSplats = numSplats;
        const floatsPer = constants.nSplatFloatParams.value;
        this.numParams = numSplats * floatsPer;

        this.splatBuffer = device.createBuffer({
            label: "splat buffer",
            size: this.numSplats * floatsPer * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.gradBuffer = device.createBuffer({
            label: "splat grad buffer",
            size: this.numParams * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
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
            size: 256 * 10,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.splatUniformsBuffer = device.createBuffer({
            label: "splat VP uniforms buffer",
            size: 160,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.binningManager = new GpuSplatBinningManager({
            device,
            numSplats,
            numParams: this.numParams,
            splatBuffer: this.splatBuffer,
        });
        this.depthSortManager = new GpuSplatDepthSortManager({
            device,
            nSplats: numSplats,
            nParams: this.numParams,
            splatBuffer: this.splatBuffer,
        });
        this.edgeManager = new GpuSplatEdgeManager({
            device,
            numSplats,
            numParams: this.numParams,
        });

        const injectConstants = (src: string) => injectWgslConstants(src, {
            ...constants,
            NUM_SPLATS: this.numSplats,
            NUM_SPLATS_PLUS_ONE: this.numSplats + 1,
            NUM_SPLATS_MINUS_ONE: this.numSplats - 1,
            NUM_SPLATS_DIV_32: Math.ceil(this.numSplats / 32),
            NUM_PARAMS: this.numParams,
        });

        this.backwardBindGroupLayout = device.createBindGroupLayout({
            label: "splat backward bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            ],
        });
        this.backwardPipeline = device.createComputePipeline({
            label: "splat backward pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.backwardBindGroupLayout] }),
            compute: { module: device.createShaderModule({ label: "splat backward", code: injectConstants(backwardModuleSrc) }), entryPoint: "main" },
        });

        this.stepBindGroupLayout = device.createBindGroupLayout({
            label: "splat step bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
        });
        this.stepPipeline = device.createComputePipeline({
            label: "splat step pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.stepBindGroupLayout] }),
            compute: { module: device.createShaderModule({ label: "splat step", code: injectConstants(stepModuleSrc) }), entryPoint: "main" },
        });
        this.stepBindGroup = device.createBindGroup({
            label: "splat step bind group",
            layout: this.stepBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.splatBuffer } },
                { binding: 1, resource: { buffer: this.gradBuffer } },
                { binding: 2, resource: { buffer: this.adamBuffer } },
                { binding: 3, resource: { buffer: this.adcBuffer } },
                { binding: 4, resource: { buffer: this.splatUniformsBuffer } },
            ],
        });

        const adcBindGroupLayout = device.createBindGroupLayout({
            label: "splat adc bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ],
        });
        this.adcPipeline = device.createComputePipeline({
            label: "splat adc pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [adcBindGroupLayout] }),
            compute: { module: device.createShaderModule({ label: "splat adc", code: injectConstants(adcModuleSrc) }), entryPoint: "main" },
        });
        this.adcBindGroup = device.createBindGroup({
            label: "splat adc bind group",
            layout: adcBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.splatBuffer } },
                { binding: 1, resource: { buffer: this.adamBuffer } },
                { binding: 2, resource: { buffer: this.adcBuffer } },
            ],
        });

        this.renderBindGroupLayout = device.createBindGroupLayout({
            label: "splat composite render bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform", hasDynamicOffset: true } },
                { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
            ],
        });
        this.blitBindGroupLayout = device.createBindGroupLayout({
            label: "splat blit render bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform", hasDynamicOffset: true } },
            ],
        });

        const compositeModule = device.createShaderModule({ label: "splat composite render", code: injectConstants(renderCompositeModuleSrc) });
        const blitModule = device.createShaderModule({ label: "splat blit render", code: injectConstants(renderBlitModuleSrc) });

        this.targetPipeline = device.createRenderPipeline({
            label: "splat target render pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.renderBindGroupLayout] }),
            vertex: { module: compositeModule, entryPoint: "vert" },
            fragment: { module: compositeModule, entryPoint: "frag_target", targets: [{ format }] },
            primitive: { topology: "triangle-list" },
        });
        this.compositePipeline = device.createRenderPipeline({
            label: "splat composite render pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.renderBindGroupLayout] }),
            vertex: { module: compositeModule, entryPoint: "vert" },
            fragment: { module: compositeModule, entryPoint: "frag_composite", targets: [{ format }] },
            primitive: { topology: "triangle-list" },
        });
        const blitLayout = device.createPipelineLayout({ bindGroupLayouts: [this.blitBindGroupLayout] });
        this.blitPipeline = device.createRenderPipeline({
            label: "splat blit render pipeline",
            layout: blitLayout,
            vertex: { module: blitModule, entryPoint: "vert" },
            fragment: { module: blitModule, entryPoint: "frag_blit", targets: [{ format }] },
            primitive: { topology: "triangle-list" },
        });
        this.blitRPipeline = device.createRenderPipeline({
            label: "splat blit R render pipeline",
            layout: blitLayout,
            vertex: { module: blitModule, entryPoint: "vert" },
            fragment: { module: blitModule, entryPoint: "frag_blit_r", targets: [{ format }] },
            primitive: { topology: "triangle-list" },
        });
        this.blitAPipeline = device.createRenderPipeline({
            label: "splat blit A render pipeline",
            layout: blitLayout,
            vertex: { module: blitModule, entryPoint: "vert" },
            fragment: { module: blitModule, entryPoint: "frag_blit_a", targets: [{ format }] },
            primitive: { topology: "triangle-list" },
        });

        const initModule = device.createShaderModule({ label: "splat init", code: injectConstants(initModuleSrc) });
        this.initPipeline = device.createComputePipeline({
            label: "splat init pipeline",
            layout: device.createPipelineLayout({ bindGroupLayouts: [device.createBindGroupLayout({ label: "splat init bgl", entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }] })] }),
            compute: { module: initModule, entryPoint: "main" },
        });
        this.initBindGroup = device.createBindGroup({ label: "splat init bind group", layout: this.initPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.splatBuffer } }] });

        const initEncoder = device.createCommandEncoder({ label: "splat init encoder" });
        const initPass = initEncoder.beginComputePass({ label: "splat init pass" });
        initPass.setPipeline(this.initPipeline);
        initPass.setBindGroup(0, this.initBindGroup);
        initPass.dispatchWorkgroups(Math.ceil(this.numSplats / 64));
        initPass.end();
        device.queue.submit([initEncoder.finish()]);
    }

    writeSplatVPMatrix(mat: Mat4, invMat: Mat4, blurEnabled: boolean = false, camWorldXYZ: readonly [number, number, number] | Float32Array) {
        this.device.queue.writeBuffer(this.splatUniformsBuffer, 0, (mat as Float32Array).buffer, (mat as Float32Array).byteOffset, (mat as Float32Array).byteLength);
        this.device.queue.writeBuffer(this.splatUniformsBuffer, 64, (invMat as Float32Array).buffer, (invMat as Float32Array).byteOffset, (invMat as Float32Array).byteLength);
        const extras = new ArrayBuffer(32);
        const f32 = new Float32Array(extras);
        const u32 = new Uint32Array(extras);
        f32[0] = camWorldXYZ[0]; f32[1] = camWorldXYZ[1]; f32[2] = camWorldXYZ[2];
        u32[4] = blurEnabled ? 1 : 0; u32[5] = this.stepCount;
        this.device.queue.writeBuffer(this.splatUniformsBuffer, 128, extras);
    }

    setBackwardTarget(targetTextureView: GPUTextureView, targetDepthTextureView: GPUTextureView, width: number, height: number) {
        this.dims = { width, height };
        this.backwardBindGroup = this.device.createBindGroup({
            label: "splat backward bind group",
            layout: this.backwardBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.splatBuffer } },
                { binding: 1, resource: { buffer: this.gradBuffer } },
                { binding: 2, resource: targetTextureView },
                { binding: 3, resource: targetDepthTextureView },
                { binding: 4, resource: { buffer: this.splatUniformsBuffer } },
                { binding: 5, resource: { buffer: this.binningManager.instanceValsBufferA } },
                { binding: 6, resource: { buffer: this.binningManager.tileStartsBuffer } },
                { binding: 7, resource: { buffer: this.binningManager.tileEndsBuffer } },
            ],
        });
    }

    setEdgeTarget(depthTextureView: GPUTextureView, edgeTextureView: GPUTextureView, normalTextureView?: GPUTextureView) {
        this.edgeManager.setTarget(depthTextureView, edgeTextureView, normalTextureView);
    }

    setRenderTarget(targetTextureView: GPUTextureView, splatViewTextureView: GPUTextureView, depthTextureView: GPUTextureView, edgeTextureView: GPUTextureView, bezierViewTextureView: GPUTextureView, baseColorBezierViewTextureView: GPUTextureView, colorBezierViewTextureView: GPUTextureView, ptTextureView: GPUTextureView) {
        this.renderBindGroup = this.device.createBindGroup({
            label: "splat render bind group",
            layout: this.renderBindGroupLayout,
            entries: [
                { binding: 0, resource: targetTextureView },
                { binding: 1, resource: splatViewTextureView },
                { binding: 2, resource: depthTextureView },
                { binding: 3, resource: edgeTextureView },
                { binding: 4, resource: bezierViewTextureView },
                { binding: 5, resource: baseColorBezierViewTextureView },
                { binding: 6, resource: colorBezierViewTextureView },
                { binding: 7, resource: { buffer: this.renderUniformsBuffer, size: 256 } },
                { binding: 8, resource: ptTextureView },
            ],
        });
        const blitTextures: Record<number, GPUTextureView> = { 2: splatViewTextureView, 3: depthTextureView, 4: edgeTextureView, 5: bezierViewTextureView, 6: baseColorBezierViewTextureView, 7: colorBezierViewTextureView };
        this.blitBindGroups = {};
        for (const [mode, tex] of Object.entries(blitTextures)) {
            this.blitBindGroups[Number(mode)] = this.device.createBindGroup({
                label: `splat blit bind group mode ${mode}`,
                layout: this.blitBindGroupLayout,
                entries: [{ binding: 0, resource: tex }, { binding: 1, resource: { buffer: this.renderUniformsBuffer, size: 256 } }],
            });
        }
    }

    writeRenderUniforms(edgeEnabled: boolean, baseColorEnabled: boolean, colorEnabled: boolean, meshSplatsEnabled: boolean, splatsEnabled: boolean, aspects: Record<number, number>) {
        const buffer = new ArrayBuffer(32);
        const u32 = new Uint32Array(buffer);
        const f32 = new Float32Array(buffer);
        u32[0] = edgeEnabled ? 1 : 0; u32[1] = baseColorEnabled ? 1 : 0; u32[2] = colorEnabled ? 1 : 0; u32[3] = meshSplatsEnabled ? 1 : 0; u32[4] = splatsEnabled ? 1 : 0;
        for (let mode = 0; mode < 10; mode++) {
            f32[5] = aspects[mode] ?? 1.0;
            this.device.queue.writeBuffer(this.renderUniformsBuffer, mode * 256, buffer);
        }
    }

    resetAdam() {
        this.device.queue.writeBuffer(this.adamBuffer, 0, new Float32Array(this.numParams * 2));
        this.device.queue.writeBuffer(this.adamBuffer, this.numParams * 8, new Uint32Array([0]));
        this.cachedAdamPixelCount = null;
        this.device.queue.writeBuffer(this.adcBuffer, 0, new Float32Array(this.numSplats));
    }

    writeNoKill(noKill: boolean) {
        this.device.queue.writeBuffer(this.adamBuffer, this.numParams * 8 + 8, new Uint32Array([noKill ? 1 : 0]));
    }

    addBinningDispatches(pass: GPUComputePassEncoder, vpMat: Mat4) {
        this.binningManager.addDispatches(pass, vpMat, this.dims.width, this.dims.height);
    }

    clearBinningBuffers(commandEncoder: GPUCommandEncoder) {
        const gridWidth = Math.ceil(this.dims.width);
        const gridHeight = Math.ceil(this.dims.height / 16);
        const nTiles = gridWidth * gridHeight;

        commandEncoder.clearBuffer(this.binningManager.binningAtomicBuffer, 0, 4);
        commandEncoder.clearBuffer(this.binningManager.tileStartsBuffer, 0, nTiles * 4);
        commandEncoder.clearBuffer(this.binningManager.tileEndsBuffer, 0, nTiles * 4);
    }

    addOptimizationDispatches(pass: GPUComputePassEncoder) {
        if (!this.backwardBindGroup) return;
        const { width, height } = this.dims;
        if (width === 0 || height === 0) return;

        const pixelCount = width * height;
        if (this.cachedAdamPixelCount !== pixelCount) {
            this.cachedAdamPixelCount = pixelCount;
            this.device.queue.writeBuffer(this.adamBuffer, this.numParams * 8 + 4, new Uint32Array([pixelCount]));
        }
        this.device.queue.writeBuffer(this.splatUniformsBuffer, 148, new Uint32Array([this.stepCount]));

        pass.setPipeline(this.backwardPipeline);
        pass.setBindGroup(0, this.backwardBindGroup);
        pass.dispatchWorkgroups(Math.ceil(width / 16), Math.ceil(height / 16));
        
        pass.setPipeline(this.stepPipeline);
        pass.setBindGroup(0, this.stepBindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.numSplats / 64));
        
        this.stepCount++;
        if (this.stepCount % constants.splatAdcPeriod === 0) {
            pass.setPipeline(this.adcPipeline);
            pass.setBindGroup(0, this.adcBindGroup);
            pass.dispatchWorkgroups(1);
        }
    }

    addEdgeDispatches(pass: GPUComputePassEncoder, width: number, height: number) {
        this.edgeManager.addDispatches(pass, width, height);
    }

    addDepthSortDispatches(pass: GPUComputePassEncoder, vpMat: Mat4) {
        this.depthSortManager.addDispatches(pass, vpMat);
    }

    addDraw(renderPassEncoder: GPURenderPassEncoder, mode: number) {
        if (mode === 0) {
            renderPassEncoder.setPipeline(this.targetPipeline);
            renderPassEncoder.setBindGroup(0, this.renderBindGroup, [0]);
            renderPassEncoder.draw(3);
        } else if (mode === 1) {
            renderPassEncoder.setPipeline(this.compositePipeline);
            renderPassEncoder.setBindGroup(0, this.renderBindGroup, [mode * 256]);
            renderPassEncoder.draw(3);
        } else {
            const bg = this.blitBindGroups[mode];
            if (bg) {
                const pipelines: Record<number, GPURenderPipeline> = { 3: this.blitRPipeline, 4: this.blitRPipeline, 5: this.blitAPipeline, 6: this.blitPipeline, 7: this.blitPipeline };
                const p = pipelines[mode] ?? this.blitPipeline;
                renderPassEncoder.setPipeline(p);
                renderPassEncoder.setBindGroup(0, bg, [mode * 256]);
                renderPassEncoder.draw(3);
            }
        }
    }

    destroy() {
        this.splatBuffer.destroy();
        this.gradBuffer.destroy();
        this.adamBuffer.destroy();
        this.adcBuffer.destroy();
        this.renderUniformsBuffer.destroy();
        this.splatUniformsBuffer.destroy();
        this.binningManager.destroy();
        this.depthSortManager.destroy();
    }
}
