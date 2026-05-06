import pathTraceSrc from "./path_trace.wgsl?raw";
import resolveSrc from "./path_trace_resolve.wgsl?raw";
import { buildBvh } from "./bvh";
import type { MeshData } from "./loadGlb";

// Progressive path tracer with BVH acceleration.
// Each dispatch() adds one sample per pixel; reset() clears on camera move.
export class GpuPathTracePipelineManager {
    private readonly device: GPUDevice;

    private vertexBuffer:  GPUBuffer | null = null;
    private bvhNodeBuffer: GPUBuffer | null = null;
    private bvhTriBuffer:  GPUBuffer | null = null;
    private numTris = 0;

    private accumBuffer:   GPUBuffer | null = null;
    private accumWidth  = 0;
    private accumHeight = 0;

    private outputTexture: GPUTexture | null = null;
    outputTextureView: GPUTextureView | null = null;

    private readonly ptUniformsBuffer:      GPUBuffer;
    private readonly resolveUniformsBuffer: GPUBuffer;

    private readonly ptPipeline:      GPUComputePipeline;
    private readonly resolvePipeline: GPUComputePipeline;

    private readonly ptBindGroupLayout:      GPUBindGroupLayout;
    private readonly resolveBindGroupLayout: GPUBindGroupLayout;

    private ptBindGroup:      GPUBindGroup | null = null;
    private resolveBindGroup: GPUBindGroup | null = null;

    private frameCount = 0;
    private readonly envTexture: GPUTexture;

    constructor({ device, envTexture }: { device: GPUDevice; envTexture: GPUTexture }) {
        this.device = device;
        this.envTexture = envTexture;

        // PTUniforms: invViewProjMat(64) + frame(4) + num_tris(4) + out_w(4) + out_h(4) = 80
        this.ptUniformsBuffer = device.createBuffer({
            label: "pt uniforms", size: 80,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.resolveUniformsBuffer = device.createBuffer({
            label: "pt resolve uniforms", size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const ptModule = device.createShaderModule({ label: "path trace", code: pathTraceSrc });
        ptModule.getCompilationInfo().then(info => {
            for (const m of info.messages) console.warn(`[path_trace] ${m.type}: ${m.message} (line ${m.lineNum})`);
        });
        const resolveModule = device.createShaderModule({ label: "pt resolve", code: resolveSrc });
        resolveModule.getCompilationInfo().then(info => {
            for (const m of info.messages) console.warn(`[pt_resolve] ${m.type}: ${m.message} (line ${m.lineNum})`);
        });

        // binding 2 = bvh_nodes, binding 3 = bvh_tris (was raw indices)
        this.ptBindGroupLayout = device.createBindGroupLayout({
            label: "path trace bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // vertices
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // bvh_nodes
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // bvh_tris
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // accum
                { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } },
            ],
        });
        this.resolveBindGroupLayout = device.createBindGroupLayout({
            label: "pt resolve bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba8unorm" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
        });

        this.ptPipeline = device.createComputePipeline({
            label: "path trace pipeline",
            layout: device.createPipelineLayout({ 
                label: "path trace pipeline layout",
                bindGroupLayouts: [this.ptBindGroupLayout] 
            }),
            compute: { module: ptModule, entryPoint: "main" },
        });
        this.resolvePipeline = device.createComputePipeline({
            label: "pt resolve pipeline",
            layout: device.createPipelineLayout({ 
                label: "pt resolve pipeline layout",
                bindGroupLayouts: [this.resolveBindGroupLayout] 
            }),
            compute: { module: resolveModule, entryPoint: "main" },
        });
    }

    setMeshes(meshes: MeshData[]) {
        const VSTRIDE = 12;
        let totalVerts = 0;
        let totalTris  = 0;
        for (const m of meshes) {
            totalVerts += m.vertices.length / VSTRIDE;
            totalTris  += m.indices.length / 3;
        }

        const allVerts = new Float32Array(totalVerts * VSTRIDE);
        const allIdx   = new Uint32Array(totalTris * 3);

        let vOff = 0, iOff = 0;
        for (const m of meshes) {
            const mv = m.vertices.length / VSTRIDE;
            allVerts.set(m.vertices, vOff * VSTRIDE);
            for (let i = 0; i < m.indices.length; i++) allIdx[iOff + i] = m.indices[i] + vOff;
            vOff += mv;
            iOff += m.indices.length;
        }

        // Build BVH on CPU
        console.time("bvh build");
        const bvh = buildBvh(allVerts, allIdx);
        console.timeEnd("bvh build");
        console.log(`BVH: ${bvh.nodes.length / 8} nodes, ${bvh.triIndices.length / 3} tris`);

        if (this.vertexBuffer)  this.vertexBuffer.destroy();
        if (this.bvhNodeBuffer) this.bvhNodeBuffer.destroy();
        if (this.bvhTriBuffer)  this.bvhTriBuffer.destroy();

        this.vertexBuffer = this.device.createBuffer({
            label: "pt vertices", size: allVerts.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.vertexBuffer, 0, allVerts);

        this.bvhNodeBuffer = this.device.createBuffer({
            label: "bvh nodes", size: bvh.nodes.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.bvhNodeBuffer, 0, bvh.nodes);

        this.bvhTriBuffer = this.device.createBuffer({
            label: "bvh tris", size: bvh.triIndices.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(this.bvhTriBuffer, 0, bvh.triIndices);
        this.numTris = bvh.triIndices.length / 3; // store reordered tri count

        this.ptBindGroup = null;
    }

    setOutputSize(width: number, height: number) {
        if (width === this.accumWidth && height === this.accumHeight) return;
        this.accumWidth  = width;
        this.accumHeight = height;

        if (this.accumBuffer)  this.accumBuffer.destroy();
        if (this.outputTexture) this.outputTexture.destroy();

        this.accumBuffer = this.device.createBuffer({
            label: "pt accum", size: width * height * 4 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.outputTexture = this.device.createTexture({
            label: "pt output", size: [width, height], format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.outputTextureView = this.outputTexture.createView({ label: "pt output view" });

        this.device.queue.writeBuffer(this.resolveUniformsBuffer, 0, new Uint32Array([width, height, 0, 0]));
        this.ptBindGroup = this.resolveBindGroup = null;
        this.frameCount = 0;
    }

    reset() {
        if (!this.accumBuffer) return;
        // clearBuffer zeros the entire buffer efficiently on GPU
        const encoder = this.device.createCommandEncoder({ label: "pt reset" });
        encoder.clearBuffer(this.accumBuffer);
        this.device.queue.submit([encoder.finish()]);
        this.frameCount = 0;
    }

    writeInvViewProjMat(mat: Float32Array) {
        this.device.queue.writeBuffer(this.ptUniformsBuffer, 0, mat.buffer, mat.byteOffset, mat.byteLength);
    }

    private rebuildBindGroups() {
        if (!this.vertexBuffer || !this.bvhNodeBuffer || !this.bvhTriBuffer ||
            !this.accumBuffer || !this.outputTextureView) return;

        this.ptBindGroup = this.device.createBindGroup({
            label: "path trace bind group",
            layout: this.ptBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.ptUniformsBuffer } },
                { binding: 1, resource: { buffer: this.vertexBuffer } },
                { binding: 2, resource: { buffer: this.bvhNodeBuffer } },
                { binding: 3, resource: { buffer: this.bvhTriBuffer } },
                { binding: 4, resource: { buffer: this.accumBuffer } },
                { binding: 5, resource: this.envTexture.createView({ label: "pt env texture view" }) },
                { binding: 6, resource: this.device.createSampler({
                    label: "pt sampler",
                    magFilter: "linear", minFilter: "linear",
                    addressModeU: "repeat", addressModeV: "clamp-to-edge",
                })},
            ],
        });
        this.resolveBindGroup = this.device.createBindGroup({
            label: "pt resolve bind group",
            layout: this.resolveBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.accumBuffer } },
                { binding: 1, resource: this.outputTextureView },
                { binding: 2, resource: { buffer: this.resolveUniformsBuffer } },
            ],
        });
    }

    dispatch(commandEncoder: GPUCommandEncoder) {
        if (!this.accumBuffer || !this.outputTextureView ||
            !this.vertexBuffer || !this.bvhNodeBuffer || !this.bvhTriBuffer) return;

        if (!this.ptBindGroup || !this.resolveBindGroup) this.rebuildBindGroups();
        if (!this.ptBindGroup || !this.resolveBindGroup) return;

        this.device.queue.writeBuffer(
            this.ptUniformsBuffer, 64,
            new Uint32Array([this.frameCount, this.numTris, this.accumWidth, this.accumHeight])
        );
        this.frameCount++;

        const w = this.accumWidth, h = this.accumHeight;
        const pass = commandEncoder.beginComputePass({ label: "path trace" });

        pass.setPipeline(this.ptPipeline);
        pass.setBindGroup(0, this.ptBindGroup);
        pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));

        pass.setPipeline(this.resolvePipeline);
        pass.setBindGroup(0, this.resolveBindGroup);
        pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));

        pass.end();
    }
}
