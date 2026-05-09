import backwardModuleSrc from "./bezier_backward.wgsl?raw";
import stepModuleSrc from "./bezier_step.wgsl?raw";
import adcModuleSrc from "./bezier_adc.wgsl?raw";
import sortModuleSrc from "./bezier_sort.wgsl?raw";
import type { Mat4 } from "wgpu-matrix";
import { constants, injectWgslConstants } from "../constants";

// Each cubic bezier is 14 optimizable parameters but stored with 16-float
// stride (4 vec4f) so the WGSL struct lays out cleanly without per-field
// padding. Mirrors the splat manager's 11-params/12-floats layout.
const PARAMS_PER_BEZIER = 18;
const FLOATS_PER_BEZIER = 20;

// Optim resolution — must match GpuRunner's OPTIM_SHORT logic.
// We use the square short-side; the actual pixel count is written at runtime.
// The pixel_loss buffer is sized to the worst-case square (OPTIM_SHORT²).
const PIXEL_LOSS_MAX = constants.PIXEL_LOSS_MAX;

export class GpuBezierOptimizerManager {
    private readonly device: GPUDevice;

    readonly numBeziers: number;
    readonly numParams: number;

    readonly bezierBuffer: GPUBuffer;
    readonly gradBuffer: GPUBuffer;
    readonly adamBuffer: GPUBuffer;
    readonly adcBuffer: GPUBuffer;
    readonly bezierUniformsBuffer: GPUBuffer;
    readonly sortIndicesBuffer: GPUBuffer;
    private readonly sortKeysBuffer: GPUBuffer;
    private readonly sortUniformsBuffer: GPUBuffer;
    private readonly pixelLossBuffer: GPUBuffer;

    private readonly backwardPipeline: GPUComputePipeline;
    private readonly stepPipeline: GPUComputePipeline;
    private readonly adcPipeline: GPUComputePipeline;

    private readonly backwardBindGroupLayout: GPUBindGroupLayout;
    private readonly stepBindGroup: GPUBindGroup;
    private readonly adcBindGroup: GPUBindGroup;
    private readonly adcBindGroupLayout: GPUBindGroupLayout;

    private readonly adcScratchBuffer: GPUBuffer;
    private readonly sortInitPipeline: GPUComputePipeline;
    private readonly sortStepPipeline: GPUComputePipeline;
    private readonly sortInitBindGroup: GPUBindGroup;
    private readonly sortStepBindGroups: GPUBindGroup[];
    private readonly sortStepUniformBuffers: GPUBuffer[];
    private readonly sortN: number;

    private backwardBindGroup: GPUBindGroup | null = null;
    private stepCount: number = 0;
    private adcPeriod: number = constants.BEZIER_ADC_PERIOD;

    private dims: { width: number, height: number } = { width: 0, height: 0 };

    constructor({
        device,
        numBeziers = 16,
    }: {
        device: GPUDevice,
        numBeziers?: number,
    }) {
        this.device = device;
        this.numBeziers = numBeziers;
        this.numParams = numBeziers * PARAMS_PER_BEZIER;

        // Initialize curves as short, randomly oriented squiggles clustered
        // near the origin. Bright grayscale colors give them an immediate
        // contribution to the silhouette image they're trying to reconstruct.
        const data = new Float32Array(numBeziers * FLOATS_PER_BEZIER);
        for (let i = 0; i < numBeziers; i++) {
            const o = i * FLOATS_PER_BEZIER;
            const cx = (Math.random() * 2 - 1);
            const cy = (Math.random() * 2 - 1);
            const len = 0.2 + Math.random() * 0.2;
            const angle = Math.random() * Math.PI * 2;
            const dx = Math.cos(angle) * len;
            const dy = Math.sin(angle) * len;
            const jitter = () => (Math.random() - 0.5) * 0.08;
            // P0 (xyz, width)
            data[o + 0] = cx - dx * 0.5;
            data[o + 1] = cy - dy * 0.5;
            data[o + 2] = (Math.random() * 2 - 1) * 0.3;
            data[o + 3] = 0.02;
            // P1 (xyz, softness)
            data[o + 4] = cx - dx * 0.15 + jitter();
            data[o + 5] = cy - dy * 0.15 + jitter();
            data[o + 6] = (Math.random() * 2 - 1) * 0.3;
            data[o + 7] = 0.005;
            // P2 (xyz, pad)
            data[o + 8] = cx + dx * 0.15 + jitter();
            data[o + 9] = cy + dy * 0.15 + jitter();
            data[o + 10] = (Math.random() * 2 - 1) * 0.3;
            data[o + 11] = 0.0;
            // P3 (xyz, pad)
            data[o + 12] = cx + dx * 0.5;
            data[o + 13] = cy + dy * 0.5;
            data[o + 14] = (Math.random() * 2 - 1) * 0.3;
            data[o + 15] = 0.0;
            // color rgba
            data[o + 16] = Math.random();
            data[o + 17] = Math.random();
            data[o + 18] = Math.random();
            data[o + 19] = 0.5;
        }

        this.bezierBuffer = device.createBuffer({
            label: "bezier buffer",
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this.bezierBuffer, 0, data);

        this.gradBuffer = device.createBuffer({
            label: "bezier grad buffer",
            size: this.numParams * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // m + v + t + pad (matches splat adam layout for shader struct compatibility).
        this.adamBuffer = device.createBuffer({
            label: "bezier adam buffer",
            size: this.numParams * 8 + 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Two f32 per curve: positional gradient norm (grad_accum) and color
        // loss contribution (loss_accum), both accumulated across each ADC
        // period and reset to 0 inside the ADC shader.
        this.adcBuffer = device.createBuffer({
            label: "bezier adc buffer",
            size: this.numBeziers * 8,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.bezierUniformsBuffer = device.createBuffer({
            label: "bezier VP uniforms buffer",
            size: 160, // mat4x4f(64) + mode(4) + max_width(4) + prune_alpha(4) + prune_width(4) + bg_penalty(4) + pad(8) + adc_period_steps(4) + vp_inv(64)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(
            this.bezierUniformsBuffer,
            92,
            new Float32Array([this.adcPeriod]),
        );

        this.pixelLossBuffer = device.createBuffer({
            label: "bezier pixel loss buffer",
            size: PIXEL_LOSS_MAX * 4, // one i32 per pixel
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this.adcScratchBuffer = device.createBuffer({
            label: "bezier adc scratch buffer",
            size: this.numBeziers * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        // Sort Buffers
        let n = 1;
        while (n < this.numBeziers) n <<= 1;
        this.sortN = n;

        this.sortKeysBuffer = device.createBuffer({
            label: "bezier sort keys",
            size: this.sortN * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.sortIndicesBuffer = device.createBuffer({
            label: "bezier sort indices",
            size: this.sortN * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.sortUniformsBuffer = device.createBuffer({
            label: "bezier sort uniforms",
            size: 80, // vp(64) + block_k(4) + sub_k(4) + pad(8)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // NUM_BEZIERS_PLUS_ONE / NUM_BEZIERS_MINUS_ONE must come before
        // NUM_BEZIERS for the same substring reason as the splat shaders.
        // OPTIM_WIDTH/HEIGHT are injected at dispatch time when dims are known;
        // PIXEL_LOSS_SIZE = OPTIM_WIDTH * OPTIM_HEIGHT is also injected then.
        // For the shader module we use placeholder values that get replaced
        // via a separate per-dispatch inject — here we bake in the max size
        // so the buffer declaration compiles. The actual dims are written via
        // writeOptimDims() before the first dispatch.
        const inject = (src: string, ow = 256, oh = 256) => {
            return injectWgslConstants(src, {
                ...constants,
                NUM_BEZIERS: this.numBeziers,
                NUM_BEZIERS_PLUS_ONE: this.numBeziers + 1,
                NUM_BEZIERS_MINUS_ONE: this.numBeziers - 1,
                NUM_BEZIERS_DIV_32: Math.ceil(this.numBeziers / 32),
                NUM_BEZIER_PARAMS: this.numParams,
                PIXEL_LOSS_SIZE: PIXEL_LOSS_MAX,
                SORT_N: this.sortN,
                OPTIM_WIDTH: ow,
                OPTIM_HEIGHT: oh,
            });
        };

        this.backwardBindGroupLayout = device.createBindGroupLayout({
            label: "bezier backward bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 8, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            ],
        });
        const backwardModule = device.createShaderModule({
            label: "bezier backward",
            code: inject(backwardModuleSrc),
        });
        backwardModule.getCompilationInfo().then(info => {
            for (const m of info.messages) console.warn(`[bezier_backward] ${m.type}: ${m.message} (line ${m.lineNum})`);
        });
        this.backwardPipeline = device.createComputePipeline({
            label: "bezier backward pipeline",
            layout: device.createPipelineLayout({ 
                label: "bezier backward pipeline layout",
                bindGroupLayouts: [this.backwardBindGroupLayout] 
            }),
            compute: { module: backwardModule, entryPoint: "main" },
        });

        const stepBindGroupLayout = device.createBindGroupLayout({
            label: "bezier step bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
        });
        const stepModule = device.createShaderModule({
            label: "bezier step",
            code: inject(stepModuleSrc),
        });
        stepModule.getCompilationInfo().then(info => {
            for (const m of info.messages) console.warn(`[bezier_step] ${m.type}: ${m.message} (line ${m.lineNum})`);
        });
        this.stepPipeline = device.createComputePipeline({
            label: "bezier step pipeline",
            layout: device.createPipelineLayout({ 
                label: "bezier step pipeline layout",
                bindGroupLayouts: [stepBindGroupLayout] 
            }),
            compute: { module: stepModule, entryPoint: "main" },
        });

        this.stepBindGroup = device.createBindGroup({
            label: "bezier step bind group",
            layout: stepBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bezierBuffer } },
                { binding: 1, resource: { buffer: this.gradBuffer } },
                { binding: 2, resource: { buffer: this.adamBuffer } },
                { binding: 3, resource: { buffer: this.adcBuffer } },
                { binding: 4, resource: { buffer: this.bezierUniformsBuffer } },
            ],
        });

        // ADC pipeline: clones/splits high-gradient curves into dead slots.
        this.adcBindGroupLayout = device.createBindGroupLayout({
            label: "bezier adc bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
            ],
        });
        const adcModule = device.createShaderModule({
            label: "bezier adc",
            code: inject(adcModuleSrc),
        });
        adcModule.getCompilationInfo().then(info => {
            for (const m of info.messages) console.warn(`[bezier_adc] ${m.type}: ${m.message} (line ${m.lineNum})`);
        });
        this.adcPipeline = device.createComputePipeline({
            label: "bezier adc pipeline",
            layout: device.createPipelineLayout({ 
                label: "bezier adc pipeline layout",
                bindGroupLayouts: [this.adcBindGroupLayout] 
            }),
            compute: { module: adcModule, entryPoint: "main" },
        });

        this.adcBindGroup = device.createBindGroup({
            label: "bezier adc bind group",
            layout: this.adcBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bezierBuffer } },
                { binding: 1, resource: { buffer: this.adamBuffer } },
                { binding: 2, resource: { buffer: this.adcBuffer } },
                { binding: 3, resource: { buffer: this.pixelLossBuffer } },
                { binding: 4, resource: { buffer: this.bezierUniformsBuffer } },
                { binding: 5, resource: { buffer: this.adcScratchBuffer } },
            ],
        });

        // Sort pipelines
        const sortBindGroupLayout = device.createBindGroupLayout({
            label: "bezier sort bind group layout",
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            ],
        });
        const sortModule = device.createShaderModule({ label: "bezier sort", code: inject(sortModuleSrc) });
        sortModule.getCompilationInfo().then(info => {
            for (const msg of info.messages) console.warn(`[bezier_sort] ${msg.type}: ${msg.message} (line ${msg.lineNum})`);
        });
        this.sortInitPipeline = device.createComputePipeline({
            label: "bezier sort init pipeline",
            layout: device.createPipelineLayout({
                label: "bezier sort init pipeline layout",
                bindGroupLayouts: [sortBindGroupLayout],
            }),
            compute: { module: sortModule, entryPoint: "init_keys" },
        });
        this.sortStepPipeline = device.createComputePipeline({
            label: "bezier sort step pipeline",
            layout: device.createPipelineLayout({
                label: "bezier sort step pipeline layout",
                bindGroupLayouts: [sortBindGroupLayout],
            }),
            compute: { module: sortModule, entryPoint: "sort_step" },
        });

        this.sortInitBindGroup = device.createBindGroup({
            label: "bezier sort init bind group",
            layout: sortBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bezierBuffer } },
                { binding: 1, resource: { buffer: this.sortKeysBuffer } },
                { binding: 2, resource: { buffer: this.sortIndicesBuffer } },
                { binding: 3, resource: { buffer: this.sortUniformsBuffer } },
            ],
        });

        const logN = Math.log2(this.sortN);
        this.sortStepBindGroups = [];
        this.sortStepUniformBuffers = [];

        for (let block_k = 1; block_k <= logN; block_k++) {
            for (let sub_k = block_k - 1; sub_k >= 0; sub_k--) {
                const buf = device.createBuffer({
                    label: `bezier sort step uniforms k=${block_k} j=${sub_k}`,
                    size: 80,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                });
                device.queue.writeBuffer(buf, 64, new Uint32Array([block_k, sub_k, 0, 0]));

                const bg = device.createBindGroup({
                    label: `bezier sort step bind group k=${block_k} j=${sub_k}`,
                    layout: sortBindGroupLayout,
                    entries: [
                        { binding: 0, resource: { buffer: this.bezierBuffer } },
                        { binding: 1, resource: { buffer: this.sortKeysBuffer } },
                        { binding: 2, resource: { buffer: this.sortIndicesBuffer } },
                        { binding: 3, resource: { buffer: buf } },
                    ],
                });

                this.sortStepUniformBuffers.push(buf);
                this.sortStepBindGroups.push(bg);
            }
        }
    }

    writeVPMatrix(mat: Float32Array | number[]) {
        this.device.queue.writeBuffer(
            this.bezierUniformsBuffer,
            0,
            (mat as Float32Array).buffer,
            (mat as Float32Array).byteOffset,
            (mat as Float32Array).byteLength
        );
    }

    writeVPInvMatrix(mat: Mat4) {
        // vp_inv is at offset 96 in BezierUniforms
        this.device.queue.writeBuffer(
            this.bezierUniformsBuffer,
            96,
            (mat as Float32Array).buffer,
            (mat as Float32Array).byteOffset,
            (mat as Float32Array).byteLength
        );
    }

    writeMode(mode: number = 0) {
        this.device.queue.writeBuffer(
            this.bezierUniformsBuffer,
            64,
            new Float32Array([mode])
        );
    }

    writeBgPenalty(weight: number = 0) {
        // Writes BezierUniforms.bg_penalty at offset 80.
        // Layout: vp(64) + mode(4) + max_width(4) + prune_alpha(4) + prune_width(4) = 80
        // 0 = disabled (base color layer), >0 = enabled (fine color layer).
        this.device.queue.writeBuffer(
            this.bezierUniformsBuffer,
            80,
            new Float32Array([weight])
        );
    }

    writeMaxWidth(maxWidth: number = 0) {
        // Writes into StepUniforms.max_width (offset 68). 0 = use default cap.
        this.device.queue.writeBuffer(
            this.bezierUniformsBuffer,
            68,
            new Float32Array([maxWidth])
        );
    }

    writeKillThresholds(alphaThresh: number = 0, widthThresh: number = 0) {
        // Writes prune_alpha_thresh (offset 72) and prune_width_thresh (offset 76).
        // 0 = use default (0.001). Set higher to kill more aggressively, lower for less.
        this.device.queue.writeBuffer(
            this.bezierUniformsBuffer,
            72,
            new Float32Array([alphaThresh, widthThresh])
        );
    }

    setAdcPeriod(period: number) {
        this.adcPeriod = period;
        this.device.queue.writeBuffer(
            this.bezierUniformsBuffer,
            92,
            new Float32Array([period]),
        );
    }

    writeNoKill(noKill: boolean) {
        // Writes adam.no_kill flag at offset numParams*8 + 8 (after t and pixel_count).
        this.device.queue.writeBuffer(
            this.adamBuffer,
            this.numParams * 8 + 8,
            new Float32Array([noKill ? 1.0 : 0.0])
        );
    }

    /**
     * Reset Adam momentum (m, v) and step counter (t) without touching the
     * bezier parameters themselves. Call this whenever the camera changes
     * during turntable training so stale cross-view momentum doesn't corrupt
     * the gradient step for the new viewpoint.
     *
     * AdamState layout: m[numParams * f32] | v[numParams * f32] | t(f32) | pixel_count(f32) | no_kill(f32) | pad(f32)
     */
    resetAdam() {
        // Zero m and v, reset t to 0. pixel_count and no_kill are written
        // separately each frame so we don't need to preserve them here.
        this.device.queue.writeBuffer(
            this.adamBuffer,
            0,
            new Float32Array(this.numParams * 2 + 1) // m + v + t, all zeros
        );
        // Reset ADC accumulators (grad_accum and loss_accum) so stale
        // per-view gradient norms don't trigger spurious clone/kill decisions.
        this.device.queue.writeBuffer(
            this.adcBuffer,
            0,
            new Float32Array(this.numBeziers * 2) // grad_accum + loss_accum
        );
        // Reset per-pixel residual loss map used for ADC seeding.
        this.device.queue.writeBuffer(
            this.pixelLossBuffer,
            0,
            new Int32Array(this.pixelLossBuffer.size / 4)
        );
    }

    setBackwardTarget(
        targetTextureView: GPUTextureView,
        targetDepthTextureView: GPUTextureView,
        bgColorTextureView: GPUTextureView,
        bgDepthTextureView: GPUTextureView,
        normalTextureView: GPUTextureView,
        width: number,
        height: number,
    ) {
        this.dims = { width, height };

        this.backwardBindGroup = this.device.createBindGroup({
            label: "bezier backward bind group",
            layout: this.backwardBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bezierBuffer } },
                { binding: 1, resource: { buffer: this.gradBuffer } },
                { binding: 2, resource: targetTextureView },
                { binding: 3, resource: targetDepthTextureView },
                { binding: 4, resource: { buffer: this.bezierUniformsBuffer } },
                { binding: 5, resource: bgColorTextureView },
                { binding: 6, resource: bgDepthTextureView },
                { binding: 7, resource: { buffer: this.adcBuffer } },
                { binding: 8, resource: normalTextureView },
                { binding: 9, resource: { buffer: this.pixelLossBuffer } },
                { binding: 10, resource: { buffer: this.sortIndicesBuffer } },
            ],
        });
    }

    dispatch(commandEncoder: GPUCommandEncoder) {
        if (!this.backwardBindGroup) return;

        // Update pixel count for normalization in the step shader.
        // AdamState layout: m [N], v [N], t [1], pixel_count [1], pad [2]
        const pixelCount = this.dims.width * this.dims.height;
        this.device.queue.writeBuffer(
            this.adamBuffer,
            this.numParams * 8 + 4,
            new Float32Array([pixelCount])
        );

        const pass = commandEncoder.beginComputePass({
            label: "bezier backward and step pass",
        });

        pass.setPipeline(this.backwardPipeline);
        pass.setBindGroup(0, this.backwardBindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.dims.width / 16), Math.ceil(this.dims.height / 16));

        pass.setPipeline(this.stepPipeline);
        pass.setBindGroup(0, this.stepBindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.numBeziers / 64));

        // ADC fires every adcPeriod steps; bezier_adc.wgsl reads the same value
        // from uniforms.adc_period_steps for grad/loss averaging.
        this.stepCount++;
        if (this.stepCount % this.adcPeriod === 0) {
            pass.setPipeline(this.adcPipeline);
            pass.setBindGroup(0, this.adcBindGroup);
            pass.dispatchWorkgroups(1);
        }

        pass.end();
    }

    dispatchSort(commandEncoder: GPUCommandEncoder, vpMat: Mat4) {
        const vpData = vpMat as Float32Array;

        this.device.queue.writeBuffer(
            this.sortUniformsBuffer, 0,
            vpData.buffer, vpData.byteOffset, vpData.byteLength,
        );

        for (const buf of this.sortStepUniformBuffers) {
            this.device.queue.writeBuffer(
                buf, 0,
                vpData.buffer, vpData.byteOffset, vpData.byteLength,
            );
        }

        const wg = Math.ceil(this.sortN / 256);

        const initPass = commandEncoder.beginComputePass({ label: "bezier sort init pass" });
        initPass.setPipeline(this.sortInitPipeline);
        initPass.setBindGroup(0, this.sortInitBindGroup);
        initPass.dispatchWorkgroups(wg);
        initPass.end();

        const stepWg = Math.ceil(this.sortN / 2 / 256);
        for (let i = 0; i < this.sortStepBindGroups.length; i++) {
            const stepPass = commandEncoder.beginComputePass({ label: `bezier sort step ${i}` });
            stepPass.setPipeline(this.sortStepPipeline);
            stepPass.setBindGroup(0, this.sortStepBindGroups[i]);
            stepPass.dispatchWorkgroups(stepWg);
            stepPass.end();
        }
    }

    destroy() {
        this.bezierBuffer.destroy();
        this.gradBuffer.destroy();
        this.adamBuffer.destroy();
        this.adcBuffer.destroy();
        this.adcScratchBuffer.destroy();
        this.bezierUniformsBuffer.destroy();
        this.pixelLossBuffer.destroy();
        this.sortKeysBuffer.destroy();
        this.sortIndicesBuffer.destroy();
        this.sortUniformsBuffer.destroy();
        for (const buf of this.sortStepUniformBuffers) {
            buf.destroy();
        }
    }
}
