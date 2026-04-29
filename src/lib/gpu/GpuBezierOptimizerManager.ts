import backwardModuleSrc from "./bezier_backward.wgsl?raw";
import stepModuleSrc from "./bezier_step.wgsl?raw";
import adcModuleSrc from "./bezier_adc.wgsl?raw";

// Each cubic bezier is 14 optimizable parameters but stored with 16-float
// stride (4 vec4f) so the WGSL struct lays out cleanly without per-field
// padding. Mirrors the splat manager's 11-params/12-floats layout.
const PARAMS_PER_BEZIER = 18;
const FLOATS_PER_BEZIER = 20;

export class GpuBezierOptimizerManager {
    private readonly device: GPUDevice;

    readonly numBeziers: number;
    readonly numParams: number;

    readonly bezierBuffer: GPUBuffer;
    readonly gradBuffer: GPUBuffer;
    readonly adamBuffer: GPUBuffer;
    readonly adcBuffer: GPUBuffer;
    readonly bezierUniformsBuffer: GPUBuffer;

    private readonly backwardPipeline: GPUComputePipeline;
    private readonly stepPipeline: GPUComputePipeline;
    private readonly adcPipeline: GPUComputePipeline;

    private readonly backwardBindGroupLayout: GPUBindGroupLayout;
    private readonly stepBindGroup: GPUBindGroup;
    private readonly adcBindGroup: GPUBindGroup;

    private backwardBindGroup: GPUBindGroup | null = null;
    private stepCount: number = 0;

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
            size: 96, // mat4x4f (64) + f32 (4) + align(16) + vec3f (12) -> size 96
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // NUM_BEZIERS_PLUS_ONE / NUM_BEZIERS_MINUS_ONE must come before
        // NUM_BEZIERS for the same substring reason as the splat shaders.
        const inject = (src: string) => src
            .replace(/NUM_BEZIERS_PLUS_ONE/g, `${numBeziers + 1}u`)
            .replace(/NUM_BEZIERS_MINUS_ONE/g, `${numBeziers - 1}u`)
            .replace(/NUM_BEZIERS_DIV_32/g, `${Math.ceil(numBeziers / 32)}u`)
            .replace(/NUM_BEZIERS/g, `${numBeziers}u`)
            .replace(/NUM_BEZIER_PARAMS/g, `${this.numParams}u`);

        this.backwardBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
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
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.backwardBindGroupLayout] }),
            compute: { module: backwardModule, entryPoint: "main" },
        });

        const stepBindGroupLayout = device.createBindGroupLayout({
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
            layout: device.createPipelineLayout({ bindGroupLayouts: [stepBindGroupLayout] }),
            compute: { module: stepModule, entryPoint: "main" },
        });

        this.stepBindGroup = device.createBindGroup({
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
        const adcBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
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
            layout: device.createPipelineLayout({ bindGroupLayouts: [adcBindGroupLayout] }),
            compute: { module: adcModule, entryPoint: "main" },
        });

        this.adcBindGroup = device.createBindGroup({
            layout: adcBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.bezierBuffer } },
                { binding: 1, resource: { buffer: this.adamBuffer } },
                { binding: 2, resource: { buffer: this.adcBuffer } },
            ],
        });
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

    writeMode(mode: number = 0) {
        this.device.queue.writeBuffer(
            this.bezierUniformsBuffer,
            64,
            new Float32Array([mode])
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

    writeNoKill(noKill: boolean) {
        // Writes adam.no_kill flag at offset numParams*8 + 8 (after t and pixel_count).
        this.device.queue.writeBuffer(
            this.adamBuffer,
            this.numParams * 8 + 8,
            new Float32Array([noKill ? 1.0 : 0.0])
        );
    }

    setBackwardTarget(
        targetTextureView: GPUTextureView,
        targetDepthTextureView: GPUTextureView,
        bgColorTextureView: GPUTextureView,
        bgDepthTextureView: GPUTextureView,
        width: number,
        height: number,
    ) {
        this.dims = { width, height };

        this.backwardBindGroup = this.device.createBindGroup({
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

        const pass = commandEncoder.beginComputePass();

        pass.setPipeline(this.backwardPipeline);
        pass.setBindGroup(0, this.backwardBindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.dims.width / 16), Math.ceil(this.dims.height / 16));

        pass.setPipeline(this.stepPipeline);
        pass.setBindGroup(0, this.stepBindGroup);
        pass.dispatchWorkgroups(Math.ceil(this.numBeziers / 64));

        // ADC fires every ADC_PERIOD steps. Period must match the ADC_PERIOD
        // constant inside bezier_adc.wgsl (used there as the grad_accum
        // averaging divisor). Less frequent ADC reduces churn so transient
        // strays from clone+parent edge competition don't accumulate.
        this.stepCount++;
        if (this.stepCount % 100 === 0) {
            pass.setPipeline(this.adcPipeline);
            pass.setBindGroup(0, this.adcBindGroup);
            pass.dispatchWorkgroups(1);
        }

        pass.end();
    }
}
