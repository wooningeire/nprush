import backwardModuleSrc from "./bezier_backward.wgsl?raw";
import stepModuleSrc from "./bezier_step.wgsl?raw";
import adcModuleSrc from "./bezier_adc.wgsl?raw";

// Each cubic bezier is 14 optimizable parameters but stored with 16-float
// stride (4 vec4f) so the WGSL struct lays out cleanly without per-field
// padding. Mirrors the splat manager's 11-params/12-floats layout.
const PARAMS_PER_BEZIER = 14;
const FLOATS_PER_BEZIER = 16;

export class GpuBezierOptimizerManager {
    private readonly device: GPUDevice;

    readonly numBeziers: number;
    readonly numParams: number;

    readonly bezierBuffer: GPUBuffer;
    readonly gradBuffer: GPUBuffer;
    readonly adamBuffer: GPUBuffer;
    readonly adcBuffer: GPUBuffer;

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
            const cx = (Math.random() * 2 - 1) * 0.5;
            const cy = (Math.random() * 2 - 1) * 0.5;
            const len = 0.2 + Math.random() * 0.2;
            const angle = Math.random() * Math.PI * 2;
            const dx = Math.cos(angle) * len;
            const dy = Math.sin(angle) * len;
            const jitter = () => (Math.random() - 0.5) * 0.08;
            // P0
            data[o + 0] = cx - dx * 0.5;
            data[o + 1] = cy - dy * 0.5;
            // P1
            data[o + 2] = cx - dx * 0.15 + jitter();
            data[o + 3] = cy - dy * 0.15 + jitter();
            // P2
            data[o + 4] = cx + dx * 0.15 + jitter();
            data[o + 5] = cy + dy * 0.15 + jitter();
            // P3
            data[o + 6] = cx + dx * 0.5;
            data[o + 7] = cy + dy * 0.5;
            // color rgba
            data[o + 8] = 0.7 + Math.random() * 0.3;
            data[o + 9] = 0.7 + Math.random() * 0.3;
            data[o + 10] = 0.7 + Math.random() * 0.3;
            data[o + 11] = 0.6;
            // width, softness, pad, pad. Bezier coordinates are normalized
            // [-1, 1] so 1 norm unit ~= dims.y/2 pixels at any resolution.
            // We optimize at 128 px (~64 px per norm unit) but visualize at
            // ~500 px panels (~250 px per norm unit). The width therefore
            // appears ~4x larger on screen than during training. Picking
            // values ~half what they look "right" at training resolution
            // keeps the display panel from rendering fat strokes.
            data[o + 12] = 0.008;
            data[o + 13] = 0.003;
            data[o + 14] = 0.0;
            data[o + 15] = 0.0;
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

        // One f32 per curve: positional gradient norm accumulated across each
        // ADC period; reset to 0 inside the ADC shader.
        this.adcBuffer = device.createBuffer({
            label: "bezier adc buffer",
            size: this.numBeziers * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // NUM_BEZIERS_PLUS_ONE / NUM_BEZIERS_MINUS_ONE must come before
        // NUM_BEZIERS for the same substring reason as the splat shaders.
        const inject = (src: string) => src
            .replace(/NUM_BEZIERS_PLUS_ONE/g, `${numBeziers + 1}u`)
            .replace(/NUM_BEZIERS_MINUS_ONE/g, `${numBeziers - 1}u`)
            .replace(/NUM_BEZIERS/g, `${numBeziers}u`)
            .replace(/NUM_BEZIER_PARAMS/g, `${this.numParams}u`);

        this.backwardBindGroupLayout = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float" } },
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

    setBackwardTarget(
        targetTextureView: GPUTextureView,
        targetEdgeTextureView: GPUTextureView,
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
                { binding: 3, resource: targetEdgeTextureView },
            ],
        });
    }

    dispatch(commandEncoder: GPUCommandEncoder) {
        if (!this.backwardBindGroup) return;

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
        if (this.stepCount % 200 === 0) {
            pass.setPipeline(this.adcPipeline);
            pass.setBindGroup(0, this.adcBindGroup);
            pass.dispatchWorkgroups(1);
        }

        pass.end();
    }
}
