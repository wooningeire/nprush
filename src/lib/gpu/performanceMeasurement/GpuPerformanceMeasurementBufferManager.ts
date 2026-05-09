import { GPU_PROFILER_PAIR_COUNT } from "./gpuProfilerPairs";

/**
 * WebGPU timestamp queries (needs `timestamp-query` device feature).
 *
 * Mirrors the CIS565 GpuPerformanceMeasurementBufferManager pattern:
 * paired begin/end timestamps per encoder pass, resolve + map readback.
 *
 * Resolved values are elapsed GPU clock ticks between the two writes for that
 * pass; we surface them as `bigint` deltas (Chrome/Dawn treat these as ns).
 */
export class GpuPerformanceMeasurementBufferManager {
    readonly querySet: GPUQuerySet;
    readonly resolveBuffer: GPUBuffer;
    readonly resultBuffer: GPUBuffer;
    readonly pairCount: number;

    constructor({
        device,
        pairCount = GPU_PROFILER_PAIR_COUNT,
    }: {
        device: GPUDevice;
        pairCount?: number;
    }) {
        const queryCount = pairCount * 2;
        const byteSize = queryCount * 8;

        const querySet = device.createQuerySet({
            type: "timestamp",
            count: queryCount,
        });

        const resolveBuffer = device.createBuffer({
            label: "gpu perf resolve buffer",
            size: byteSize,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });

        const resultBuffer = device.createBuffer({
            label: "gpu perf readback buffer",
            size: byteSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        this.pairCount = pairCount;
        this.querySet = querySet;
        this.resolveBuffer = resolveBuffer;
        this.resultBuffer = resultBuffer;
    }

    writes(pairIndex: number): NonNullable<GPUComputePassDescriptor["timestampWrites"]> {
        const b = pairIndex * 2;
        return {
            querySet: this.querySet,
            beginningOfPassWriteIndex: b,
            endOfPassWriteIndex: b + 1,
        };
    }

    /** Append resolve+copy immediately before `finish()` while `resultBuffer` is unmapped. */
    addResolve(commandEncoder: GPUCommandEncoder) {
        if (this.resultBuffer.mapState !== "unmapped") return;

        const q = this.querySet.count * 8;
        commandEncoder.resolveQuerySet(this.querySet, 0, this.querySet.count, this.resolveBuffer, 0);
        commandEncoder.copyBufferToBuffer(this.resolveBuffer, 0, this.resultBuffer, 0, q);
    }

    /**
     * Per-pair deltas (newest end − newest begin). Runs after queue submit so
     * the GPU finishes recording query values before resolve.
     */
    async mapDeltasNanoseconds(): Promise<bigint[]> {
        if (this.resultBuffer.mapState === "pending") {
            return Array.from({ length: this.pairCount }, () => 0n);
        }

        await this.resultBuffer.mapAsync(GPUMapMode.READ);
        try {
            const raw = new BigUint64Array(this.resultBuffer.getMappedRange());
            const out = new Array<bigint>(this.pairCount);
            for (let i = 0; i < this.pairCount; i++) {
                const begin = raw[i * 2];
                const end = raw[i * 2 + 1];
                out[i] = end >= begin ? end - begin : 0n;
            }
            return out;
        } finally {
            this.resultBuffer.unmap();
        }
    }

    destroy() {
        this.querySet.destroy?.();
        this.resolveBuffer.destroy();
        this.resultBuffer.destroy();
    }
}
