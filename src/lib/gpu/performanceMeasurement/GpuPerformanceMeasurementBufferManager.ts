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

    private labelToIndex = new Map<string, number>();

    constructor({
        device,
        pairCount = 64,
    }: {
        device: GPUDevice;
        pairCount?: number;
    }) {
        const queryCount = pairCount * 2;
        const resultByteSize = queryCount * 8;
        // WebGPU requires destinationOffset for resolveQuerySet to be a multiple of 256 bytes.
        // We allocate 256 bytes per pair so we can resolve each pair individually at an aligned offset.
        const resolveByteSize = pairCount * 256;

        const querySet = device.createQuerySet({
            type: "timestamp",
            count: queryCount,
        });

        const resolveBuffer = device.createBuffer({
            label: "gpu perf resolve buffer",
            size: resolveByteSize,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });

        const resultBuffer = device.createBuffer({
            label: "gpu perf readback buffer",
            size: resultByteSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        this.pairCount = pairCount;
        this.querySet = querySet;
        this.resolveBuffer = resolveBuffer;
        this.resultBuffer = resultBuffer;
    }

    getIndex(label: string): number {
        if (!this.labelToIndex.has(label)) {
            if (this.labelToIndex.size >= this.pairCount) {
                console.warn(`Exceeded max GPU profiling pairs (${this.pairCount})`);
                return 0; // fallback
            }
            this.labelToIndex.set(label, this.labelToIndex.size);
        }
        return this.labelToIndex.get(label)!;
    }

    getLabels(): string[] {
        const labels = new Array(this.labelToIndex.size);
        for (const [label, index] of this.labelToIndex.entries()) {
            labels[index] = label;
        }
        return labels;
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
    addResolve(commandEncoder: GPUCommandEncoder, activeIndices?: Set<number>) {
        if (this.resultBuffer.mapState !== "unmapped") return;

        if (activeIndices) {
            for (let i = 0; i < this.pairCount; i++) {
                if (activeIndices.has(i)) {
                    const startQuery = i * 2;
                    const resolveOffset = i * 256;
                    const resultOffset = i * 16;
                    commandEncoder.resolveQuerySet(this.querySet, startQuery, 2, this.resolveBuffer, resolveOffset);
                    commandEncoder.copyBufferToBuffer(this.resolveBuffer, resolveOffset, this.resultBuffer, resultOffset, 16);
                }
            }
        } else {
            const q = this.querySet.count * 8;
            commandEncoder.resolveQuerySet(this.querySet, 0, this.querySet.count, this.resolveBuffer, 0);
            commandEncoder.copyBufferToBuffer(this.resolveBuffer, 0, this.resultBuffer, 0, q);
        }
    }

    /**
     * Per-pair deltas (newest end − newest begin). Runs after queue submit so
     * the GPU finishes recording query values before resolve.
     */
    async mapDeltasNanoseconds(activeIndices?: Set<number>): Promise<(bigint | null)[] | null> {
        if (this.resultBuffer.mapState === "pending") return null;

        await this.resultBuffer.mapAsync(GPUMapMode.READ);
        try {
            const raw = new BigUint64Array(this.resultBuffer.getMappedRange());
            const out = new Array<bigint | null>(this.pairCount);
            for (let i = 0; i < this.pairCount; i++) {
                if (activeIndices && !activeIndices.has(i)) {
                    out[i] = null;
                    continue;
                }
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
