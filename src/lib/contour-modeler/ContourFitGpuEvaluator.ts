import type { ContourGpuFitEvaluator } from "./fitSolver.ts";
import {
    evaluateRenderedFeatureLoss,
    type RenderedFitTarget,
} from "./renderedFeatureLoss.ts";
import type { ImplicitBodyParams } from "./types.ts";

// The contour mode now fits against rendered feature maps. This class keeps the
// GPU-evaluator contract stable, but intentionally avoids submitting WebGPU
// compute from drawing/fitting until a bounded rendered-loss compute pipeline is
// wired in. The active path still runs off the main thread via the fit worker.
export class ContourFitGpuEvaluator implements ContourGpuFitEvaluator {
    static async create(): Promise<ContourFitGpuEvaluator> {
        return new ContourFitGpuEvaluator();
    }

    async evaluateCandidates(
        candidates: ImplicitBodyParams[],
        target: RenderedFitTarget,
    ): Promise<number[]> {
        await new Promise<void>(resolve => setTimeout(resolve, 0));
        return candidates.map(candidate => evaluateRenderedFeatureLoss(candidate, target));
    }

    destroy() {
        // No resources yet.
    }
}
