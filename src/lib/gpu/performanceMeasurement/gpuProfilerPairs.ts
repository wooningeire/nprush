/**
 * Indices into the timestamp query pairs (each pair consumes two timestamps:
 * beginning + end of the same pass).
 *
 * Order must stay aligned with GPU_PROFILER_LABELS entries.
 *
 * Note: multipass bitonic sort (splats + beziers) is not broken out separately;
 * its cost rides along adjacent passes until we thread timers through sort steps.
 */
export const GpuProfilingPair = {
    MeshFullRaster: 0,
    MeshOptimRaster: 1,
    PathTrace: 2,
    BlurOptimTarget: 3,
    BlurOptimDepth: 4,
    DepthAwareBlur: 5,
    SplatEdgeDetectOptim: 6,
    SplatBackwardStep: 7,
    SplatRasterOptim: 8,
    BezierEdgeOptim: 9,
    BezierCoarseOptim: 10,
    BezierFineOptim: 11,
    SplatEdgeDetectFull: 12,
    SplatRasterFull: 13,
    BezierEdgeRasterFull: 14,
    BezierCoarseRasterFull: 15,
    BezierFineRasterFull: 16,
    FinalCompositor: 17,
} as const satisfies Record<string, number>;

/** Number of pass pairs (= half the timestamp-query count). */
export const GPU_PROFILER_PAIR_COUNT = 18;

/** Rolling frame history length for profiler charts / HUD sparkline */
export const GPU_PROFILER_HISTORY_FRAMES = 180;

export const GPU_PROFILER_LABELS: readonly string[] = [
    "Mesh raster — full viewport",
    "Mesh raster — optim target",
    "Path trace (compute)",
    "Separable blur — optim color",
    "Separable blur — optim depth",
    "Depth-aware blur",
    "Splat silhouette edge — optim res",
    "Splat backward + Adam step",
    "Splat forward — optim res",
    "Bézier edge layer — backward + step",
    "Bézier coarse — backward + step",
    "Bézier fine — backward + step",
    "Splat silhouette edge — full res",
    "Splat forward — full viewport",
    "Bézier edge forward — full res",
    "Bézier coarse forward — full res",
    "Bézier fine forward — full res",
    "Final compositor (screen)",
];
