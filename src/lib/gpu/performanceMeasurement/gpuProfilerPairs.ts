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
    SplatOptimization: 7,
    BezierCoarseOptimization: 8,
    BezierFineOptimization: 9,
    BezierEdgeOptimization: 10,
    SplatForwardOptim: 11,
    SplatForwardFull: 12,
    BezierEdgeForwardFull: 13,
    BezierCoarseForwardFull: 14,
    BezierFineForwardFull: 15,
    EdgeDetectFull: 16,
    FinalCompositor: 17,
} as const satisfies Record<string, number>;

/** Number of pass pairs (= half the timestamp-query count). */
export const GPU_PROFILER_PAIR_COUNT = 24;

/** Rolling frame history length for profiler charts / HUD sparkline */
export const GPU_PROFILER_HISTORY_FRAMES = 180;

export const GPU_PROFILER_LABELS: readonly string[] = [
    "Mesh: full viewport",
    "Mesh: optim target",
    "Path trace (compute)",
    "Blur: optim color",
    "Blur: optim depth",
    "Blur: depth-aware",
    "Splat: edge detect (optim)",
    "Splat: optimization",
    "Bézier (coarse): optimization",
    "Bézier (fine): optimization",
    "Bézier (edge): optimization",
    "Splat: forward (optim res)",
    "Splat: forward (display res)",
    "Bézier (edge): forward (display res)",
    "Bézier (coarse): forward (display res)",
    "Bézier (fine): forward (display res)",
    "Edge detect (display res)",
    "Final compositor (screen)",
    "(unused 18)",
    "(unused 19)",
    "(unused 20)",
    "(unused 21)",
    "(unused 22)",
    "(unused 23)",
];
