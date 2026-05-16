export const constants = {
    // Shared / Global
    ADAM_BETA1: 0.9,
    ADAM_BETA2: 0.999,
    ADAM_EPS: 1e-8,
    PIXEL_LOSS_MAX: 512 * 512,

    // Bezier ADC & Optimization
    BEZIER_ADC_PERIOD: 25,
    BEZIER_TAU_POS: 0.0002,      // must be moving to clone
    BEZIER_TAU_LOSS: 0.001,      // kill if stuck AND contributing to loss
    BEZIER_TAU_EDGE_SUPPORT: 0.001, // edge layer: kill static curves without target-edge support
    BEZIER_SPLIT_LEN_THRESHOLD: 0.25,
    BEZIER_MAX_SPAWNS: 512,
    BEZIER_SPAWN_TANGENT_LEN: 0.025,
    BEZIER_EDGE_SPAWN_HALF_LEN_PX: 2.0,
    BEZIER_SPAWN_WIDTH: 0.015,
    BEZIER_SPAWN_SOFTNESS: 0.005,
    BEZIER_OFFSCREEN_MARGIN: 1.01,
    BEZIER_KILL_ALPHA_THRESH: 0.005,
    BEZIER_PRUNE_ALPHA_DEFAULT: 0.001,
    BEZIER_PRUNE_WIDTH_DEFAULT: 0.001,
    BEZIER_FP_SCALE_POS: 10000.0,
    BEZIER_FP_SCALE_COL: 100000.0,
    BEZIER_MAX_TILE_BEZIERS: 4096,
    /**
     * Screen-space polyline resolution for Bézier distance (fragment loop + backward pass).
     * Keep modest: each forward pixel still pays O(N_SEG) inside the hull quad.
     */
    BEZIER_POLY_SEG: 4,
    /** Optimizer params per curve (XYZ×4 CPs + RGBA + width/soft + degree-1 SH×9 + opacity SH×3). */
    BEZIER_PARAMS_PER: 30,
    /** GPU storage floats per curve (9 × vec4: CPs + color + RGB SH + opacity SH). */
    BEZIER_FLOATS_PER: 36,

    // Splat ADC & Optimization
    SPLAT_ADC_PERIOD: 25,
    SPLAT_GRAD_THRESH: 0.00005,
    SPLAT_OPACITY_KILL_THRESH: 0.05,
    SPLAT_MAX_SPAWNS: 1024,
    SPLAT_VOLUME_KILL_THRESH: 1e-4,
    SPLAT_MAX_TILE_SPLATS: 4096,
    SPLAT_FP_SCALE_POS: 10000.0,
    SPLAT_FP_SCALE_COL: 100000.0,
    SPLAT_RENDER_NUM_PANELS: 6.0,
    SPLAT_EDGE_THRESHOLD: 0.01,        // depth Laplacian: step H fires at ~H
    SPLAT_EDGE_NORMAL_THRESHOLD: 1.5,  // normal Laplacian: ~0.52 at 30°, ~1.41 at 90°
    
    OPTIMIZATION_SHORT: 128,
    /** Floats stored per Gaussian (pos, color DC, quat, shape, RGB + opacity degree-1 SH, vec4-packed). */
    SPLAT_PARAMS_PER_SPLAT: 32,
    N_GAUSSIAN_SPLATS: 512,
    N_EDGE_BEZIERS: 1024,
    N_COARSE_COLOR_BEZIERS: 2048,
    N_FINE_COLOR_BEZIERS: 16384,
    
    MESH_SPLAT_MAX_COUNT: 4096,
    
    BLUR_SIGMA_S: 10.0,
    BLUR_SIGMA_C: 0.1,
    BLUR_SIGMA_D: 0.005,
    BLUR_SIGMA_N: 0.05,
};

/**
 * Replaces placeholders in the style of {@ABC_DEF_123} with values from an object.
 */
export const injectWgslConstants = (src: string, substitutions: Record<string, any>): string => {
    return src.replace(/{@([A-Z0-9_]+)}/g, (match, key) => {
        if (Object.hasOwn(substitutions, key)) {
            return String(substitutions[key]);
        }
        console.warn(`WGSL injection: ${key} not defined`);
        return match;
    });
};
