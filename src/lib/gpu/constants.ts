/**
 * Tunable constants for the GPU pipelines.
 * Extracted from WGSL and Manager files to allow easier experimentation.
 */
export const GPU_CONSTANTS = {
    // Shared / Global
    ADAM_BETA1: 0.9,
    ADAM_BETA2: 0.999,
    ADAM_EPS: 1e-8,
    PIXEL_LOSS_MAX: 512 * 512,

    // Bezier ADC & Optimization
    BEZIER_ADC_PERIOD: 50,
    BEZIER_TAU_POS: 0.0002,      // must be moving to clone
    BEZIER_TAU_LOSS: 0.001,      // kill if stuck AND contributing to loss
    BEZIER_SPLIT_LEN_THRESHOLD: 0.25,
    BEZIER_MAX_SPAWNS: 512,
    BEZIER_SPAWN_TANGENT_LEN: 0.025,
    BEZIER_SPAWN_WIDTH: 0.015,
    BEZIER_SPAWN_SOFTNESS: 0.005,
    BEZIER_OFFSCREEN_MARGIN: 1.2,
    BEZIER_KILL_ALPHA_THRESH: 0.005,
    BEZIER_PRUNE_ALPHA_DEFAULT: 0.001,
    BEZIER_PRUNE_WIDTH_DEFAULT: 0.001,
    BEZIER_FP_SCALE_POS: 10000.0,
    BEZIER_FP_SCALE_COL: 100000.0,
    BEZIER_MAX_TILE_BEZIERS: 1024,

    // Splat ADC & Optimization
    SPLAT_ADC_PERIOD: 25,
    SPLAT_GRAD_THRESH: 0.00005,
    SPLAT_OPACITY_KILL_THRESH: 0.05,
    SPLAT_MAX_SPAWNS: 1024,
    SPLAT_VOLUME_KILL_THRESH: 1e-7,
    SPLAT_MAX_TILE_SPLATS: 1024,
    SPLAT_FP_SCALE_POS: 10000.0,
    SPLAT_FP_SCALE_COL: 100000.0,
    SPLAT_RENDER_STRIP_HEIGHT: 0.18,
    SPLAT_RENDER_NUM_PANELS: 6.0,
    SPLAT_EDGE_THRESHOLD_MIN: 0.02,
    SPLAT_EDGE_THRESHOLD_MAX: 0.06,
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
