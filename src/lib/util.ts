export const PI = Math.PI;
export const REV = 2 * PI;
export const PI_2 = PI / 2;

// Fraction of canvas height used by the debug strip at the bottom of the viewer.
// Must match STRIP_HEIGHT in src/lib/gpu/splat_render.wgsl.
export const STRIP_HEIGHT_FRAC = 0.18;

export const mod = (a: number, b: number) => {
    const remainder = a % b;

    if (remainder < 0) {
        return remainder + b;
    }
    return remainder;
};