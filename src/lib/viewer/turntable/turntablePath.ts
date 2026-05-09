const TWO_PI = Math.PI * 2;

/** Camera orbit sample on the turntable animation path. */
export interface TurntableOrbitSample {
    long: number;
    lat: number;
    radius: number;
}

/** Sinusoidal lat/radius modulation parameters (see ViewerState UI). */
export interface TurntablePathParams {
    latCenter: number;
    latAmplitude: number;
    latCycles: number;
    radiusCenter: number;
    radiusAmplitude: number;
    radiusCycles: number;
}

/**
 * Evaluate the turntable path at normalized time t ∈ [0, 1].
 *
 * long(t) = baseLong + t * 2π
 * lat(t)  = latCenter + latAmplitude * sin(t * 2π * latCycles)
 * radius(t) = radiusCenter + radiusAmplitude * sin(t * 2π * radiusCycles)
 */
export function evaluateTurntablePath(
    t: number,
    baseLong: number,
    params: TurntablePathParams,
): TurntableOrbitSample {
    return {
        long: baseLong + t * TWO_PI,
        lat: params.latCenter + params.latAmplitude * Math.sin(t * TWO_PI * params.latCycles),
        radius:
            params.radiusCenter +
            params.radiusAmplitude * Math.sin(t * TWO_PI * params.radiusCycles),
    };
}
