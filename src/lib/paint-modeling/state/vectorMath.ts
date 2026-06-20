import type { Vec3 } from "../types.ts";

export function add3(a: Vec3, b: Vec3): Vec3 {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function sub3(a: Vec3, b: Vec3): Vec3 {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function scale3(v: Vec3, scale: number): Vec3 {
    return [v[0] * scale, v[1] * scale, v[2] * scale];
}

export function dot3(a: Vec3, b: Vec3): number {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function cross3(a: Vec3, b: Vec3): Vec3 {
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
}

export function distance3(a: Vec3, b: Vec3): number {
    return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

export function normalize3(v: Vec3, fallback: Vec3): Vec3 {
    const length = Math.hypot(v[0], v[1], v[2]);
    if (length <= 1e-8) return fallback;
    return [v[0] / length, v[1] / length, v[2] / length];
}

export function lerp(a: number, b: number, t: number): number {
    return a + (b - a) * t;
}

export function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}