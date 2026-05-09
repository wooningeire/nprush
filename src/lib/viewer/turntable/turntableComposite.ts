export interface TurntableCompositeLayers {
    splat: ImageData;
    baseColorBezier: ImageData | null;
    colorBezier: ImageData | null;
    edgeBezier: ImageData | null;
}

/**
 * Composite splat + bezier layers like splat_render.wgsl right-half (premultiplied over + edge lighten).
 */
export function compositeTurntableLayers(width: number, height: number, layers: TurntableCompositeLayers): ImageData {
    const { splat, baseColorBezier, colorBezier, edgeBezier } = layers;
    const result = new ImageData(width, height);
    const n = width * height;

    for (let i = 0; i < n; i++) {
        const o = i * 4;
        let r = splat.data[o] / 255;
        let g = splat.data[o + 1] / 255;
        let b = splat.data[o + 2] / 255;

        if (baseColorBezier) {
            const ba = baseColorBezier.data[o + 3] / 255;
            r = r * (1 - ba) + baseColorBezier.data[o] / 255;
            g = g * (1 - ba) + baseColorBezier.data[o + 1] / 255;
            b = b * (1 - ba) + baseColorBezier.data[o + 2] / 255;
        }

        if (colorBezier) {
            const ca = colorBezier.data[o + 3] / 255;
            r = r * (1 - ca) + colorBezier.data[o] / 255;
            g = g * (1 - ca) + colorBezier.data[o + 1] / 255;
            b = b * (1 - ca) + colorBezier.data[o + 2] / 255;
        }

        if (edgeBezier) {
            const e = Math.min(1, edgeBezier.data[o] / 255);
            r = r + (1 - r) * e;
            g = g + (1 - g) * e;
            b = b + (1 - b) * e;
        }

        result.data[o] = Math.min(255, Math.round(r * 255));
        result.data[o + 1] = Math.min(255, Math.round(g * 255));
        result.data[o + 2] = Math.min(255, Math.round(b * 255));
        result.data[o + 3] = 255;
    }

    return result;
}
