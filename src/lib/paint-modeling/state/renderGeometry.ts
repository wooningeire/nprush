import type {
    PaintRibbon,
    PaintStroke,
    RenderPrimitive,
    Vec3,
    Vec4,
} from "../types.ts";

export const appendStrokeRenderRibbon = (
    segments: RenderPrimitive[],
    stroke: PaintStroke,
    shadeRibbons: boolean,
) => {
    appendRibbonRenderPrimitive(
        segments,
        stroke.ribbon,
        parseColor(stroke.style.color, stroke.style.opacity),
        shadeRibbons ? 1 : 0,
    );
};

export const appendRibbonRenderPrimitive = (
    segments: RenderPrimitive[],
    ribbon: PaintRibbon,
    color: Vec4,
    shade: number,
) => {
    segments.push({
        kind: "ribbon",
        vertices: ribbon.vertices,
        closed: ribbon.closed,
        color,
        shade,
    });
};

export const appendWorldStrokeRun = (
    segments: RenderPrimitive[],
    points: Array<Vec3 | null>,
    color: Vec4,
    width: number,
) => {
    let run: Vec3[] = [];
    const flushRun = () => {
        if (run.length >= 2) {
            segments.push({
                kind: "stroke",
                points: run,
                color,
                width,
            });
        }
        run = [];
    };

    for (const point of points) {
        if (point) {
            run.push(point);
        } else {
            flushRun();
        }
    }
    flushRun();
};

export const parseColor = (color: string, opacity: number): Vec4 => {
    const value = color.startsWith("#") ? color.slice(1) : color;
    const r = parseInt(value.slice(0, 2), 16) / 255;
    const g = parseInt(value.slice(2, 4), 16) / 255;
    const b = parseInt(value.slice(4, 6), 16) / 255;
    return [r, g, b, opacity];
};
