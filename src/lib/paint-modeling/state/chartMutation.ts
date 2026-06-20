import type { PaintObject } from "../types.ts";

export const touchPaintCharts = (
    objects: PaintObject[],
    chartIds: Set<string>,
): PaintObject[] => {
    if (chartIds.size === 0) return objects;

    return objects.map(object => ({
        ...object,
        charts: object.charts.map(chart => chartIds.has(chart.id)
            ? {
                ...chart,
                depths: [...chart.depths],
                coverage: [...chart.coverage],
                seams: [...chart.seams],
            }
            : chart),
    }));
};
