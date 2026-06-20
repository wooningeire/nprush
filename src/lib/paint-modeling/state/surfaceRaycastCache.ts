import { chartHasCoverage } from "./chartPainting.ts";
import {
    createChartRaycastCache,
    type ChartRaycastCache,
} from "./chartRaycast.ts";
import { chartPointToWorldFromView } from "./projection.ts";
import type { PaintChart, PaintObject, PaintView } from "../types.ts";

export type SurfaceRaycastTarget = {
    chartId: string,
    cache: ChartRaycastCache,
};

type CachedChartRaycast = {
    sourceView: PaintView,
    cache: ChartRaycastCache,
};

export class PaintSurfaceRaycastCache {
    private readonly cacheByChart = new WeakMap<PaintChart, CachedChartRaycast>();
    private readonly onCacheMiss?: () => void;

    constructor(onCacheMiss?: () => void) {
        this.onCacheMiss = onCacheMiss;
    }

    targetsForObject(
        object: PaintObject,
        views: PaintView[],
        excludeChartId?: string,
    ): SurfaceRaycastTarget[] {
        const targets: SurfaceRaycastTarget[] = [];
        for (const chart of object.charts) {
            if (chart.id === excludeChartId) continue;
            if (!chartHasCoverage(chart)) continue;

            const sourceView = views.find(item => item.id === chart.sourceViewId);
            if (!sourceView) continue;
            targets.push({
                chartId: chart.id,
                cache: this.cacheForChart(chart, sourceView),
            });
        }
        return targets;
    }

    private cacheForChart(chart: PaintChart, sourceView: PaintView): ChartRaycastCache {
        const cached = this.cacheByChart.get(chart);
        if (cached && cached.sourceView === sourceView) return cached.cache;

        this.onCacheMiss?.();
        const cache = createChartRaycastCache(
            chart,
            uv => chartPointToWorldFromView(chart, sourceView, uv),
        );
        this.cacheByChart.set(chart, { sourceView, cache });
        return cache;
    }
}