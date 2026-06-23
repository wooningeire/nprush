import { CHART_RESOLUTION } from "./constants.ts";
import type {
    ChartProjectionMode,
    ChartRole,
    OcclusionClaim,
    PaintChart,
    PaintLayer,
    PaintObject,
    PaintStroke,
    PaintView,
    Vec3,
} from "../types.ts";

export const createChart = ({
    objectId,
    sourceViewId,
    role,
    projectionMode,
    depth,
}: {
    objectId: string,
    sourceViewId: string,
    role: ChartRole,
    projectionMode: ChartProjectionMode,
    depth: number,
}): PaintChart => {
    return {
        id: makeId("chart"),
        objectId,
        sourceViewId,
        role,
        projectionMode,
        width: CHART_RESOLUTION,
        height: CHART_RESOLUTION,
        depths: Array.from({ length: CHART_RESOLUTION * CHART_RESOLUTION }, () => depth),
        coverage: Array.from({ length: CHART_RESOLUTION * CHART_RESOLUTION }, () => 0),
        seams: Array.from({ length: CHART_RESOLUTION * CHART_RESOLUTION }, () => false),
        createdAt: Date.now(),
    };
};

export const cloneView = (view: PaintView): PaintView => {
    return {
        ...view,
        offset: [...view.offset] as Vec3,
        viewProjMat: [...view.viewProjMat],
        viewProjInvMat: [...view.viewProjInvMat],
        viewMat: [...view.viewMat],
        viewInvMat: [...view.viewInvMat],
    };
};

export const cloneObject = (object: PaintObject): PaintObject => {
    return {
        ...object,
        charts: object.charts.map(cloneChart),
    };
};

export const cloneChart = (chart: PaintChart): PaintChart => {
    return {
        ...chart,
        depths: [...chart.depths],
        coverage: [...chart.coverage],
        seams: [...chart.seams],
    };
};

export const clonePaintLayer = (layer: PaintLayer): PaintLayer => ({ ...layer });

export const cloneStroke = (stroke: PaintStroke): PaintStroke => {
    return {
        ...stroke,
        samples: stroke.samples.map(sample => ({
            ...sample,
            sourcePoint: { ...sample.sourcePoint },
            surfaceRef: {
                chartId: sample.surfaceRef.chartId,
                uv: { ...sample.surfaceRef.uv },
            },
        })),
        style: { ...stroke.style },
    };
};

export const cloneOcclusionClaim = (claim: OcclusionClaim): OcclusionClaim => {
    return {
        ...claim,
        backRefs: claim.backRefs.map(ref => ({
            chartId: ref.chartId,
            uv: { ...ref.uv },
        })),
        mask: claim.mask.map(point => ({ ...point })),
    };
};

export const makeId = (prefix: string): string => `${prefix}-${Math.random().toString(36).slice(2, 10)}`;