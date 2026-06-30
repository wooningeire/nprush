import {
    cameraCenter,
    viewForward,
} from "./projection.ts";
import {
    appendRibbonMeshTriangles,
    appendStrokeRenderTriangles,
    parseColor,
} from "./renderGeometry.ts";
import { BASE_PAINT_LAYER_ID } from "./paintLayers.ts";
import { buildRibbonGeometryFromDraft } from "./strokeMesh.ts";
import { dot3, sub3 } from "./vectorMath.ts";
import type {
    BrushStyle,
    PaintLayer,
    PaintObject,
    PaintRenderOptions,
    PaintStroke,
    PaintView,
    RenderPrimitive,
    Vec2,
    Vec3,
} from "../types.ts";

export type RenderAssemblyContext = {
    objects: PaintObject[],
    views: PaintView[],
    strokes: PaintStroke[],
    paintLayers: PaintLayer[],
    renderView: PaintView | null,
    activeObject: PaintObject | null,
    activeView: PaintView | null,
    draftStroke: Vec2[] | null,
    brush: BrushStyle,
};

export const buildPaintRenderSegments = (
    context: RenderAssemblyContext,
    options: boolean | PaintRenderOptions = true,
): RenderPrimitive[] => {
    const renderOptions = normalizeRenderOptions(options);
    const segments: RenderPrimitive[] = [];
    const objectById = new Map(context.objects.map(object => [object.id, object]));

    for (const stroke of sortedStrokesForRender(
        context.strokes,
        objectById,
        context.paintLayers,
        context.renderView,
    )) {
        appendStrokeRenderTriangles(segments, stroke, renderOptions.shadeRibbons);
    }

    if (renderOptions.showDraftStroke) {
        appendDraftStrokePreviewSegments(segments, context, renderOptions.shadeRibbons);
    }

    return segments;
};

export const buildDraftPaintRenderSegments = (context: RenderAssemblyContext): RenderPrimitive[] => {
    const segments: RenderPrimitive[] = [];
    appendDraftStrokePreviewSegments(segments, context, true);
    return segments;
};

const normalizeRenderOptions = (options: boolean | PaintRenderOptions): Required<PaintRenderOptions> => {
    if (typeof options === "boolean") {
        return {
            showDraftStroke: true,
            shadeRibbons: true,
        };
    }
    return {
        showDraftStroke: options.showDraftStroke ?? true,
        shadeRibbons: options.shadeRibbons ?? true,
    };
};

const sortedStrokesForRender = (
    strokes: PaintStroke[],
    objectById: Map<string, PaintObject>,
    paintLayers: PaintLayer[],
    renderView: PaintView | null,
): PaintStroke[] => {
    const layerById = new Map(paintLayers.map(layer => [layer.id, layer]));
    const layerOrderForStroke = (stroke: PaintStroke): number => {
        const layer = layerById.get(stroke.layerId ?? BASE_PAINT_LAYER_ID);
        return layer?.order ?? 0;
    };
    const isLayerVisible = (stroke: PaintStroke): boolean => {
        const layer = layerById.get(stroke.layerId ?? BASE_PAINT_LAYER_ID);
        return layer?.visible ?? true;
    };

    return strokes
        .filter(stroke => {
            const object = objectById.get(stroke.objectId);
            return !!object?.visible && isLayerVisible(stroke);
        })
        .map(stroke => ({
            stroke,
            depth: strokeDepthForRender(stroke, renderView),
        }))
        .sort((a, b) =>
            layerOrderForStroke(a.stroke) - layerOrderForStroke(b.stroke)
            || b.depth - a.depth
            || (objectById.get(a.stroke.objectId)?.layerIndex ?? 0)
                - (objectById.get(b.stroke.objectId)?.layerIndex ?? 0)
            || a.stroke.paintOrder - b.stroke.paintOrder
        )
        .map(item => item.stroke);
};

const strokeDepthForRender = (
    stroke: PaintStroke,
    renderView: PaintView | null,
): number => {
    if (!renderView) return 0;
    const origin = cameraCenter(renderView);
    const forward = viewForward(renderView);
    let total = 0;
    let count = 0;

    for (const point of stroke.centerline) {
        const depth = dot3(sub3(point, origin), forward);
        if (!Number.isFinite(depth)) continue;
        total += depth;
        count += 1;
    }

    return count === 0 ? 0 : total / count;
};

const appendDraftStrokePreviewSegments = (
    segments: RenderPrimitive[],
    context: RenderAssemblyContext,
    shadeRibbons: boolean,
) => {
    const object = context.activeObject;
    const view = context.activeView;
    if (!context.draftStroke || context.draftStroke.length < 2 || !object?.visible || object.locked || !view) return;

    const geometry = buildRibbonGeometryFromDraft(context.draftStroke, view, context.brush.width);
    if (!geometry) return;

    const color = parseColor(context.brush.color, 0.72);
    appendRibbonMeshTriangles(segments, geometry.mesh, color, shadeRibbons ? 1 : 0);
};
