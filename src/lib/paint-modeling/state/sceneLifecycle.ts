import { vec3 } from "wgpu-matrix";
import { distance3 } from "./vectorMath.ts";
import type {
    PaintObject,
    PaintStroke,
    PaintView,
    Vec3,
} from "../types.ts";

type PaintCameraSource = {
    viewProjMat: ArrayLike<number>,
    viewProjInvMat: ArrayLike<number>,
    viewMat: ArrayLike<number>,
    viewInvMat: ArrayLike<number>,
};

type PaintOrbitSource = {
    long: number,
    lat: number,
    radius: number,
    offset: ArrayLike<number>,
};

export type ObjectDeletionResult = {
    objects: PaintObject[],
    strokes: PaintStroke[],
    activeObjectId: string | null,
};

export type ViewDeletionResult = {
    views: PaintView[],
    strokes: PaintStroke[],
    selectViewId: string | null,
};

export const capturePaintView = (
    name: string,
    order: number,
    width: number,
    height: number,
    orbit: PaintOrbitSource,
    camera: PaintCameraSource,
): PaintView => ({
    id: makeViewId(),
    name,
    order,
    long: orbit.long,
    lat: orbit.lat,
    radius: orbit.radius,
    offset: Array.from(orbit.offset).slice(0, 3) as Vec3,
    width,
    height,
    viewProjMat: Array.from(camera.viewProjMat),
    viewProjInvMat: Array.from(camera.viewProjInvMat),
    viewMat: Array.from(camera.viewMat),
    viewInvMat: Array.from(camera.viewInvMat),
    createdAt: Date.now(),
});

export const selectPaintView = (
    views: PaintView[],
    orbit: PaintOrbitSource & { offset: Float32Array },
    viewId: string,
): PaintView | null => {
    const view = views.find(item => item.id === viewId);
    if (!view) return null;

    orbit.long = view.long;
    orbit.lat = view.lat;
    orbit.radius = view.radius;
    orbit.offset = vec3.fromValues(view.offset[0], view.offset[1], view.offset[2]);
    return view;
};

export const cameraMovedFromPaintView = (
    view: PaintView,
    orbit: PaintOrbitSource,
): boolean => {
    const offset = Array.from(orbit.offset).slice(0, 3) as Vec3;
    const offsetDelta = distance3(offset, view.offset);
    return Math.abs(orbit.long - view.long) > 0.015
        || Math.abs(orbit.lat - view.lat) > 0.015
        || Math.abs(Math.log(orbit.radius / view.radius)) > 0.015
        || offsetDelta > 0.01;
};

export const viewHasAuthoredContent = (
    viewId: string,
    strokes: PaintStroke[],
): boolean => strokes.some(stroke => stroke.sourceViewId === viewId);

export const deletePaintObject = (
    objectId: string,
    objects: PaintObject[],
    strokes: PaintStroke[],
    activeObjectId: string | null,
): ObjectDeletionResult | null => {
    if (!objects.some(object => object.id === objectId)) return null;

    const nextObjects = objects.filter(object => object.id !== objectId);
    return {
        objects: nextObjects,
        strokes: strokes.filter(stroke => stroke.objectId !== objectId),
        activeObjectId: activeObjectId === objectId
            ? nextObjects[0]?.id ?? null
            : activeObjectId,
    };
};

export const deletePaintView = (
    viewId: string,
    views: PaintView[],
    strokes: PaintStroke[],
    activeViewId: string | null,
): ViewDeletionResult | null => {
    if (!views.some(view => view.id === viewId)) return null;

    const nextViews = views.filter(view => view.id !== viewId);
    return {
        views: nextViews,
        strokes: strokes.filter(stroke => stroke.sourceViewId !== viewId),
        selectViewId: activeViewId === viewId
            ? nextViews[0]?.id ?? null
            : activeViewId,
    };
};

const makeViewId = (): string => `view-${Math.random().toString(36).slice(2, 10)}`;
