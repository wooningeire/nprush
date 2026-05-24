import type {
    FitImplicitBodyOptions,
    FitImplicitBodyResult,
} from "./fitSolver.ts";
import type { ContourStroke, FitView, ImplicitBodyParams, Vec2 } from "./types.ts";

type WorkerProgressMessage = {
    type: "progress";
    requestId: number;
    progress: number;
    bestLoss: number;
};

type WorkerDoneMessage = {
    type: "done";
    requestId: number;
    result: FitImplicitBodyResult;
};

type WorkerErrorMessage = {
    type: "error" | "canceled";
    requestId: number;
    error: string;
};

type WorkerResponse = WorkerProgressMessage | WorkerDoneMessage | WorkerErrorMessage;

let nextRequestId = 1;

export function supportsContourFitWorker(): boolean {
    return typeof Worker !== "undefined";
}

export function fitImplicitBodyInWorker({
    initialParams,
    strokes,
    views,
    signal,
    meshResolution,
    iterations,
    candidatesPerIteration,
    onProgress,
}: FitImplicitBodyOptions): Promise<FitImplicitBodyResult> {
    if (!supportsContourFitWorker()) {
        return Promise.reject(new Error("Contour fit worker is unavailable"));
    }

    const requestId = nextRequestId++;
    const worker = new Worker(new URL("./contourFit.worker.ts", import.meta.url), {
        type: "module",
        name: "contour-fit-worker",
    });

    return new Promise((resolve, reject) => {
        let settled = false;

        const cleanup = () => {
            signal?.removeEventListener("abort", onAbort);
            worker.terminate();
        };

        const finish = (callback: () => void) => {
            if (settled) return;
            settled = true;
            cleanup();
            callback();
        };

        const onAbort = () => {
            worker.postMessage({ type: "cancel", requestId });
            finish(() => reject(new DOMException("Fit canceled", "AbortError")));
        };

        worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
            const message = event.data;
            if (message.requestId !== requestId) return;

            if (message.type === "progress") {
                onProgress?.(message.progress, message.bestLoss);
                return;
            }

            if (message.type === "done") {
                finish(() => resolve(message.result));
                return;
            }

            if (message.type === "canceled") {
                finish(() => reject(new DOMException("Fit canceled", "AbortError")));
                return;
            }

            finish(() => reject(new Error(message.error)));
        };

        worker.onerror = event => {
            finish(() => reject(new Error(event.message || "Contour fit worker failed")));
        };

        if (signal?.aborted) {
            onAbort();
            return;
        }

        signal?.addEventListener("abort", onAbort, { once: true });
        try {
            worker.postMessage({
                type: "fit",
                requestId,
                initialParams: cloneParams(initialParams),
                strokes: cloneStrokes(strokes),
                views: cloneViews(views),
                meshResolution,
                iterations,
                candidatesPerIteration,
            });
        } catch (error) {
            finish(() => reject(error));
        }
    });
}

function cloneParams(params: ImplicitBodyParams | undefined): ImplicitBodyParams | undefined {
    if (!params) return undefined;
    return {
        center: [params.center[0], params.center[1], params.center[2]],
        axisX: params.axisX ? [params.axisX[0], params.axisX[1], params.axisX[2]] : undefined,
        axisY: params.axisY ? [params.axisY[0], params.axisY[1], params.axisY[2]] : undefined,
        axisZ: params.axisZ ? [params.axisZ[0], params.axisZ[1], params.axisZ[2]] : undefined,
        height: params.height,
        radiusBottom: params.radiusBottom,
        radiusTop: params.radiusTop,
        bulge: params.bulge,
        ovalX: params.ovalX,
        ovalZ: params.ovalZ,
        boxiness: params.boxiness ?? 0,
    };
}

function cloneStrokes(strokes: ContourStroke[]): ContourStroke[] {
    return strokes.map(stroke => ({
        id: stroke.id,
        kind: stroke.kind,
        viewId: stroke.viewId,
        shapeId: stroke.shapeId,
        points: clonePoints(stroke.points),
        resampledPoints: clonePoints(stroke.resampledPoints),
        tangents: clonePoints(stroke.tangents),
        normals: clonePoints(stroke.normals),
        weight: stroke.weight,
        depthNdc: stroke.depthNdc,
        depthOffset: stroke.depthOffset,
        depthLocked: stroke.depthLocked,
        depthSamplesNdc: stroke.depthSamplesNdc ? Array.from(stroke.depthSamplesNdc) : undefined,
        depthSamplesOffset: stroke.depthSamplesOffset ? Array.from(stroke.depthSamplesOffset) : undefined,
        depthSamplesLocked: stroke.depthSamplesLocked ? Array.from(stroke.depthSamplesLocked) : undefined,
    }));
}

function cloneViews(views: FitView[]): FitView[] {
    return views.map(view => ({
        id: view.id,
        viewProjMat: Array.from(view.viewProjMat),
        viewProjInvMat: view.viewProjInvMat ? Array.from(view.viewProjInvMat) : undefined,
        viewInvMat: view.viewInvMat ? Array.from(view.viewInvMat) : undefined,
        width: view.width,
        height: view.height,
    }));
}

function clonePoints(points: Vec2[]): Vec2[] {
    return points.map(point => ({ x: point.x, y: point.y }));
}
