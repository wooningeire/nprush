/// <reference lib="webworker" />

import { fitImplicitBody } from "./fitSolver.ts";
import type {
    ContourStroke,
    FitView,
    ImplicitBodyParams,
} from "./types.ts";

interface FitRequest {
    type: "fit";
    requestId: number;
    initialParams?: ImplicitBodyParams;
    strokes: ContourStroke[];
    views: FitView[];
    meshResolution?: number;
    iterations?: number;
    candidatesPerIteration?: number;
}

interface CancelRequest {
    type: "cancel";
    requestId: number;
}

type WorkerRequest = FitRequest | CancelRequest;

const workerScope = self as DedicatedWorkerGlobalScope;
let activeRequestId: number | null = null;
let activeAbortController: AbortController | null = null;

workerScope.onmessage = (event: MessageEvent<WorkerRequest>) => {
    const message = event.data;

    if (message.type === "cancel") {
        if (message.requestId === activeRequestId) {
            activeAbortController?.abort();
        }
        return;
    }

    void runFit(message);
};

async function runFit(message: FitRequest) {
    activeAbortController?.abort();

    const abortController = new AbortController();
    activeRequestId = message.requestId;
    activeAbortController = abortController;

    try {
        const result = await fitImplicitBody({
            initialParams: message.initialParams,
            strokes: message.strokes,
            views: message.views,
            gpuEvaluator: null,
            signal: abortController.signal,
            meshResolution: message.meshResolution,
            iterations: message.iterations,
            candidatesPerIteration: message.candidatesPerIteration,
            onProgress: (progress, bestLoss) => {
                workerScope.postMessage({
                    type: "progress",
                    requestId: message.requestId,
                    progress,
                    bestLoss,
                });
            },
        });

        workerScope.postMessage({
            type: "done",
            requestId: message.requestId,
            result,
        }, [
            result.mesh.vertices.buffer,
            result.mesh.indices.buffer,
        ]);
    } catch (error) {
        const isAbort = abortController.signal.aborted
            || (error as DOMException)?.name === "AbortError";
        workerScope.postMessage({
            type: isAbort ? "canceled" : "error",
            requestId: message.requestId,
            error: serializeError(error),
        });
    } finally {
        if (activeRequestId === message.requestId) {
            activeRequestId = null;
            activeAbortController = null;
        }
    }
}

function serializeError(error: unknown): string {
    if (error instanceof Error) return error.message;
    return String(error);
}
