import type { FrameWriter } from "$/util/export.ts";
import type { TurntableOrbitSample } from "./turntablePath.ts";

export interface TurntableOrbitWritable {
    long: number;
    lat: number;
    radius: number;
}

/**
 * Sweep the turntable path, wait `stepsPerFrame` rAF ticks per frame, capture
 * composited ImageData, and write PNGs via `writer`. Restores orbit in `finally`.
 */
export async function runTurntableExport(options: {
    totalFrames: number;
    stepsPerFrame: number;
    isCanceled: () => boolean;
    captureFrame: () => Promise<ImageData>;
    writer: FrameWriter;
    orbit: TurntableOrbitWritable;
    restoreOrbit: TurntableOrbitWritable;
    evalAtT: (t: number) => TurntableOrbitSample;
    onProgress: (completedFraction: number) => void;
}): Promise<void> {
    const {
        totalFrames,
        stepsPerFrame,
        isCanceled,
        captureFrame,
        writer,
        orbit,
        restoreOrbit,
        evalAtT,
        onProgress,
    } = options;

    try {
        for (let frame = 0; frame < totalFrames; frame++) {
            if (isCanceled()) break;

            const t = frame / totalFrames;
            const p = evalAtT(t);
            orbit.long = p.long;
            orbit.lat = p.lat;
            orbit.radius = p.radius;

            for (let step = 0; step < stepsPerFrame; step++) {
                if (isCanceled()) break;
                await new Promise<void>(r => requestAnimationFrame(() => r()));
            }

            if (isCanceled()) break;

            const imageData = await captureFrame();
            await writer.write(imageData);
            onProgress((frame + 1) / totalFrames);
        }
    } finally {
        await writer.close();
        orbit.long = restoreOrbit.long;
        orbit.lat = restoreOrbit.lat;
        orbit.radius = restoreOrbit.radius;
    }
}
