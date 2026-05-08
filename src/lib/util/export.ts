/**
 * Utilities for exporting data from the browser (downloads, video encoding).
 */

/**
 * Triggers a browser download for a Blob.
 */
export function downloadBlob(blob: Blob, filename: string) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

/**
 * Encodes a sequence of ImageData frames into a video Blob.
 * Uses MediaRecorder with a temporary canvas.
 */
export async function encodeFramesToVideo(
    frames: ImageData[],
    fps: number = 30,
    bitrate: number = 8_000_000
): Promise<{ blob: Blob; mimeType: string }> {
    if (frames.length === 0) {
        throw new Error("No frames provided for video encoding");
    }

    const w = frames[0].width;
    const h = frames[0].height;
    const frameDuration = 1000 / fps;

    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d")!;

    const stream = canvas.captureStream(fps);
    
    const mimeType = [
        "video/webm;codecs=vp9",
        "video/webm;codecs=vp8",
        "video/webm",
        "video/mp4",
    ].find(t => MediaRecorder.isTypeSupported(t)) || "video/webm";

    const recorder = new MediaRecorder(stream, {
        mimeType,
        videoBitsPerSecond: bitrate,
    });

    const chunks: Blob[] = [];
    recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
    };

    const done = new Promise<void>((resolve, reject) => {
        recorder.onstop = () => resolve();
        recorder.onerror = (e) => reject(e);
    });

    recorder.start();
    // Brief delay to ensure recorder is ready
    await new Promise(r => setTimeout(r, 100));

    for (const frame of frames) {
        ctx.putImageData(frame, 0, 0);
        await new Promise(r => setTimeout(r, frameDuration));
    }

    // Brief delay to ensure last frame is captured
    await new Promise(r => setTimeout(r, 100));

    recorder.stop();
    await done;

    return {
        blob: new Blob(chunks, { type: mimeType }),
        mimeType
    };
}
