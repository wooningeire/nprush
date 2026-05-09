export class TurntableFrameCaptureQueue {
    private pending: {
        resolve: (img: ImageData) => void;
        reject: (err: Error) => void;
    } | null = null;

    enqueue(): Promise<ImageData> {
        return new Promise((resolve, reject) => {
            this.pending = { resolve, reject };
        });
    }

    dequeue(): {
        resolve: (img: ImageData) => void;
        reject: (err: Error) => void;
    } | null {
        const p = this.pending;
        this.pending = null;
        return p;
    }
}
