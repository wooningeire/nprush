/**
 * Utilities for reading back data from GPU textures.
 */

/**
 * Reads a GPUTexture into an ImageData object.
 * Handles padding and format conversion (BGRA -> RGBA).
 */
export async function readTextureToImageData(
    device: GPUDevice,
    texture: GPUTexture,
    width: number,
    height: number,
    format: GPUTextureFormat
): Promise<ImageData> {
    const bytesPerPixel = 4;
    const unpaddedBytesPerRow = width * bytesPerPixel;
    const paddedBytesPerRow = Math.ceil(unpaddedBytesPerRow / 256) * 256;

    const buffer = device.createBuffer({
        label: "texture readback buffer",
        size: paddedBytesPerRow * height,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const enc = device.createCommandEncoder({ label: "texture readback encoder" });
    enc.copyTextureToBuffer(
        { texture },
        { buffer, bytesPerRow: paddedBytesPerRow },
        [width, height],
    );
    device.queue.submit([enc.finish()]);

    await buffer.mapAsync(GPUMapMode.READ);
    const raw = new Uint8Array(buffer.getMappedRange().slice(0));
    buffer.unmap();
    buffer.destroy();

    const imageData = new ImageData(width, height);
    const isBgra = format === "bgra8unorm" || texture.format === "bgra8unorm";
    
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const src = y * paddedBytesPerRow + x * bytesPerPixel;
            const dst = (y * width + x) * 4;
            if (isBgra) {
                imageData.data[dst + 0] = raw[src + 2];
                imageData.data[dst + 1] = raw[src + 1];
                imageData.data[dst + 2] = raw[src + 0];
                imageData.data[dst + 3] = raw[src + 3];
            } else {
                imageData.data[dst + 0] = raw[src + 0];
                imageData.data[dst + 1] = raw[src + 1];
                imageData.data[dst + 2] = raw[src + 2];
                imageData.data[dst + 3] = raw[src + 3];
            }
        }
    }
    return imageData;
}

/**
 * Converts ImageData to a Blob (PNG).
 */
export async function imageDataToBlob(imageData: ImageData): Promise<Blob> {
    const canvas = document.createElement("canvas");
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    const ctx = canvas.getContext("2d")!;
    ctx.putImageData(imageData, 0, 0);
    
    return new Promise((resolve, reject) => {
        canvas.toBlob((blob) => {
            if (blob) resolve(blob);
            else reject(new Error("Failed to create blob from ImageData"));
        }, "image/png");
    });
}
