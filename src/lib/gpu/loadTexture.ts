export async function loadTexture(device: GPUDevice, url: string): Promise<GPUTexture> {
    const response = await fetch(url);
    const blob = await response.blob();
    const source = await createImageBitmap(blob);

    const texture = device.createTexture({
        label: url,
        size: [source.width, source.height],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    device.queue.copyExternalImageToTexture(
        { source, flipY: false },
        { texture },
        [source.width, source.height]
    );

    return texture;
}
