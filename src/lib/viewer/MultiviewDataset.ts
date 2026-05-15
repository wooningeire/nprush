export class MultiviewDataset {
    readonly textures: GPUTexture[];
    readonly textureViews: GPUTextureView[];
    readonly viewProjMats: Float32Array[];
    readonly viewMats: Float32Array[];
    readonly invViewProjMats: Float32Array[];
    readonly invViewMats: Float32Array[];
    readonly numViews: number;

    constructor(device: GPUDevice, nViews: number, width: number, height: number) {
        this.numViews = nViews;
        this.textures = [];
        this.textureViews = [];
        this.viewProjMats = [];
        this.viewMats = [];
        this.invViewProjMats = [];
        this.invViewMats = [];

        for (let i = 0; i < nViews; i++) {
            const texture = device.createTexture({
                label: `multiview dataset slot ${i}`,
                size: [width, height],
                format: "rgba8unorm",
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
            this.textures.push(texture);

            const view = texture.createView({ label: `multiview view ${i}` });
            this.textureViews.push(view);

            this.viewProjMats.push(new Float32Array(16));
            this.viewMats.push(new Float32Array(16));
            this.invViewProjMats.push(new Float32Array(16));
            this.invViewMats.push(new Float32Array(16));
        }
    }

    destroy() {
        for (const textures of this.textures) {
            textures.destroy();
        }
    }
}