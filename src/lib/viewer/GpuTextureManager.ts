import { MESH_DEPTH_FORMAT } from "$/gpu/GpuMeshRenderPipelineManager.ts";
import { computeOptimizationTextureSize } from "$/gpu/optimizationTextureSize.ts";
import { constants } from "$/gpu/constants.ts";

const OPTIMIZATION_SHORT = constants.OPTIMIZATION_SHORT;

export class GpuTextureManager {
    // Full-res textures
    fullResColorTexture: GPUTexture | null = null;
    fullResColorTextureView: GPUTextureView | null = null;
    fullResDepthTexture: GPUTexture | null = null;
    fullResDepthTextureView: GPUTextureView | null = null;
    fullResZBufferTexture: GPUTexture | null = null;
    fullResZBufferTextureView: GPUTextureView | null = null;
    fullResNormalTexture: GPUTexture | null = null;
    fullResNormalTextureView: GPUTextureView | null = null;
    fullResEdgeTexture: GPUTexture | null = null;
    fullResEdgeTextureView: GPUTextureView | null = null;
    fullResSplatColorTexture: GPUTexture | null = null;
    fullResSplatColorTextureView: GPUTextureView | null = null;
    fullResSplatDepthTexture: GPUTexture | null = null;
    fullResSplatDepthTextureView: GPUTextureView | null = null;
    fullResEdgeBezierTexture: GPUTexture | null = null;
    fullResEdgeBezierTextureView: GPUTextureView | null = null;
    fullResCoarseBezierTexture: GPUTexture | null = null;
    fullResCoarseBezierTextureView: GPUTextureView | null = null;
    fullResFineBezierTexture: GPUTexture | null = null;
    fullResFineBezierTextureView: GPUTextureView | null = null;
    fullResBlurredTexture: GPUTexture | null = null;
    fullResBlurTempTexture: GPUTexture | null = null;
    
    fullResWidth = 0;
    fullResHeight = 0;

    // Optimization textures
    optimizationColorTexture: GPUTexture | null = null;
    optimizationColorTextureView: GPUTextureView | null = null;
    optimizationDepthTexture: GPUTexture | null = null;
    optimizationDepthTextureView: GPUTextureView | null = null;
    optimizationZBufferTexture: GPUTexture | null = null;
    optimizationZBufferTextureView: GPUTextureView | null = null;
    optimizationNormalTexture: GPUTexture | null = null;
    optimizationNormalTextureView: GPUTextureView | null = null;
    optimizationEdgeTexture: GPUTexture | null = null;
    optimizationEdgeTextureView: GPUTextureView | null = null;
    optimizationSplatColorTexture: GPUTexture | null = null;
    optimizationSplatColorTextureView: GPUTextureView | null = null;
    optimizationSplatDepthTexture: GPUTexture | null = null;
    optimizationSplatDepthTextureView: GPUTextureView | null = null;
    optimizationBlurredTexture: GPUTexture | null = null;
    optimizationBlurredTextureView: GPUTextureView | null = null;
    optimizationDepthAwareBlurredTexture: GPUTexture | null = null;
    optimizationDepthAwareBlurredTextureView: GPUTextureView | null = null;
    optimizationBlurredDepthTexture: GPUTexture | null = null;
    optimizationBlurredDepthTextureView: GPUTextureView | null = null;
    optimizationBlurTempTexture: GPUTexture | null = null;
    optimizationBlurTempTextureView: GPUTextureView | null = null;

    optimizationWidth = 0;
    optimizationHeight = 0;

    dummyTexture: GPUTexture | null = null;
    dummyTextureView: GPUTextureView | null = null;

    constructor(private device: GPUDevice, private format: GPUTextureFormat) {
        this.dummyTexture = this.device.createTexture({
            label: "dummy 1x1 texture",
            size: [1, 1],
            format: "rgba8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING,
        });
        this.dummyTextureView = this.dummyTexture.createView();
    }

    recreateFullResTextures(width: number, height: number): boolean {
        if (width === this.fullResWidth && height === this.fullResHeight && this.fullResColorTexture) {
            return false;
        }

        this.destroyFullRes();

        this.fullResWidth = width;
        this.fullResHeight = height;

        this.fullResColorTexture = this.device.createTexture({
            label: "full-res color texture",
            size: [width, height],
            format: this.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.fullResColorTextureView = this.fullResColorTexture.createView();
        
        this.fullResNormalTexture = this.device.createTexture({
            label: "full-res normal texture",
            size: [width, height],
            format: this.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.fullResNormalTextureView = this.fullResNormalTexture.createView();

        this.fullResDepthTexture = this.device.createTexture({
            label: "full-res depth visualization",
            size: [width, height],
            format: "r16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.fullResDepthTextureView = this.fullResDepthTexture.createView();

        this.fullResZBufferTexture = this.device.createTexture({
            label: "full-res z-buffer",
            size: [width, height],
            format: MESH_DEPTH_FORMAT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.fullResZBufferTextureView = this.fullResZBufferTexture.createView();

        this.fullResEdgeTexture = this.device.createTexture({
            label: "full-res edge map",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.fullResEdgeTextureView = this.fullResEdgeTexture.createView();

        this.fullResSplatColorTexture = this.device.createTexture({
            label: "full-res splat color",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        this.fullResSplatColorTextureView = this.fullResSplatColorTexture.createView();

        this.fullResSplatDepthTexture = this.device.createTexture({
            label: "full-res splat depth",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.fullResSplatDepthTextureView = this.fullResSplatDepthTexture.createView();

        this.fullResEdgeBezierTexture = this.device.createTexture({
            label: "full-res edge bezier view",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        this.fullResEdgeBezierTextureView = this.fullResEdgeBezierTexture.createView();

        this.fullResCoarseBezierTexture = this.device.createTexture({
            label: "full-res coarse bezier view",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        this.fullResCoarseBezierTextureView = this.fullResCoarseBezierTexture.createView();

        this.fullResFineBezierTexture = this.device.createTexture({
            label: "full-res fine bezier view",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        this.fullResFineBezierTextureView = this.fullResFineBezierTexture.createView();

        this.fullResBlurredTexture = this.device.createTexture({
            label: "full-res blurred target",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });

        this.fullResBlurTempTexture = this.device.createTexture({
            label: "full-res blur temp",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });

        return true;
    }

    recreateOptimizationTextures(panelAspect: number): boolean {
        const { width, height } = computeOptimizationTextureSize(OPTIMIZATION_SHORT, panelAspect);

        if (width === this.optimizationWidth && height === this.optimizationHeight && this.optimizationColorTexture) {
            return false;
        }

        this.destroyOptimization();

        this.optimizationWidth = width;
        this.optimizationHeight = height;

        this.optimizationColorTexture = this.device.createTexture({
            label: "optimization color texture",
            size: [width, height],
            format: this.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimizationColorTextureView = this.optimizationColorTexture.createView();
        
        this.optimizationNormalTexture = this.device.createTexture({
            label: "optimization normal texture",
            size: [width, height],
            format: this.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimizationNormalTextureView = this.optimizationNormalTexture.createView();

        this.optimizationDepthTexture = this.device.createTexture({
            label: "optimization depth visualization",
            size: [width, height],
            format: "r16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimizationDepthTextureView = this.optimizationDepthTexture.createView();

        this.optimizationZBufferTexture = this.device.createTexture({
            label: "optimization z-buffer",
            size: [width, height],
            format: MESH_DEPTH_FORMAT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.optimizationZBufferTextureView = this.optimizationZBufferTexture.createView();

        this.optimizationEdgeTexture = this.device.createTexture({
            label: "optimization edge map",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimizationEdgeTextureView = this.optimizationEdgeTexture.createView();

        this.optimizationSplatColorTexture = this.device.createTexture({
            label: "optimization splat color",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimizationSplatColorTextureView = this.optimizationSplatColorTexture.createView();

        this.optimizationSplatDepthTexture = this.device.createTexture({
            label: "optimization splat depth",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimizationSplatDepthTextureView = this.optimizationSplatDepthTexture.createView();

        this.optimizationBlurredTexture = this.device.createTexture({
            label: "optimization blurred target",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimizationBlurredTextureView = this.optimizationBlurredTexture.createView();

        this.optimizationDepthAwareBlurredTexture = this.device.createTexture({
            label: "optimization depth-aware blurred target",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimizationDepthAwareBlurredTextureView = this.optimizationDepthAwareBlurredTexture.createView();

        this.optimizationBlurredDepthTexture = this.device.createTexture({
            label: "optimization blurred depth",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimizationBlurredDepthTextureView = this.optimizationBlurredDepthTexture.createView();

        this.optimizationBlurTempTexture = this.device.createTexture({
            label: "optimization blur temp",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.optimizationBlurTempTextureView = this.optimizationBlurTempTexture.createView();

        return true;
    }

    destroyFullRes() {
        this.fullResColorTexture?.destroy();
        this.fullResDepthTexture?.destroy();
        this.fullResZBufferTexture?.destroy();
        this.fullResNormalTexture?.destroy();
        this.fullResEdgeTexture?.destroy();
        this.fullResSplatColorTexture?.destroy();
        this.fullResSplatDepthTexture?.destroy();
        this.fullResEdgeBezierTexture?.destroy();
        this.fullResCoarseBezierTexture?.destroy();
        this.fullResFineBezierTexture?.destroy();
        this.fullResBlurredTexture?.destroy();
        this.fullResBlurTempTexture?.destroy();
        
        this.fullResColorTexture = null;
        this.fullResColorTextureView = null;
        this.fullResDepthTexture = null;
        this.fullResDepthTextureView = null;
        this.fullResZBufferTexture = null;
        this.fullResZBufferTextureView = null;
        this.fullResNormalTexture = null;
        this.fullResNormalTextureView = null;
        this.fullResEdgeTexture = null;
        this.fullResEdgeTextureView = null;
        this.fullResSplatColorTexture = null;
        this.fullResSplatColorTextureView = null;
        this.fullResSplatDepthTexture = null;
        this.fullResSplatDepthTextureView = null;
        this.fullResEdgeBezierTexture = null;
        this.fullResEdgeBezierTextureView = null;
        this.fullResCoarseBezierTexture = null;
        this.fullResCoarseBezierTextureView = null;
        this.fullResFineBezierTexture = null;
        this.fullResFineBezierTextureView = null;
        this.fullResBlurredTexture = null;
        this.fullResBlurTempTexture = null;
    }

    destroyOptimization() {
        this.optimizationColorTexture?.destroy();
        this.optimizationNormalTexture?.destroy();
        this.optimizationDepthTexture?.destroy();
        this.optimizationZBufferTexture?.destroy();
        this.optimizationEdgeTexture?.destroy();
        this.optimizationSplatColorTexture?.destroy();
        this.optimizationSplatDepthTexture?.destroy();
        this.optimizationBlurredTexture?.destroy();
        this.optimizationDepthAwareBlurredTexture?.destroy();
        this.optimizationBlurredDepthTexture?.destroy();
        this.optimizationBlurTempTexture?.destroy();

        this.optimizationColorTexture = null;
        this.optimizationColorTextureView = null;
        this.optimizationNormalTexture = null;
        this.optimizationNormalTextureView = null;
        this.optimizationDepthTexture = null;
        this.optimizationDepthTextureView = null;
        this.optimizationZBufferTexture = null;
        this.optimizationZBufferTextureView = null;
        this.optimizationEdgeTexture = null;
        this.optimizationEdgeTextureView = null;
        this.optimizationSplatColorTexture = null;
        this.optimizationSplatColorTextureView = null;
        this.optimizationSplatDepthTexture = null;
        this.optimizationSplatDepthTextureView = null;
        this.optimizationBlurredTexture = null;
        this.optimizationBlurredTextureView = null;
        this.optimizationDepthAwareBlurredTexture = null;
        this.optimizationDepthAwareBlurredTextureView = null;
        this.optimizationBlurredDepthTexture = null;
        this.optimizationBlurredDepthTextureView = null;
        this.optimizationBlurTempTexture = null;
        this.optimizationBlurTempTextureView = null;
    }

    destroy() {
        this.destroyFullRes();
        this.destroyOptimization();
        this.dummyTexture?.destroy();
        this.dummyTexture = null;
        this.dummyTextureView = null;
    }
}
