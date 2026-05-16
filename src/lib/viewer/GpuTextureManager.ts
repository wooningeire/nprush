import { MESH_DEPTH_FORMAT } from "$/gpu/GpuMeshRenderPipelineManager.ts";
import { computeOptimizationTextureSize } from "$/gpu/optimizationTextureSize.ts";
import { constants } from "$/gpu/constants.ts";

const OPTIMIZATION_SHORT = constants.OPTIMIZATION_SHORT;

export class GpuTextureManager {
    // Full-res textures
    displayResColorTexture: GPUTexture | null = null;
    displayResColorTextureView: GPUTextureView | null = null;
    displayResDepthTexture: GPUTexture | null = null;
    displayResDepthTextureView: GPUTextureView | null = null;
    displayResZBufferTexture: GPUTexture | null = null;
    displayResZBufferTextureView: GPUTextureView | null = null;
    displayResNormalTexture: GPUTexture | null = null;
    displayResNormalTextureView: GPUTextureView | null = null;
    displayResEdgeTexture: GPUTexture | null = null;
    displayResEdgeTextureView: GPUTextureView | null = null;
    displayResSplatColorTexture: GPUTexture | null = null;
    displayResSplatColorTextureView: GPUTextureView | null = null;
    displayResSplatDepthTexture: GPUTexture | null = null;
    displayResSplatDepthTextureView: GPUTextureView | null = null;
    displayResEdgeBezierTexture: GPUTexture | null = null;
    displayResEdgeBezierTextureView: GPUTextureView | null = null;
    displayResCoarseBezierTexture: GPUTexture | null = null;
    displayResCoarseBezierTextureView: GPUTextureView | null = null;
    displayResFineBezierTexture: GPUTexture | null = null;
    displayResFineBezierTextureView: GPUTextureView | null = null;
    displayResBlurredTexture: GPUTexture | null = null;
    displayResBlurTempTexture: GPUTexture | null = null;
    
    displayResWidth = 0;
    displayResHeight = 0;

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

    recreateDisplayResTextures(width: number, height: number): boolean {
        if (width === this.displayResWidth && height === this.displayResHeight && this.displayResColorTexture) {
            return false;
        }

        this.destroyDisplayRes();

        this.displayResWidth = width;
        this.displayResHeight = height;

        this.displayResColorTexture = this.device.createTexture({
            label: "display-res color texture",
            size: [width, height],
            format: this.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.displayResColorTextureView = this.displayResColorTexture.createView();
        
        this.displayResNormalTexture = this.device.createTexture({
            label: "display-res normal texture",
            size: [width, height],
            format: this.format,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.displayResNormalTextureView = this.displayResNormalTexture.createView();

        this.displayResDepthTexture = this.device.createTexture({
            label: "display-res depth visualization",
            size: [width, height],
            format: "r16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.displayResDepthTextureView = this.displayResDepthTexture.createView();

        this.displayResZBufferTexture = this.device.createTexture({
            label: "display-res z-buffer",
            size: [width, height],
            format: MESH_DEPTH_FORMAT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        this.displayResZBufferTextureView = this.displayResZBufferTexture.createView();

        this.displayResEdgeTexture = this.device.createTexture({
            label: "display-res edge map",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.displayResEdgeTextureView = this.displayResEdgeTexture.createView();

        this.displayResSplatColorTexture = this.device.createTexture({
            label: "display-res splat color",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        this.displayResSplatColorTextureView = this.displayResSplatColorTexture.createView();

        this.displayResSplatDepthTexture = this.device.createTexture({
            label: "display-res splat depth",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.displayResSplatDepthTextureView = this.displayResSplatDepthTexture.createView();

        this.displayResEdgeBezierTexture = this.device.createTexture({
            label: "display-res edge bezier view",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        this.displayResEdgeBezierTextureView = this.displayResEdgeBezierTexture.createView();

        this.displayResCoarseBezierTexture = this.device.createTexture({
            label: "display-res coarse bezier view",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        this.displayResCoarseBezierTextureView = this.displayResCoarseBezierTexture.createView();

        this.displayResFineBezierTexture = this.device.createTexture({
            label: "display-res fine bezier view",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        this.displayResFineBezierTextureView = this.displayResFineBezierTexture.createView();

        this.displayResBlurredTexture = this.device.createTexture({
            label: "display-res blurred target",
            size: [width, height],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });

        this.displayResBlurTempTexture = this.device.createTexture({
            label: "display-res blur temp",
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

    destroyDisplayRes() {
        this.displayResColorTexture?.destroy();
        this.displayResDepthTexture?.destroy();
        this.displayResZBufferTexture?.destroy();
        this.displayResNormalTexture?.destroy();
        this.displayResEdgeTexture?.destroy();
        this.displayResSplatColorTexture?.destroy();
        this.displayResSplatDepthTexture?.destroy();
        this.displayResEdgeBezierTexture?.destroy();
        this.displayResCoarseBezierTexture?.destroy();
        this.displayResFineBezierTexture?.destroy();
        this.displayResBlurredTexture?.destroy();
        this.displayResBlurTempTexture?.destroy();
        
        this.displayResColorTexture = null;
        this.displayResColorTextureView = null;
        this.displayResDepthTexture = null;
        this.displayResDepthTextureView = null;
        this.displayResZBufferTexture = null;
        this.displayResZBufferTextureView = null;
        this.displayResNormalTexture = null;
        this.displayResNormalTextureView = null;
        this.displayResEdgeTexture = null;
        this.displayResEdgeTextureView = null;
        this.displayResSplatColorTexture = null;
        this.displayResSplatColorTextureView = null;
        this.displayResSplatDepthTexture = null;
        this.displayResSplatDepthTextureView = null;
        this.displayResEdgeBezierTexture = null;
        this.displayResEdgeBezierTextureView = null;
        this.displayResCoarseBezierTexture = null;
        this.displayResCoarseBezierTextureView = null;
        this.displayResFineBezierTexture = null;
        this.displayResFineBezierTextureView = null;
        this.displayResBlurredTexture = null;
        this.displayResBlurTempTexture = null;
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
        this.destroyDisplayRes();
        this.destroyOptimization();
        this.dummyTexture?.destroy();
        this.dummyTexture = null;
        this.dummyTextureView = null;
    }
}
