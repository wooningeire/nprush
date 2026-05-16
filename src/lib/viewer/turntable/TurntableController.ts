import type { Camera } from "../Camera.svelte.ts";
import type { ViewerState } from "../ViewerState.svelte.ts";
import type { GpuPathTracePipelineManager } from "$/gpu/pathtrace/GpuPathTracePipelineManager.ts";
import type { GpuSplatOptimizerManager } from "$/gpu/splat/GpuSplatOptimizerManager.ts";
import type { GpuBezierOptimizerManager } from "$/gpu/bezier/GpuBezierOptimizerManager.ts";
import type { GpuBezierForwardPipelineManager } from "$/gpu/bezier/GpuBezierForwardPipelineManager.ts";
import type { GpuUniformsBufferManager } from "$/gpu/GpuUniformsBufferManager.ts";
import { evaluateTurntablePath } from "./turntablePath.ts";
import { compositeTurntableLayers } from "./turntableComposite.ts";
import { TurntableFrameCaptureQueue } from "./turntableFrameCapture.ts";
import { MultiviewDataset } from "../MultiviewDataset.ts";
import { RENDER_MODE_MULTIVIEW } from "../renderMode.ts";
import { readTextureToImageData } from "$/gpu/file-save/readback.ts";
import type { Mat4 } from "wgpu-matrix";

/** Per-frame view state resolved by the turntable controller. */
export interface TurntableFrameView {
	/** Dataset texture view if training from prerendered data; null otherwise. */
	datasetView: GPUTextureView | null;
	/** View-projection matrix for sorting / rendering (dataset or live camera). */
	sortVp: Mat4;
	/** Inverse view-projection matrix (dataset or live camera). */
	vpInv: Mat4;
	/** Inverse view matrix (dataset or live camera). */
	invView: Mat4;
}

/** Managers that the controller needs to write dataset matrices into. */
export interface TurntableGpuManagers {
	uniformsManager: GpuUniformsBufferManager;
	pathTracePipelineManager: GpuPathTracePipelineManager;
	splatOptimizerManager: GpuSplatOptimizerManager;
	edgeLayerBezierManager: GpuBezierOptimizerManager;
	coarseColorLayerBezierManager: GpuBezierOptimizerManager;
	fineColorLayerBezierManager: GpuBezierOptimizerManager;
	bezierForwardManager: GpuBezierForwardPipelineManager;
	baseColorBezierForwardManager: GpuBezierForwardPipelineManager;
	colorBezierForwardManager: GpuBezierForwardPipelineManager;
}

/**
 * Owns turntable-specific state that was previously scattered across GpuRunner:
 * - TurntableFrameCaptureQueue (PNG frame readback)
 * - MultiviewDataset lifecycle (prerender, train-from-dataset view selection)
 * - Per-frame dataset view resolution
 */
export class TurntableController {
	private readonly device: GPUDevice;
	private readonly camera: Camera;
	private readonly viewerState: ViewerState;
	private readonly managers: TurntableGpuManagers;

	private readonly captureQueue = new TurntableFrameCaptureQueue();

	private multiviewDataset: MultiviewDataset | null = null;
	/** Dataset slot reused for consecutive frames before resampling; -1 means unset. */
	private heldDatasetIndex = -1;
	private framesHeldOnDataset = 0;

	constructor(opts: {
		device: GPUDevice;
		camera: Camera;
		viewerState: ViewerState;
		managers: TurntableGpuManagers;
	}) {
		this.device = opts.device;
		this.camera = opts.camera;
		this.viewerState = opts.viewerState;
		this.managers = opts.managers;
	}

	// ── Capture queue delegation ──────────────────────────────────

	/** Enqueue a turntable frame capture request (resolved after the next readback). */
	captureTurntableFrame(): Promise<ImageData> {
		return this.captureQueue.enqueue();
	}

	/** Whether a capture readback is pending (used to bypass frozen-viewport skip). */
	hasPendingCapture(): boolean {
		return this.captureQueue.hasPending();
	}

	/**
	 * Dequeue a pending capture and resolve it by reading back composited layers.
	 * Call this at the end of a frame, after GPU submission + onSubmittedWorkDone.
	 */
	async resolvePendingCapture(
		fullWidth: number,
		fullHeight: number,
		fullSplatTexture: GPUTexture | null,
		fullBaseColorBezierTexture: GPUTexture | null,
		fullColorBezierTexture: GPUTexture | null,
		fullBezierTexture: GPUTexture | null,
	): Promise<void> {
		const pending = this.captureQueue.dequeue();
		if (!pending) return;

		const { resolve, reject } = pending;
		try {
			if (!fullWidth || !fullHeight || !fullSplatTexture) {
				reject(new Error("Textures not ready"));
				return;
			}

			const splat = await readTextureToImageData(
				this.device, fullSplatTexture, fullWidth, fullHeight, fullSplatTexture.format,
			);
			const baseColorBezier = this.viewerState.coarseColorBeziersEnabled && fullBaseColorBezierTexture
				? await readTextureToImageData(
					this.device, fullBaseColorBezierTexture, fullWidth, fullHeight, fullBaseColorBezierTexture.format,
				)
				: null;
			const colorBezier = this.viewerState.fineColorBeziersEnabled && fullColorBezierTexture
				? await readTextureToImageData(
					this.device, fullColorBezierTexture, fullWidth, fullHeight, fullColorBezierTexture.format,
				)
				: null;
			const edgeBezier = this.viewerState.edgeBeziersEnabled && fullBezierTexture
				? await readTextureToImageData(
					this.device, fullBezierTexture, fullWidth, fullHeight, fullBezierTexture.format,
				)
				: null;

			resolve(compositeTurntableLayers(fullWidth, fullHeight, {
				splat,
				baseColorBezier,
				colorBezier,
				edgeBezier,
			}));
		} catch (e) {
			reject(e as Error);
		}
	}

	// ── Prerender dataset ─────────────────────────────────────────

	/**
	 * Prerender path-traced views into a GPU-resident dataset.
	 * Blocks (via rAF) until all views have converged.
	 */
	async prerenderDataset(optimizationWidth: number, optimizationHeight: number): Promise<void> {
		const viewerState = this.viewerState;
		const numViews = viewerState.multiviewNumViews as number;
		const samplesPerView = viewerState.turntableMinSamplesPerView as number;

		// Wait until optimization textures are ready (loop may not have run yet).
		while (optimizationWidth === 0) {
			await new Promise<void>(r => requestAnimationFrame(() => r()));
		}

		// Destroy any previous dataset and allocate fresh slots.
		this.multiviewDataset?.destroy();
		this.multiviewDataset = null;
		const dataset = new MultiviewDataset(this.device, numViews, optimizationWidth, optimizationHeight);

		viewerState.multiviewPrerendering = true;
		viewerState.multiviewPrerenderProgress = 0;
		viewerState.multiviewDatasetReady = false;

		try {
			for (let i = 0; i < numViews; i++) {
				if (!viewerState.turntableTraining) break; // canceled

				// Sample a deterministic view position spread evenly around the path.
				const t = i / numViews;
				const p = evaluateTurntablePath(t, viewerState.turntableBaseLong, viewerState.getTurntablePathParams());
				viewerState.orbit.long = p.long;
				viewerState.orbit.lat = p.lat;
				viewerState.orbit.radius = p.radius;

				// Reset PT accumulation for this new view.
				this.managers.pathTracePipelineManager.reset();

				// Accumulate PT samples — one per rAF tick.
				for (let s = 0; s < samplesPerView; s++) {
					if (!viewerState.turntableTraining) break;
					await new Promise<void>(r => requestAnimationFrame(() => r()));
				}

				if (!viewerState.turntableTraining) break;

				// The main loop submits path-trace+resolve on rAF without waiting. If we copy
				// the output texture immediately when our rAF promise resolves, we can race
				// the GPU and snapshot a stale or partially-updated resolve.
				await this.device.queue.onSubmittedWorkDone();

				// Copy the resolved PT output into the dataset slot.
				// The PT output texture is already at optim-res (ow × oh).
				const ptTex = this.managers.pathTracePipelineManager.outputTexture;
				if (ptTex) {
					const enc = this.device.createCommandEncoder({ label: `prerender copy view ${i}` });
					enc.copyTextureToTexture(
						{ texture: ptTex },
						{ texture: dataset.textures[i] },
						[optimizationWidth, optimizationHeight, 1],
					);
					this.device.queue.submit([enc.finish()]);
				}

				// Store the camera matrices for this view.
				dataset.viewProjMats[i].set(this.camera.viewProjMat as Float32Array);
				dataset.viewMats[i].set(this.camera.viewMat as Float32Array);
				dataset.invViewProjMats[i].set(this.camera.viewProjInvMat as Float32Array);
				dataset.invViewMats[i].set(this.camera.viewInvMat as Float32Array);

				viewerState.multiviewPrerenderProgress = (i + 1) / numViews;
			}
		} finally {
			viewerState.multiviewPrerendering = false;
			if (viewerState.turntableTraining) {
				this.multiviewDataset = dataset;
				viewerState.multiviewDatasetReady = true;
				// Reset Adam so training starts fresh from the new dataset.
				this.managers.splatOptimizerManager.resetAdam();
				this.managers.edgeLayerBezierManager.resetAdam();
				this.managers.coarseColorLayerBezierManager.resetAdam();
				this.managers.fineColorLayerBezierManager.resetAdam();
				this.managers.edgeLayerBezierManager.resetAdcState();
				this.managers.coarseColorLayerBezierManager.resetAdcState();
				this.managers.fineColorLayerBezierManager.resetAdcState();
				this.heldDatasetIndex = -1;
				this.framesHeldOnDataset = 0;
			} else {
				dataset.destroy();
			}
		}
	}

	// ── Per-frame view resolution ─────────────────────────────────

	/**
	 * Resolve the current frame's view: either a randomly-sampled dataset slot
	 * (during multiview training) or the live camera matrices.
	 *
	 * When a dataset view is selected, the relevant GPU uniform buffers are
	 * written directly so the mesh render + optimizers match the target image.
	 */
	resolveFrameView(): TurntableFrameView {
		const viewerState = this.viewerState;
		const camera = this.camera;

		if (
			viewerState.turntableTraining
			&& viewerState.multiviewDatasetReady
			&& this.multiviewDataset
		) {
			const ds = this.multiviewDataset;
			const lingerFrames = Math.max(
				1,
				Math.floor(Number(viewerState.multiviewDisplayFramesPerView)) || 1,
			);
			if (
				this.heldDatasetIndex < 0
				|| this.framesHeldOnDataset >= lingerFrames
				|| this.heldDatasetIndex >= ds.numViews
			) {
				this.heldDatasetIndex = Math.floor(Math.random() * ds.numViews);
				this.framesHeldOnDataset = 0;
			}
			const idx = this.heldDatasetIndex;
			this.framesHeldOnDataset += 1;

			const datasetView = ds.textureViews[idx];
			const sortVp = ds.viewProjMats[idx] as Mat4;
			const vpInv = ds.invViewProjMats[idx] as Mat4;
			const invView = ds.invViewMats[idx] as Mat4;

			// Write dataset matrices into GPU buffers.
			this.managers.uniformsManager.writeViewProjMat(ds.viewProjMats[idx]);
			this.managers.uniformsManager.writeViewMat(ds.viewMats[idx]);
			this.managers.uniformsManager.writeInvViewProjMat(ds.invViewProjMats[idx]);
			this.managers.edgeLayerBezierManager.writeVPMatrix(ds.viewProjMats[idx]);
			this.managers.coarseColorLayerBezierManager.writeVPMatrix(ds.viewProjMats[idx]);
			this.managers.fineColorLayerBezierManager.writeVPMatrix(ds.viewProjMats[idx]);
			this.managers.edgeLayerBezierManager.writeVPInvMatrix(ds.invViewProjMats[idx]);
			this.managers.coarseColorLayerBezierManager.writeVPInvMatrix(ds.invViewProjMats[idx]);
			this.managers.fineColorLayerBezierManager.writeVPInvMatrix(ds.invViewProjMats[idx]);
			const sortVpFa = sortVp as Float32Array;
			this.managers.bezierForwardManager.writeVPMatrix(sortVpFa);
			this.managers.baseColorBezierForwardManager.writeVPMatrix(sortVpFa);
			this.managers.colorBezierForwardManager.writeVPMatrix(sortVpFa);

			return { datasetView, sortVp, vpInv, invView };
		}

		// Not training from dataset — use the live camera.
		this.heldDatasetIndex = -1;
		this.framesHeldOnDataset = 0;

		return {
			datasetView: null,
			sortVp: camera.viewProjMat as Mat4,
			vpInv: camera.viewProjInvMat as Mat4,
			invView: camera.viewInvMat as Mat4,
		};
	}

	// ── Cleanup ───────────────────────────────────────────────────

	destroy(): void {
		this.multiviewDataset?.destroy();
		this.multiviewDataset = null;
	}
}
