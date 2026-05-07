import { onDestroy, onMount } from "svelte";
import { Camera } from "./Camera.svelte.ts";
import { CameraOrbit } from "./CameraOrbit.svelte.ts";
import { requestGpu } from "$/gpu/requestGpu";
import { GpuRunner } from "./GpuRunner.svelte.ts";
import { loadGlb } from "$/gpu/loadGlb";
import artelorianUrl from "$/assets/artelorian.glb?url";
import groundUrl from "$/assets/ground.glb?url";
import hdrUrl from "$/assets/lakeside_sunrise_2k.hdr?url";
import brushUrl from "$/assets/brush.png?url";
import groundAlbedoUrl from "$/assets/brown_mud_03_diff_2k.jpg?url";
import groundNormalUrl from "$/assets/brown_mud_03_nor_gl_2k.png?url";
import { loadHdrTexture } from "$/gpu/loadHdrTexture";
import { loadTexture } from "$/gpu/loadTexture";
import { buildBvh, raycastBvh, type BvhResult } from "$/gpu/bvh";
import { vec4 } from "wgpu-matrix";
import { STRIP_HEIGHT_FRAC } from "$/util";

export class ViewerState {
    width = $state(300);
    height = $state(150);
    edgeBeziersEnabled = $state(false);
    baseColorBeziersEnabled = $state(true);
    colorBeziersEnabled = $state(true);
    splatTrainingPaused = $state(false);
    edgeBezierTrainingPaused = $state(false);
    baseColorBezierTrainingPaused = $state(false);
    colorBezierTrainingPaused = $state(false);
    compareBlurred = $state(true);
    blurRadius = $state(16);
    meshSplatsEnabled = $state(true);
    
    meshVerts: Float32Array | null = null;
    meshBvh: BvhResult | null = null;
    
    runner = $state<GpuRunner | null>(null);
    
    readonly orbit = new CameraOrbit();
    readonly camera = new Camera({
        controlScheme: this.orbit,
        screenDims: { width: () => this.width, height: () => this.height },
    });

    onPaintDrag(x: number, y: number) {
        if (!this.meshBvh || !this.meshVerts || !this.runner || !this.meshSplatsEnabled) {
            console.log("onPaintDrag skipped", { bvh: !!this.meshBvh, verts: !!this.meshVerts, runner: !!this.runner, enabled: this.meshSplatsEnabled });
            return;
        }
        
        const targetWidth = this.width / 2;
        const targetHeight = this.height * (1 - STRIP_HEIGHT_FRAC);
        
        // Ignore clicks in the bottom strip
        if (y > targetHeight) return;
        
        // Map x to either the left or right half
        const localX = x % targetWidth;
        
        const ndcX = (localX / targetWidth) * 2 - 1;
        const ndcY = -((y / targetHeight) * 2 - 1);
        
        const originNdC = [ndcX, ndcY, 0, 1];
        const targetNdC = [ndcX, ndcY, 1, 1];
        
        const originWorldW = vec4.transformMat4(originNdC, this.camera.viewProjInvMat);
        const targetWorldW = vec4.transformMat4(targetNdC, this.camera.viewProjInvMat);
        
        const origin = [originWorldW[0]/originWorldW[3], originWorldW[1]/originWorldW[3], originWorldW[2]/originWorldW[3]] as [number, number, number];
        const target = [targetWorldW[0]/targetWorldW[3], targetWorldW[1]/targetWorldW[3], targetWorldW[2]/targetWorldW[3]] as [number, number, number];
        
        const dir = [target[0]-origin[0], target[1]-origin[1], target[2]-origin[2]] as [number, number, number];
        const len = Math.sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
        dir[0] /= len; dir[1] /= len; dir[2] /= len;
        
        const hit = raycastBvh(this.meshBvh, this.meshVerts, origin, dir);
        if (hit) {
            console.log("Splat Hit at:", hit.p);
            // Paint a splat with a random color and size
            const radius = 0.02 + Math.random() * 0.08;
            const color = [Math.random(), Math.random(), Math.random()] as [number, number, number];
            this.runner.meshRenderPipelineManager.addSplat(hit.p, radius, color);
        } else {
            console.log("No hit");
        }
    }

    static mount({
        canvasPromise,
        numSplats = 512,
    }: {
        canvasPromise: Promise<HTMLCanvasElement>,
        numSplats?: number,
    }) {
        const state = new ViewerState();
        
        let stopLoop: (() => void) | null = null;
        onMount(async () => {
            // Kick off mesh load and gpu request concurrently; both are awaited
            // before we build the runner since the mesh is a constructor input.
            const [gpu, mesh, groundMesh, groundPbrMesh] = await Promise.all([
                requestGpu({ canvas: await canvasPromise }),
                loadGlb(artelorianUrl),
                loadGlb(groundUrl, false, [1, 1, 1, 0]),           // Plane — specular mirror, world-space
                loadGlb(groundUrl, false, [1, 1, 1, 1], 'Plane.001'), // Plane.001 — PBR textured
            ]);
            if (!gpu) return;

            state.meshVerts = new Float32Array(mesh.vertices);
            state.meshBvh = buildBvh(state.meshVerts, new Uint32Array(mesh.indices));

            const [envTexture, brushTexture, groundAlbedoTexture, groundNormalTexture] = await Promise.all([
                loadHdrTexture(gpu.device, hdrUrl),
                loadTexture(gpu.device, brushUrl),
                loadTexture(gpu.device, groundAlbedoUrl),
                loadTexture(gpu.device, groundNormalUrl),
            ]);

            const gpuRunner = new GpuRunner({
                device: gpu.device,
                context: gpu.context,
                format: gpu.format,
                camera: state.camera,
                viewerState: state,
                mesh,
                groundMesh,
                groundPbrMesh,
                matcapTexture: envTexture,
                brushTexture,
                groundAlbedoTexture,
                groundNormalTexture,
                numSplats,
            });
            state.runner = gpuRunner;

            stopLoop = gpuRunner.loop();
        });

        onDestroy(() => {
            stopLoop?.();
            state.runner?.destroy();
        });

        return state;
    }
}