import { dismissToast, showToast } from "../../viewer/toast.svelte.ts";
import type { ViewerState } from "../../viewer/ViewerState.svelte.ts";
import { requestGpu } from "./requestGpu.ts";
import { loadGlb } from "../file-load/loadGlb.ts";
import artelorianUrl from "$/assets/artelorian.glb?url";
import groundUrl from "$/assets/ground.glb?url";
import hdrUrl from "$/assets/lakeside_sunrise_2k.hdr?url";
import brushUrl from "$/assets/brush.png?url";
import groundAlbedoUrl from "$/assets/brown_mud_03_diff_2k.jpg?url";
import groundNormalUrl from "$/assets/brown_mud_03_nor_gl_2k.png?url";
import { loadHdrTexture } from "../file-load/loadHdrTexture.ts";
import { loadTexture } from "../file-load/loadTexture.ts";
import { buildBvh } from "../bvh.ts";

export const loadInitialAssetsAndGpu = async (state: ViewerState) => {
    const loadingToast = showToast("loading gpu & meshes…", "info", 0);
    const [gpu, primaryMesh, groundMesh, groundPbrMesh] = await Promise.all([
        requestGpu({
            onStatusChange: text => showToast(text, "info", 2500),
            onErr: text => showToast(text, "error", 0),
        }),
        loadGlb(artelorianUrl)
            .then(primaryMesh => {
                showToast("mesh loaded", "info", 2000);
                return primaryMesh;
            }),
        loadGlb(groundUrl, false, [1, 1, 1, 0]),
        loadGlb(groundUrl, false, [1, 1, 1, 1], 'Plane.001'),
    ]);

    dismissToast(loadingToast);

    if (gpu === null) return null;



    state.gpuTimestampQuerySupported = gpu.supportsTimestamp;

    const bvhToast = showToast("building BVH…", "info", 0);
    state.meshVerts = new Float32Array(primaryMesh.vertices);
    state.meshBvh = buildBvh(state.meshVerts, new Uint32Array(primaryMesh.indices));
    dismissToast(bvhToast);
    showToast("BVH ready", "info", 2000);

    const texturesToast = showToast("loading textures…", "info", 0);
    const [envTexture, brushTexture, groundAlbedoTexture, groundNormalTexture] = await Promise.all([
        loadHdrTexture(gpu.device, hdrUrl)
            .then(texture => {
                showToast("environment loaded", "info", 2000);
                return texture;
            }),
        loadTexture(gpu.device, brushUrl),
        loadTexture(gpu.device, groundAlbedoUrl),
        loadTexture(gpu.device, groundNormalUrl)
            .then(texture => {
                showToast("textures loaded", "info", 2000);
                return texture;
            }),
    ]);
    dismissToast(texturesToast);

    return {
        gpu,
        primaryMesh,
        groundMesh,
        groundPbrMesh,
        envTexture,
        brushTexture,
        groundAlbedoTexture,
        groundNormalTexture,
    };
};