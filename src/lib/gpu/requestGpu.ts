export const requestGpu = async ({
    onStatusChange,
    onErr,
}: {
    onStatusChange?: (text: string) => void,
    onErr?: (text: string) => void,
}) => {
    onStatusChange?.("accessing gpu adapter");
    if (navigator.gpu === undefined) {
        onErr?.("webgpu not supported by your browser");
        return null;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (adapter === null) {
        onErr?.("could not get gpu adapter; may need to restart your device");
        return null;
    }

    const supportsTimestamp = adapter.features.has("timestamp-query");
    const requiredFeatures: GPUFeatureName[] = [];
    if (supportsTimestamp) {
        requiredFeatures.push("timestamp-query");
    }

    onStatusChange?.("accessing gpu device");
    const device = await adapter.requestDevice({
        requiredFeatures,
        // requiredLimits: {
        //     maxStorageBufferBindingSize: Math.min(536_870_912, adapter.limits.maxStorageBufferBindingSize),
        //     maxStorageBuffersPerShaderStage: 10,
        // },
    });
    if (device === null) {
        onErr?.("could not get gpu device; may need to restart your device");
        return null;
    }

    device.lost.then(() => {
        onErr?.("gpu device was lost/hung or a shader took too long");
    });

    if (typeof window !== "undefined") {
        (window as any).__wgpu_errors = [];
        device.addEventListener("uncapturederror", (e: any) => {
            (window as any).__wgpu_errors.push(e.error.message);
        });
    }

    const format = navigator.gpu.getPreferredCanvasFormat();

    return {
        device,
        format,
        supportsTimestamp,
    };
};