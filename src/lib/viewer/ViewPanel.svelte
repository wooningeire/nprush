<script lang="ts">
import { onDestroy, onMount } from "svelte";

let {
    canvas = $bindable(),
}: {
    canvas: HTMLCanvasElement | null,
} = $props();

let width = $state<number | null>(null);
let height = $state<number | null>(null);

$effect(() => {
    if (width === null || height === null || canvas === null) return;

    canvas.width = devicePixelRatio * width;
    canvas.height = devicePixelRatio * height;
});


let devicePixelRatio = $state(1);
let mediaQueryList: MediaQueryList | null = null;
const listenForDevicePixelRatioChange = () => {
    mediaQueryList = matchMedia(`screen and (resolution: ${devicePixelRatio}dppx)`);
    mediaQueryList.addEventListener("change", listenForDevicePixelRatioChange, {once: true});
};

onMount(() => {
    listenForDevicePixelRatioChange();
});

onDestroy(() => {
    mediaQueryList?.removeEventListener("change", listenForDevicePixelRatioChange);
});
</script>

<view-panel
    bind:clientWidth={() => null, newWidth => width = newWidth}
    bind:clientHeight={() => null, newHeight => height = newHeight}
>
    <canvas bind:this={canvas}></canvas>
</view-panel>

<style lang="scss">
view-panel {
    display: grid;
    place-items: center;
}

canvas {
    max-width: 100%;
    max-height: 100%;
}
</style>