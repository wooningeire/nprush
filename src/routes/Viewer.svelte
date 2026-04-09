<script lang="ts">
import { onMount } from "svelte";

import Canvas from "./Canvas.svelte";
import { SimulationState } from "./SimulationState.svelte";

let canvas = $state<HTMLCanvasElement | null>(null);
let canvasPromise = Promise.withResolvers<HTMLCanvasElement>();

const simulationState = SimulationState.mount({
    canvasPromise: canvasPromise.promise,
});

onMount(() => {
    canvasPromise.resolve(canvas!);
});
</script>

<main>
    <Canvas
        {simulationState}
        bind:canvas
    />

</main>


<style lang="scss">
main {
    width: 100vw;
    height: 100vh;

    display: grid;

    > :global(*) {
        grid-area: 1/1;
    }
}
</style>