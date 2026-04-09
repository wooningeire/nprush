<script lang="ts">
import {Draggable} from "@vaie/hui";
import type { ViewerState } from "./ViewerState.svelte";

let {
    viewerState,
    canvas = $bindable(),
}: {
    viewerState: ViewerState,
    canvas?: HTMLCanvasElement | null;
} = $props();
</script>

<section
    bind:clientWidth={null, clientWidth => viewerState.width = clientWidth!}
    bind:clientHeight={null, clientHeight => viewerState.height = clientHeight!}
>
    <Draggable
        onDown={({ pointerEvent }) => {
            // if (pointerEvent.button === 2) {
            //     // Right click: Interact
            //     simulationState.onInteractionStart(pointerEvent.clientX, pointerEvent.clientY, canvas!);
            // } else {
            //     // Left/Middle: Camera
            //     canvas?.requestPointerLock();
            // }
        }}

        onDrag={async ({ movement, button, pointerEvent }) => {
            switch (button) {
                case 0:
                    viewerState.orbit.turn(movement);
                    break;

                case 1:
                    viewerState.orbit.pan(movement);
                    break;
                
                case 2:
                    // viewerState.onInteractionDrag(pointerEvent.clientX, pointerEvent.clientY, canvas!);
                    break;
            }

            pointerEvent.preventDefault();
        }}

        onUp={({ pointerEvent }) => {
            if (pointerEvent.button === 2) {
                // viewerState.onInteractionEnd();
            } else {
                document.exitPointerLock();
            }
        }}
    >
        {#snippet dragTarget({ onpointerdown })}
            <canvas
                bind:this={canvas}
                width={viewerState.width}
                height={viewerState.height}
                {onpointerdown}
                oncontextmenu={(e) => { e.preventDefault(); }}
                onwheel={(event) => {
                    viewerState.orbit.radius *= 2 ** (event.deltaY * 0.001);
                    event.preventDefault();
                }}
            ></canvas>
        {/snippet}
    </Draggable>
</section>

<style lang="scss">
section {
    width: 100vw;
    height: 100vh;
}
</style>