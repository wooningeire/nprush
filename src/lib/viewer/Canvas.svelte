<script lang="ts">
import {Draggable} from "@vaie/hui";
import type { ViewerState } from "./ViewerState.svelte";
import { STRIP_HEIGHT_FRAC } from "$/util";

let {
    viewerState,
    canvas = $bindable(),
}: {
    viewerState: ViewerState,
    canvas?: HTMLCanvasElement | null;
} = $props();

// Order and length must match NUM_PANELS in splat_render.wgsl. Each label
// is positioned at i / STRIP_LABELS.length, so a missing entry shifts
// every subsequent label out of alignment with its panel.
const STRIP_LABELS = [
    "Target Color",
    "Splat Color",
    "Target Depth",
    "Target Edges",
    "Splat Edges",
    "Bezier Edges",
];
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

    <div class="labels" style:--strip-frac={STRIP_HEIGHT_FRAC}>
        <div class="label main left">Target</div>
        <div class="label main right">Splats</div>
        {#each STRIP_LABELS as text, i}
            <div
                class="label strip"
                style:left="{(i / STRIP_LABELS.length) * 100}%"
                style:width="{(1 / STRIP_LABELS.length) * 100}%"
            >{text}</div>
        {/each}
    </div>
</section>

<style lang="scss">
section {
    width: 100vw;
    height: 100vh;
    position: relative;
}

.labels {
    position: absolute;
    inset: 0;
    pointer-events: none;
    font-family: system-ui, sans-serif;
    color: rgba(255, 255, 255, 0.85);
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.9);
    user-select: none;
}

.label {
    position: absolute;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.label.main {
    top: 8px;
    font-size: 12px;
}
.label.main.left {
    left: 12px;
}
.label.main.right {
    right: 12px;
    left: auto;
}

.label.strip {
    top: calc((1 - var(--strip-frac)) * 100% + 4px);
    text-align: center;
    padding: 0 4px;
    box-sizing: border-box;
    font-size: 10px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
</style>