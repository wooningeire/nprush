<script lang="ts">
import {Draggable, Hotkey} from "@vaie/hui";
import type { ViewerState } from "./ViewerState.svelte";
import { STRIP_HEIGHT_FRAC } from "$/util";

let {
    viewerState,
    canvases = $bindable(),
}: {
    viewerState: ViewerState,
    canvases?: Record<string, HTMLCanvasElement>;
} = $props();


const STRIP_LABELS = [
    { id: "splatColor", text: "Splat Color" },
    { id: "targetDepth", text: "Target Depth" },
    { id: "targetEdges", text: "Target Edges" },
    { id: "edgeBeziers", text: "Edge Beziers" },
    { id: "coarseBezier", text: "Coarse bezier" },
    { id: "fineBezier", text: "Fine bezier" },
];

let shiftHeld = $state(false);
let container = $state<HTMLElement | null>(null);

$effect(() => {
    if (!canvases || !container) return;

    const ro = new ResizeObserver(() => {
        for (const canvas of Object.values(canvases)) {
            if (!canvas) continue;
            // Use parent because canvas itself might be 100% of parent
            const rect = canvas.parentElement!.getBoundingClientRect();

            const w = Math.round(rect.width * devicePixelRatio);
            const h = Math.round(rect.height * devicePixelRatio);
            if (canvas.width !== w || canvas.height !== h) {
                canvas.width = w;
                canvas.height = h;
            }
        }
    });

    ro.observe(container);
    return () => ro.disconnect();
});


$effect(() => {
    if (!canvases) canvases = {};
});
</script>

<Hotkey
    key="Shift"
    onKeyDown={() => shiftHeld = true}
    onKeyUp={() => shiftHeld = false}
/>

<section
    bind:clientWidth={null, clientWidth => viewerState.width = clientWidth!}
    bind:clientHeight={null, clientHeight => viewerState.height = clientHeight!}
>
    <Draggable
        onDown={({ button, pointerEvent }) => {
            if (button === 1) {
                pointerEvent.preventDefault();
            } else if (button === 0) {
                const rect = (pointerEvent.currentTarget as HTMLElement).getBoundingClientRect();
                viewerState.onPaintDrag(pointerEvent.clientX - rect.left, pointerEvent.clientY - rect.top);
                pointerEvent.preventDefault();
            }
        }}

        onDrag={async ({ movement, button, pointerEvent }) => {
            switch (button) {
                case 1:
                    if (shiftHeld) {
                        viewerState.orbit.pan(movement);
                    } else {
                        viewerState.orbit.turn(movement);
                    }

                    pointerEvent.preventDefault();
                    break;
                
                case 0: {
                    const rect = (pointerEvent.currentTarget as HTMLElement).getBoundingClientRect();
                    viewerState.onPaintDrag(pointerEvent.clientX - rect.left, pointerEvent.clientY - rect.top);
                    break;
                }
                
                case 2:
                    break;

                default:
                    break;
            }

            pointerEvent.preventDefault();
        }}

        onUp={({ pointerEvent }) => {
            if (pointerEvent.button === 2) {
            } else {
                document.exitPointerLock();
            }
        }}
    >
        {#snippet dragTarget({ onpointerdown })}
            <div
                bind:this={container}
                class="views-container"
                {onpointerdown}
                oncontextmenu={event => event.preventDefault()}
                onwheel={event => {
                    viewerState.orbit.zoom(event.deltaY);
                    event.preventDefault();
                }}
                role="presentation"
            >
                <div class="main-views">
                    <div class="view-panel">
                        <div class="label main left">Target</div>
                        <canvas bind:this={canvases.target}></canvas>
                    </div>
                    <div class="view-panel separator"></div>
                    <div class="view-panel">
                        <div class="label main right">Splats</div>
                        <canvas bind:this={canvases.splats}></canvas>
                    </div>
                </div>
                <div class="strip-views" style:--strip-frac={STRIP_HEIGHT_FRAC}>
                    {#each STRIP_LABELS as { id, text }}
                        <div class="view-panel">
                            <div class="label strip">{text}</div>
                            <canvas bind:this={canvases[id]}></canvas>
                        </div>
                    {/each}
                </div>
            </div>
        {/snippet}
    </Draggable>
</section>

<style lang="scss">
section {
    width: 100%;
    height: 100vh;
    position: relative;
}

.views-container {
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
}

.main-views {
    flex: 1;
    display: flex;
    flex-direction: row;
}

.strip-views {
    height: calc(var(--strip-frac) * 100%);
    display: flex;
    flex-direction: row;
    border-top: 2px solid #4d4d4d;
}

.view-panel {
    flex: 1;
    position: relative;
    border-right: 2px solid #4d4d4d;
    background: #0d0d0d;
}

.view-panel:last-child {
    border-right: none;
}

.view-panel.separator {
    flex: 0 0 2px;
    background: #fff;
    border: none;
}

.view-panel canvas {
    width: 100%;
    height: 100%;
    display: block;
    object-fit: cover;
}

.label {
    position: absolute;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.85);
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.9);
    pointer-events: none;
    z-index: 10;
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
    top: 4px;
    width: 100%;
    text-align: center;
    padding: 0 4px;
    box-sizing: border-box;
    font-size: 10px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
</style>