<script lang="ts">
import {Draggable, Hotkey} from "@vaie/hui";
import type { ViewerState } from "./ViewerState.svelte";
    import ViewPanel from "./ViewPanel.svelte";
    import ViewSeparator from "./ViewSeparator.svelte";
    import ViewPanelControlBox from "./ViewPanelControlBox.svelte";

let {
    viewerState,
    canvases = $bindable({}),
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


$effect(() => {
    if (!canvases) canvases = {};
});
</script>

<Hotkey
    key="Shift"
    onKeyDown={() => shiftHeld = true}
    onKeyUp={() => shiftHeld = false}
/>

<Draggable
    onDown={({ button, pointerEvent }) => {
        if (button === 1) {
            pointerEvent.preventDefault();
        } else if (button === 0) {
            const target = pointerEvent.target as HTMLElement;
            if (target.tagName.toLowerCase() === 'canvas') {
                const rect = target.getBoundingClientRect();
                viewerState.onPaintDrag(pointerEvent.clientX - rect.left, pointerEvent.clientY - rect.top, rect.width, rect.height);
            }
            pointerEvent.preventDefault();
        }
    }}

    onDrag={async ({ movement, button, pointerEvent }) => {
        switch (button) {
            case 1:
                if (shiftHeld) {
                    viewerState.viewportOrbit.pan(movement);
                } else {
                    viewerState.viewportOrbit.turn(movement);
                }

                pointerEvent.preventDefault();
                break;
            
            case 0: {
                const target = pointerEvent.target as HTMLElement;
                if (target.tagName.toLowerCase() === 'canvas') {
                    const rect = target.getBoundingClientRect();
                    viewerState.onPaintDrag(pointerEvent.clientX - rect.left, pointerEvent.clientY - rect.top, rect.width, rect.height);
                }
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
        <views-container
            {onpointerdown}
            oncontextmenu={(event: PointerEvent) => event.preventDefault()}
            onwheel={(event: WheelEvent) => {
                viewerState.viewportOrbit.zoom(event.deltaY);
                event.preventDefault();
            }}
            role="presentation"
        >
            <view-panels-primary>
                <ViewPanel
                    bind:canvas={canvases.target}
                    label="target"
                />
                
                <ViewSeparator />

                <ViewPanel
                    bind:canvas={canvases.splats}
                    label="render"
                >
                    <view-panel-overlays>
                        {#if viewerState.viewportRenderingFrozen}
                            <viewport-frozen-overlay>Frozen</viewport-frozen-overlay>
                        {/if}

                        <ViewPanelControlBox>
                            <label>
                                Render res:
                                
                                <input type="range" min="128" max="2048" step="128"
                                    value={viewerState.renderWidth}
                                    oninput={(e) => {
                                        const v = parseInt((e.target as HTMLInputElement).value);
                                        viewerState.renderWidth = v;
                                        viewerState.renderHeight = v;
                                    }}
                                    disabled={viewerState.isCapturing || viewerState.isTurntableRendering}
                                />

                                {viewerState.renderWidth}×{viewerState.renderHeight}
                            </label>
        
                            <label>
                                <input type="checkbox" bind:checked={viewerState.viewportRenderingFrozen} />
                                Freeze preview (faster)
                            </label>
                        </ViewPanelControlBox>
                    </view-panel-overlays>
                </ViewPanel>
            </view-panels-primary>

            <ViewSeparator />

            <view-panels-strip>
                {#each STRIP_LABELS as { id, text }, i}
                    <ViewPanel
                        bind:canvas={canvases[id]}
                        label={text}
                    >
                    </ViewPanel>
                    {#if i < STRIP_LABELS.length - 1}
                        <ViewSeparator />
                    {/if}
                {/each}
            </view-panels-strip>
        </views-container>
    {/snippet}
</Draggable>

<style lang="scss">
views-container {
    flex-grow: 1;

    display: flex;
    flex-direction: column;
}

view-panels-primary {
    flex-grow: 1;

    display: flex;
    align-items: stretch;

    > :global(:not(view-separator)) {
        flex-grow: 1;
        flex-shrink: 1;
        flex-basis: 0;
    }
}

view-panels-strip {
    height: 20%;

    display: flex;

    > :global(:not(view-separator)) {
        flex-grow: 1;
        flex-shrink: 1;
        flex-basis: 0;
    }
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

view-panel-overlays {
    display: grid;
    place-items: stretch;

    > :global(*) {
        grid-area: 1/1;
    }
}

viewport-frozen-overlay {
    display: grid;
    place-items: center;

    font-size: 1.5rem;

    background: oklch(0.1 0.05 300 / 0.5);
}
</style>