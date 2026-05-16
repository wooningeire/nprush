<script lang="ts">
import { onDestroy } from "svelte";
import { Draggable, Hotkey } from "@vaie/hui";
import { CameraOrbit } from "../CameraOrbit.svelte.ts";
import { MaterialSphereRenderer } from "./MaterialSphereRenderer.ts";
import ViewPanel from "../ViewPanel.svelte";

let {
    active,
}: {
    active: boolean,
} = $props();

let canvas = $state<HTMLCanvasElement | null>(null);
let renderer: MaterialSphereRenderer | null = null;
let rendererInitializing = false;
let animFrameId = 0;

const orbit = new CameraOrbit();

let shiftHeld = $state(false);

// --- lifecycle ---

// Lazily create the renderer once the canvas has non-zero pixel dimensions
// (ViewPanel sets canvas.width/height reactively from clientWidth/clientHeight,
//  which are 0 while the parent view-mode-container is display:none).
async function ensureRenderer() {
    if (renderer || rendererInitializing || !canvas) return;
    if (canvas.width === 0 || canvas.height === 0) return;

    rendererInitializing = true;
    try {
        renderer = await MaterialSphereRenderer.create(canvas);
    } finally {
        rendererInitializing = false;
    }
}

onDestroy(() => {
    stopLoop();
    renderer?.destroy();
    renderer = null;
});

// Start/stop loop when active prop changes
$effect(() => {
    if (active) {
        startLoop();
    } else {
        stopLoop();
    }
});

function startLoop() {
    if (animFrameId !== 0) return;
    const loop = async () => {
        await ensureRenderer();
        if (renderer) {
            renderer.render(orbit.view, orbit.viewInv);
        }
        animFrameId = requestAnimationFrame(loop);
    };
    animFrameId = requestAnimationFrame(loop);
}

function stopLoop() {
    if (animFrameId !== 0) {
        cancelAnimationFrame(animFrameId);
        animFrameId = 0;
    }
}
</script>

<Hotkey
    key="Shift"
    onKeyDown={() => shiftHeld = true}
    onKeyUp={() => shiftHeld = false}
/>

<material-creator-content>
    <div class="control-panel">
        <div class="section-title">Material Preview</div>

        <div class="separator"></div>

        <div class="info-text">
            Drag with <b>middle mouse</b> to orbit.<br />
            Hold <b>Shift + middle mouse</b> to pan.<br />
            <b>Scroll</b> to zoom.
        </div>

        <div class="separator"></div>

        <div class="slider-group">
            <label>
                Zoom: {orbit.radius.toFixed(2)}
                <input type="range" min="0.5" max="5" step="0.05" bind:value={orbit.radius} />
            </label>
        </div>
    </div>

    <Draggable
        onDown={({ button, pointerEvent }) => {
            if (button === 1) {
                pointerEvent.preventDefault();
            }
        }}

        onDrag={({ movement, button, pointerEvent }) => {
            switch (button) {
                case 1:
                    if (shiftHeld) {
                        orbit.pan(movement);
                    } else {
                        orbit.turn(movement);
                    }
                    pointerEvent.preventDefault();
                    break;
                default:
                    break;
            }
            pointerEvent.preventDefault();
        }}

        onUp={() => {
            document.exitPointerLock();
        }}
    >
        {#snippet dragTarget({ onpointerdown })}
            <sphere-viewport
                {onpointerdown}
                oncontextmenu={(event: PointerEvent) => event.preventDefault()}
                onwheel={(event: WheelEvent) => {
                    orbit.zoom(event.deltaY);
                    event.preventDefault();
                }}
                role="presentation"
            >
                <ViewPanel
                    bind:canvas
                    label="material"
                />
            </sphere-viewport>
        {/snippet}
    </Draggable>
</material-creator-content>

<style lang="scss">
material-creator-content {
    flex-grow: 1;

    display: flex;
    align-items: stretch;

    overflow: hidden;
}

.control-panel {
    overflow-y: auto;
    padding: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.1);

    .section-title {
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: rgba(255, 255, 255, 0.5);
        margin-bottom: 0.35rem;
    }

    .separator {
        height: 1px;
        background: rgba(255, 255, 255, 0.15);
        margin: 0.5rem 0;
    }

    .info-text {
        color: rgba(255, 255, 255, 0.55);
        font-size: 0.8rem;
        line-height: 1.5;
        max-width: 30ch;
    }

    .slider-group {
        margin-bottom: 0.25rem;

        label {
            color: rgba(255, 255, 255, 0.7);
            display: flex;
            flex-direction: column;
            gap: 0.15rem;
            cursor: pointer;
            user-select: none;
        }

        input[type="range"] {
            width: 100%;
            accent-color: #a855f7;
            cursor: pointer;
        }
    }
}

sphere-viewport {
    flex-grow: 1;

    display: grid;
    align-items: stretch;
}
</style>
