<script lang="ts">
import type { PaintModelingState } from "./PaintModelingState.svelte.ts";
import type { Vec3 } from "./types.ts";

let {
    modelerState,
    planePickArmed,
    requestRender,
    setPlanePickArmed,
}: {
    modelerState: PaintModelingState,
    planePickArmed: boolean,
    requestRender: () => void,
    setPlanePickArmed: (value: boolean) => void,
} = $props();

const normalAxes = [
    { label: "X", value: [1, 0, 0] as Vec3 },
    { label: "Y", value: [0, 1, 0] as Vec3 },
    { label: "Z", value: [0, 0, 1] as Vec3 },
];

const setNormalComponent = (index: number, value: number): void => {
    if (!Number.isFinite(value)) return;
    const normal = [...modelerState.constructionPlane.normal] as Vec3;
    normal[index] = value;
    modelerState.setConstructionPlaneNormal(normal);
    requestRender();
};

const formatValue = (value: number): string => (
    Number.isFinite(value) ? value.toFixed(3) : "0.000"
);
</script>

<div class="plane-editor" aria-label="Construction plane">
    <div class="plane-actions">
        <button
            type="button"
            class:active={planePickArmed}
            aria-pressed={planePickArmed}
            onclick={() => setPlanePickArmed(!planePickArmed)}
        >Pick</button>
        <button
            type="button"
            onclick={() => {
                modelerState.alignConstructionPlaneToView();
                requestRender();
            }}
        >View</button>
        <button
            type="button"
            onclick={() => {
                modelerState.flipConstructionPlaneNormal();
                requestRender();
            }}
        >Flip</button>
    </div>
    <label class="numeric-row">
        <span>Depth</span>
        <input
            aria-label="Construction plane depth"
            type="number"
            min="0.06"
            step="0.01"
            value={formatValue(modelerState.constructionPlaneViewDepth)}
            onchange={(event) => {
                modelerState.setConstructionPlaneViewDepth(
                    Number((event.currentTarget as HTMLInputElement).value),
                );
                requestRender();
            }}
        />
    </label>
    <div class="normal-row">
        <span>Normal</span>
        <div class="vector-inputs">
            {#each modelerState.constructionPlane.normal as value, index}
                <input
                    aria-label={`Construction plane normal ${["X", "Y", "Z"][index]}`}
                    type="number"
                    step="0.01"
                    value={formatValue(value)}
                    onchange={(event) => {
                        setNormalComponent(
                            index,
                            Number((event.currentTarget as HTMLInputElement).value),
                        );
                    }}
                />
            {/each}
        </div>
    </div>
    <div class="axis-row">
        <span>Align</span>
        <div class="axis-buttons">
            {#each normalAxes as axis}
                <button
                    type="button"
                    aria-label={`Align construction plane to ${axis.label} axis`}
                    onclick={() => {
                        modelerState.setConstructionPlaneNormal(axis.value);
                        requestRender();
                    }}
                >{axis.label}</button>
            {/each}
        </div>
    </div>
</div>

<style lang="scss">
.plane-editor {
    display: grid;
    gap: 0.45rem;
    padding: 0.55rem;
    border: 1px solid oklch(76% 0.045 220 / 0.24);
    border-radius: 6px;
    background: oklch(19% 0.018 220 / 0.58);
}

.numeric-row,
.normal-row,
.axis-row {
    display: grid;
    grid-template-columns: 4.8rem minmax(0, 1fr);
    align-items: center;
    gap: 0.5rem;
    color: oklch(82% 0.012 210 / 0.72);
    font-size: 0.78rem;
}

.numeric-row input,
.vector-inputs input {
    min-width: 0;
    height: 1.8rem;
    padding: 0 0.4rem;
    border: 1px solid oklch(78% 0.018 210 / 0.22);
    border-radius: 4px;
    background: oklch(18% 0.018 210 / 0.86);
    color: oklch(92% 0.012 210);
    font: inherit;
}

.plane-actions,
.axis-buttons {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.3rem;

    button {
        min-width: 0;
        min-height: 1.75rem;
        border: 1px solid oklch(78% 0.018 210 / 0.2);
        border-radius: 6px;
        background: oklch(94% 0.01 210 / 0.08);
        color: oklch(90% 0.012 210 / 0.9);
        font-size: 0.72rem;
        cursor: pointer;

        &:hover {
            background: oklch(94% 0.01 210 / 0.14);
        }
    }
}

.plane-actions button.active {
    border-color: oklch(78% 0.12 190 / 0.64);
    background: oklch(62% 0.11 190 / 0.3);
    color: oklch(94% 0.045 190);
}

.vector-inputs {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.3rem;

    input {
        width: 100%;
        padding-inline: 0.25rem;
        text-align: center;
    }
}
</style>