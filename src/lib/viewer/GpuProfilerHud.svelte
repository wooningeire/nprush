<script lang="ts">
import type { ViewerState } from "./ViewerState.svelte.ts";

function rowSum(row: readonly number[]): number {
    return row.reduce((a, b) => a + b, 0);
}

const {
    viewerState,
}: {
    viewerState: ViewerState;
} = $props();

const W = 172;
const H = 44;
const PAD = 4;

const frameTotals = $derived(
    viewerState.gpuProfilingHistoryFrames.map((row) => rowSum(row)),
);

const latestTotal = $derived(rowSum(viewerState.gpuProfilingMs));

const ymax = $derived(Math.max(...frameTotals, 1e-6));

const points = $derived.by(() => {
    const totals = frameTotals;
    const n = totals.length;
    if (n === 0) return "";
    const iw = W - PAD * 2;
    const ih = H - PAD * 2;
    const ym = ymax;
    return totals
        .map((t, i) => {
            const x = PAD + (n <= 1 ? iw / 2 : (i / (n - 1)) * iw);
            const y = PAD + ih - (t / ym) * ih;
            return `${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(" ");
});

const subtitle = $derived(
    frameTotals.length >= 2
        ? `peak ${ymax.toFixed(2)} ms / ${frameTotals.length} frames`
        : `${frameTotals.length} frame sample`,
);
</script>

{#if viewerState.gpuTimestampQuerySupported && viewerState.gpuProfilingEnabled}
    <aside class="gpu-hud" aria-label="GPU frame time HUD">
        <div class="gpu-hud-inner">
            <div class="gpu-hud-num">{latestTotal.toFixed(2)}<span class="gpu-hud-unit"> ms</span></div>
            <svg class="gpu-hud-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
                {#if points}
                    <polygon
                        fill="rgba(147,197,253,0.14)"
                        points={`${PAD},${H - PAD} ${points} ${W - PAD},${H - PAD}`}
                    />
                    <polyline
                        fill="none"
                        stroke="rgba(186,230,253,0.9)"
                        stroke-width="1.5"
                        points={points}
                    />
                {/if}
                <line
                    stroke="rgba(255,255,255,0.12)"
                    stroke-width="1"
                    x1={PAD}
                    y1={H - PAD}
                    x2={W - PAD}
                    y2={H - PAD}
                />
            </svg>
            <div class="gpu-hud-sub">{subtitle}</div>
        </div>
    </aside>
{/if}

<style lang="scss">
    .gpu-hud {
        position: fixed;
        right: max(12px, env(safe-area-inset-right));
        bottom: max(12px, env(safe-area-inset-bottom));
        z-index: 92;
        pointer-events: none;
    }

    .gpu-hud-inner {
        pointer-events: auto;
        min-width: 176px;
        padding: 0.45rem 0.55rem 0.4rem;
        background: rgba(8, 10, 16, 0.72);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        backdrop-filter: blur(10px);
        font-variant-numeric: tabular-nums;
    }

    .gpu-hud-num {
        font-size: 1rem;
        font-weight: 600;
        color: rgba(226, 232, 255, 0.98);
        line-height: 1.1;

        .gpu-hud-unit {
            font-size: 0.68rem;
            font-weight: 500;
            color: rgba(255, 255, 255, 0.42);
            margin-left: 0.1rem;
        }
    }

    .gpu-hud-svg {
        display: block;
        width: 100%;
        height: 44px;
        margin-top: 0.2rem;
    }

    .gpu-hud-sub {
        margin-top: 0.2rem;
        font-size: 0.56rem;
        color: rgba(255, 255, 255, 0.4);
        line-height: 1.25;
        max-width: 200px;
    }
</style>
