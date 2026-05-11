<script lang="ts">
import {
    GPU_PROFILER_LABELS,
    GPU_PROFILER_PAIR_COUNT,
} from "$/gpu/performanceMeasurement/gpuProfilerPairs";
import type { ViewerState } from "./ViewerState.svelte.ts";

function rowSum(row: readonly (number | null)[]): number {
    return row.reduce((a, b) => a + (b ?? 0), 0);
}

const {
    viewerState,
}: {
    viewerState: ViewerState;
} = $props();

const W = 220;
const H = 72;
const PAD = 6;

const frameTotals = $derived(
    viewerState.gpuProfilingHistoryFrames.map((row) => rowSum(row)),
);

const latestTotal = $derived(rowSum(viewerState.gpuProfilingMs));

const sparkYmax = $derived(Math.max(...frameTotals, 1e-6));

const sparkPoints = $derived.by(() => {
    const totals = frameTotals;
    const n = totals.length;
    if (n === 0) return "";
    const iw = W - PAD * 2;
    const ih = H - PAD * 2;
    const ym = sparkYmax;
    return totals
        .map((t, i) => {
            const x = PAD + (n <= 1 ? iw / 2 : (i / (n - 1)) * iw);
            const y = PAD + ih - (t / ym) * ih;
            return `${x.toFixed(1)},${y.toFixed(1)}`;
        })
        .join(" ");
});

const barMaxMs = $derived.by(() => {
    const msValues = viewerState.gpuProfilingMs.filter((v): v is number => v !== null);
    return msValues.length > 0 ? Math.max(...msValues, 1e-6) : 1e-6;
});
</script>

{#if viewerState.gpuTimestampQuerySupported && viewerState.gpuProfilingEnabled}
    <fieldset class="gpu-profiler-charts">
        <legend>GPU profiling</legend>
        <p class="gpu-profiler-intro">
            Per-pass GPU time (Chrome / timestamp queries). Charts show the latest sampled frame unless noted.
        </p>

        <div class="gpu-profiler-metrics">
            <div class="gpu-profiler-row gpu-profiler-summary">
                <span class="gpu-profiler-summary-label">Frame total</span>
                <span class="gpu-profiler-ms">{latestTotal.toFixed(2)} ms</span>
            </div>
            {#if viewerState.gpuProfilingHistoryFrames.length}
                <div class="gpu-chart-block">
                    <div class="gpu-profiler-chart-title">Frame GPU time ({viewerState.gpuProfilingHistoryFrames.length} frames)</div>
                    <svg class="gpu-spark-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
                        {#if sparkPoints}
                            <polygon
                                fill="rgba(147,197,253,0.12)"
                                points={`${PAD},${H - PAD} ${sparkPoints} ${W - PAD},${H - PAD}`}
                            />
                            <polyline
                                fill="none"
                                stroke="rgba(186,230,253,0.85)"
                                stroke-width="2"
                                points={sparkPoints}
                            />
                        {/if}
                        <line
                            stroke="rgba(255,255,255,0.1)"
                            stroke-width="1"
                            x1={PAD}
                            y1={H - PAD}
                            x2={W - PAD}
                            y2={H - PAD}
                        />
                    </svg>
                    <div class="gpu-chart-sub">Peak in window ≈ {sparkYmax.toFixed(3)} ms</div>
                </div>
            {/if}
        </div>

        {#if viewerState.gpuProfilingMs.length === GPU_PROFILER_PAIR_COUNT}
            <div class="gpu-pass-bars-heading">Latest frame — passes (total {latestTotal.toFixed(2)} ms)</div>
            <div class="gpu-pass-bars" role="list">
                {#each viewerState.gpuProfilingMs as ms, idx (idx)}
                    <div class="gpu-pass-row" role="listitem">
                        <div class="gpu-pass-label">{GPU_PROFILER_LABELS[idx] ?? idx}</div>
                        <div class="gpu-pass-bar-track" aria-valuemin={0} aria-valuemax={barMaxMs} aria-valuenow={ms ?? 0}>
                            <div
                                class="gpu-pass-bar-fill"
                                style={`width: ${ms === null ? 0 : Math.min((ms / barMaxMs) * 100, 100)}%; opacity: ${ms === null ? 0.2 : 1};`}
                            />
                        </div>
                        <div class="gpu-pass-ms">{ms === null ? "-.--" : ms.toFixed(2)} ms</div>
                    </div>
                {/each}
            </div>
        {/if}
    </fieldset>
{/if}

<style lang="scss">
    .gpu-profiler-charts {
        border-radius: var(--radii-rounded);
        border: 1px solid rgba(var(--accent-rgb) / var(--accent-opacity-strong));
        background: rgb(var(--main-bg-alt-rgb) / 0.6);
        margin: 8px -4px 0;
        padding-inline: calc(14px / 14 * 8);
        scroll-margin-top: calc(56px / 14 * 1);

        legend {
            padding-inline: 0.55rem;
        }

        :global(details) &.gpu-profiler-charts {
            border-color: rgb(var(--main-text-rgb) / 14%);
            background-color: unset;
            margin-inline: unset;
            margin-top: unset;
            margin-bottom: unset;
            padding-inline: unset;
            padding-block: unset;
            container-type: unset;
            container-name: unset;
        }
    }

    .gpu-profiler-intro {
        font-size: 0.6875rem;
        line-height: 1.38;
        color: rgb(var(--main-text-muted-rgb) / 94%);
        margin: 6px 0 8px;
    }

    .gpu-profiler-metrics {
        display: flex;
        flex-direction: column;
        gap: 0.45rem;
        margin-block-end: 0.45rem;

        &:last-child {
            margin-block-end: 4px;
        }
    }

    .gpu-profiler-row {
        display: flex;
        align-items: center;
        gap: 0.45rem;

        &.gpu-profiler-summary {
            border-bottom: none;
            font-weight: 600;
            font-variant-numeric: tabular-nums;
        }

        &.gpu-profiler-summary .gpu-profiler-ms {
            flex: unset;
            text-align: end;
            min-width: 4.75rem;
        }
    }

    .gpu-profiler-summary-label {
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-variant-numeric: tabular-nums;
    }

    .gpu-chart-block {
        background: rgb(var(--main-bg-alt-rgb) / 52%);
        border-radius: calc(var(--radii-rounded) / 2);
        border: 1px solid rgb(var(--main-bg-rgb) / 92%);
        padding: 8px;

    }

    .gpu-profiler-chart-title {
        font-variant-numeric: tabular-nums;
        letter-spacing: 0.025em;
        font-size: 0.6875rem;
        color: rgb(var(--main-text-muted-rgb) / var(--muted-opacity-less));
        margin-bottom: 8px;

        &::after {
            content: "";
            display: block;
            margin-top: 6px;
            height: 1px;
            background: linear-gradient(to right, transparent, rgb(var(--main-text-rgb) / 35%), transparent);
        }
    }

    .gpu-spark-svg {
        display: block;
        width: 100%;
        height: min(112px, 22vw);
    }

    .gpu-chart-sub {
        margin-top: 6px;
        font-variant-numeric: tabular-nums;
        letter-spacing: 0.035em;
        font-size: 0.59375rem;
        color: rgb(var(--main-text-muted-rgb) / var(--muted-opacity-less));

        &::before {
            content: "";
            display: block;
            margin-bottom: 4px;
            height: 1px;
            background: linear-gradient(to right, transparent, rgb(var(--main-text-rgb) / 32%), transparent);
        }
    }

    .gpu-pass-bars-heading {
        font-variant-numeric: tabular-nums;
        letter-spacing: 0.025em;
        font-size: 0.6875rem;
        color: rgb(var(--main-text-muted-rgb) / var(--muted-opacity-less));
        margin: 0.6rem 0 0.45rem;

        &::after {
            content: "";
            display: block;
            margin-top: 6px;
            height: 1px;
            background: linear-gradient(to right, transparent, rgb(var(--main-text-rgb) / 35%), transparent);
        }
    }

    .gpu-pass-bars {
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
        margin-block-end: 4px;
    }

    .gpu-pass-row {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(96px, 2fr) 4rem;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.625rem;
    }

    .gpu-pass-label {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-variant-numeric: tabular-nums;
    }

    .gpu-pass-bar-track {
        height: 6px;
        border-radius: 999px;
        background: rgb(var(--main-bg-rgb) / 92%);
        border: 1px solid rgb(var(--main-text-rgb) / 8%);
        overflow: hidden;
    }

    .gpu-pass-bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(
            to right,
            rgb(var(--accent-muted-rgb) / 45%),
            rgb(var(--accent-rgb) / 78%),
            rgb(var(--accent-strong-rgb) / 94%)
        );
    }

    .gpu-pass-ms {
        justify-self: end;
        font-variant-numeric: tabular-nums;
        color: rgb(var(--main-text-muted-rgb) / 92%);
        min-width: 3.75rem;
        text-align: end;
    }
</style>
