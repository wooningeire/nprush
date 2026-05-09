<script lang="ts">
import type { ViewerState } from "./ViewerState.svelte";
import {
    RENDER_MODE_MULTIVIEW,
    RENDER_MODE_SINGLE_VIEW_REALTIME,
    type RenderMode,
} from "./renderMode.ts";

const {
    viewerState,
}: {
    viewerState: ViewerState;
} = $props();

const latDeg = $derived(Math.round(viewerState.turntableLatAmplitude * 180 / Math.PI));
const radPct = $derived(Math.round(viewerState.turntableRadiusAmplitude * 100));
</script>

<div class="overlays">
    <div class="slider-group">
        <label style="flex-direction: row; justify-content: space-between;">
            Render Mode
            <select
                value={viewerState.renderMode}
                onchange={(e) =>
                    viewerState.setRenderMode((e.target as HTMLSelectElement).value as RenderMode)}
                style="background: rgba(0,0,0,0.5); color: white; border: 1px solid rgba(255,255,255,0.2); border-radius: 4px; padding: 2px 4px;"
            >
                <option value={RENDER_MODE_SINGLE_VIEW_REALTIME}>Single-view (realtime)</option>
                <option value={RENDER_MODE_MULTIVIEW}>Multiview</option>
            </select>
        </label>
    </div>
    
    <label>
        <input type="checkbox" bind:checked={viewerState.viewportRenderingFrozen} />
        Freeze Viewport Render
    </label>

    <div class="separator"></div>

    <label>
        <input type="checkbox" bind:checked={viewerState.edgeBeziersEnabled} />
        Edge Beziers
    </label>
    {#if viewerState.edgeBeziersEnabled}
        <label class="sub">
            <input type="checkbox" bind:checked={viewerState.edgeBezierTrainingPaused} />
            Pause training
        </label>
    {/if}
    <label>
        <input type="checkbox" bind:checked={viewerState.baseColorBeziersEnabled} />
        Base Color Beziers
    </label>
    {#if viewerState.baseColorBeziersEnabled}
        <label class="sub">
            <input type="checkbox" bind:checked={viewerState.baseColorBezierTrainingPaused} />
            Pause training
        </label>
    {/if}
    <label>
        <input type="checkbox" bind:checked={viewerState.colorBeziersEnabled} />
        Color Beziers
    </label>
    {#if viewerState.colorBeziersEnabled}
        <label class="sub">
            <input type="checkbox" bind:checked={viewerState.colorBezierTrainingPaused} />
            Pause training
        </label>
    {/if}
    <label>
        <input type="checkbox" bind:checked={viewerState.splatsEnabled} />
        Gaussian Splats
    </label>
    {#if viewerState.splatsEnabled}
        <label class="sub">
            <input type="checkbox" bind:checked={viewerState.splatTrainingPaused} />
            Pause training
        </label>
    {/if}
    <label>
        <input type="checkbox" bind:checked={viewerState.meshSplatsEnabled} />
        Mesh Texture Splats
    </label>
    <label>
        <input type="checkbox" bind:checked={viewerState.compareBlurred} />
        Compare Blurred
    </label>
    {#if viewerState.compareBlurred}
        <div class="slider-group" style="margin-left: 1rem; margin-bottom: 0.5rem;">
            <label style="font-size: 0.8rem; color: rgba(255, 255, 255, 0.7);">
                Radius: {viewerState.blurRadius}
                <input type="range" min="1" max="64" step="1" bind:value={viewerState.blurRadius} />
            </label>
        </div>
    {/if}

    <div class="separator"></div>

    <button
        class="render-btn"
        onclick={() => viewerState.takeScreenshot()}
        disabled={viewerState.isCapturing || viewerState.isTurntableRendering}
    >
        {#if viewerState.isCapturing}
            <div class="spinner"></div>
            Rendering…
        {:else}
            📷 Render to File
        {/if}
    </button>

    <div class="separator"></div>

    <div class="slider-group">
        <label>
            Render res: {viewerState.renderWidth}×{viewerState.renderHeight}
            <input type="range" min="256" max="2048" step="256"
                value={viewerState.renderWidth}
                oninput={(e) => {
                    const v = parseInt((e.target as HTMLInputElement).value);
                    viewerState.renderWidth = v;
                    viewerState.renderHeight = v;
                }}
                disabled={viewerState.isCapturing || viewerState.isTurntableRendering}
            />
        </label>
    </div>

    <div class="separator"></div>

    <div class="turntable-section">
        <div class="section-title">Turntable</div>

        {#if viewerState.renderMode === RENDER_MODE_MULTIVIEW}
            <button
                class="render-btn"
                class:training-active={viewerState.turntableTraining}
                onclick={() => viewerState.toggleTurntableTraining()}
                disabled={viewerState.isTurntableRendering}
            >
                {#if viewerState.multiviewPrerendering}
                    <div class="spinner"></div>
                    Prerendering… {Math.round(viewerState.multiviewPrerenderProgress * 100)}%
                {:else if viewerState.turntableTraining}
                    <div class="pulse-dot"></div>
                    Stop Multiview Training
                {:else}
                    🔄 Start Multiview Training
                {/if}
            </button>

            {#if !viewerState.multiviewDatasetReady || viewerState.turntableTraining}
                <div class="slider-group">
                    <label>
                        Views: {viewerState.multiviewNumViews}
                        <input type="range" min="8" max="128" step="8" bind:value={viewerState.multiviewNumViews}
                            disabled={viewerState.turntableTraining} />
                    </label>
                </div>
                <div class="slider-group">
                    <label>
                        PT samples/view: {viewerState.turntableMinSamplesPerView}
                        <input type="range" min="8" max="256" step="8" bind:value={viewerState.turntableMinSamplesPerView}
                            disabled={viewerState.turntableTraining} />
                    </label>
                </div>
            {/if}

            {#if viewerState.multiviewPrerendering}
                <div class="progress-container">
                    <div class="progress-bar prerender" style:width="{viewerState.multiviewPrerenderProgress * 100}%"></div>
                </div>
            {/if}
        {/if}

        <div class="slider-group">
            <label>
                Frames: {viewerState.turntableFrameCount}
                <input type="range" min="12" max="360" step="12" bind:value={viewerState.turntableFrameCount}
                    disabled={viewerState.isTurntableRendering} />
            </label>
        </div>

        <div class="slider-group">
            <label>
                Steps/frame: {viewerState.turntableStepsPerFrame}
                <input type="range" min="1" max="200" step="1" bind:value={viewerState.turntableStepsPerFrame}
                    disabled={viewerState.isTurntableRendering} />
            </label>
        </div>

        <div class="param-header">Path Variation</div>

        <div class="slider-group">
            <label>
                Lat oscillation: {latDeg}°
                <input type="range" min="0" max={Math.PI / 3} step="0.01" bind:value={viewerState.turntableLatAmplitude}
                    disabled={viewerState.isTurntableRendering} />
            </label>
        </div>
        {#if viewerState.turntableLatAmplitude > 0}
            <div class="slider-group sub-param">
                <label>
                    Lat cycles: {viewerState.turntableLatCycles}
                    <input type="range" min="1" max="8" step="1" bind:value={viewerState.turntableLatCycles}
                        disabled={viewerState.isTurntableRendering} />
                </label>
            </div>
        {/if}

        <div class="slider-group">
            <label>
                Radius oscillation: {radPct}%
                <input type="range" min="0" max="0.5" step="0.01" bind:value={viewerState.turntableRadiusAmplitude}
                    disabled={viewerState.isTurntableRendering} />
            </label>
        </div>
        {#if viewerState.turntableRadiusAmplitude > 0}
            <div class="slider-group sub-param">
                <label>
                    Radius cycles: {viewerState.turntableRadiusCycles}
                    <input type="range" min="1" max="8" step="1" bind:value={viewerState.turntableRadiusCycles}
                        disabled={viewerState.isTurntableRendering} />
                </label>
            </div>
        {/if}

        {#if viewerState.isTurntableRendering}
            <div class="progress-container">
                <div class="progress-bar" style:width="{viewerState.turntableProgress * 100}%"></div>
            </div>
            <div class="progress-label">
                {Math.round(viewerState.turntableProgress * 100)}%
                — Frame {Math.round(viewerState.turntableProgress * viewerState.turntableFrameCount)}/{viewerState.turntableFrameCount}
            </div>
            <button
                class="render-btn cancel"
                onclick={() => viewerState.cancelTurntable()}
            >
                ✕ Cancel
            </button>
        {:else}
            <button
                class="render-btn turntable"
                onclick={() => viewerState.renderTurntable()}
                disabled={viewerState.isCapturing}
            >
                🎬 Save Frames to Folder
            </button>
        {/if}
    </div>
</div>

<style lang="scss">
    .overlays {
        position: absolute;
        top: 1rem;
        left: 1rem;
        z-index: 100;
        background: rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(8px);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        color: white;
        font-family: sans-serif;
        font-size: 0.9rem;
        border: 1px solid rgba(255, 255, 255, 0.1);

        label {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            cursor: pointer;
            user-select: none;

            &.sub {
                margin-left: 1.25rem;
                font-size: 0.8rem;
                color: rgba(255, 255, 255, 0.65);
            }
        }

        input {
            cursor: pointer;
        }

        .separator {
            height: 1px;
            background: rgba(255, 255, 255, 0.15);
            margin: 0.5rem 0;
        }

        .render-btn {
            margin-top: 0.25rem;
            width: 100%;
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            color: white;
            border: none;
            padding: 0.5rem 0.75rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.4rem;
            box-shadow: 0 3px 10px rgba(99, 102, 241, 0.3);

            &:hover:not(:disabled) {
                transform: translateY(-1px);
                box-shadow: 0 5px 14px rgba(99, 102, 241, 0.4);
                filter: brightness(1.1);
            }

            &:active:not(:disabled) {
                transform: translateY(0);
            }

            &:disabled {
                background: #444;
                box-shadow: none;
                cursor: not-allowed;
                opacity: 0.7;
            }

            &.turntable {
                background: linear-gradient(135deg, #059669 0%, #10b981 100%);
                box-shadow: 0 3px 10px rgba(16, 185, 129, 0.3);

                &:hover:not(:disabled) {
                    box-shadow: 0 5px 14px rgba(16, 185, 129, 0.4);
                }
            }

            &.cancel {
                background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
                box-shadow: 0 3px 10px rgba(239, 68, 68, 0.3);

                &:hover:not(:disabled) {
                    box-shadow: 0 5px 14px rgba(239, 68, 68, 0.4);
                }
            }

            &.training-active {
                background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%);
                box-shadow: 0 3px 10px rgba(245, 158, 11, 0.3);
                animation: glow-pulse 2s ease-in-out infinite;

                &:hover:not(:disabled) {
                    box-shadow: 0 5px 14px rgba(245, 158, 11, 0.4);
                }
            }
        }

        .spinner {
            width: 14px;
            height: 14px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: white;
            animation: pulse 1.5s ease-in-out infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(0.8); }
        }

        @keyframes glow-pulse {
            0%, 100% { box-shadow: 0 3px 10px rgba(245, 158, 11, 0.3); }
            50% { box-shadow: 0 3px 16px rgba(245, 158, 11, 0.5); }
        }

        .turntable-section {
            .section-title {
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                color: rgba(255, 255, 255, 0.5);
                margin-bottom: 0.35rem;
            }

            .param-header {
                font-size: 0.7rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: rgba(255, 255, 255, 0.35);
                margin-top: 0.4rem;
                margin-bottom: 0.15rem;
            }
        }

        .slider-group {
            margin-bottom: 0.25rem;

            label {
                font-size: 0.8rem;
                color: rgba(255, 255, 255, 0.7);
                display: flex;
                flex-direction: column;
                gap: 0.15rem;
            }

            input[type="range"] {
                width: 100%;
                accent-color: #a855f7;
            }

            &.sub-param {
                margin-left: 0.75rem;
                label {
                    color: rgba(255, 255, 255, 0.5);
                    font-size: 0.75rem;
                }
            }
        }

        .progress-container {
            width: 100%;
            height: 4px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            overflow: hidden;
            margin: 0.4rem 0;
        }

        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #10b981, #34d399);
            border-radius: 2px;
            transition: width 0.3s ease;

            &.prerender {
                background: linear-gradient(90deg, #6366f1, #a855f7);
            }
        }

        .progress-label {
            font-size: 0.75rem;
            color: rgba(255, 255, 255, 0.6);
            text-align: center;
            margin-bottom: 0.25rem;
        }
    }
</style>