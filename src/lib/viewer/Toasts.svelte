<script lang="ts">
    import { fade, fly } from "svelte/transition";
import { toasts, dismissToast } from "./toast.svelte.ts";
    import { flip } from "svelte/animate";
</script>

<div class="toast-container">
    {#each toasts as toast (toast.id)}
        <div
            class="toast {toast.kind}"
            role="alert"
            in:fly={{duration: 200}}
            out:fade={{duration: 200}}
            animate:flip={{duration: 200}}
        >
            {#if toast.kind === "error"}
                <div>🛑</div>
            {/if}

            <div class="message">
                <div class="message-text">
                    {toast.message}
                </div>

                {#if toast.kind === "error"}
                    <div class="error-description">
                        this is an error; you will need to reload the page
                    </div>
                {/if}
            </div>

            <button
                class="close"
                onclick={() => dismissToast(toast.id)}
                aria-label="Dismiss"
            >✕</button>
        </div>
    {/each}
</div>

<style lang="scss">
.toast-container {
    position: fixed;

    bottom: 1rem;
    right: 1rem;
    
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    align-items: flex-end;

    pointer-events: none;
}

.toast {
    pointer-events: all;

    display: flex;
    align-items: center;

    text-align: right;

    gap: 0.75rem;
    padding: 0.6rem 1rem;
    border-radius: 8px;
    font-size: 0.875rem;
    color: white;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    max-width: 480px;

    &.info    { background: rgba(30, 30, 50, 0.85); border: 1px solid rgba(255,255,255,0.15); }
    &.success { background: rgba(5, 100, 60, 0.85);  border: 1px solid rgba(16,185,129,0.4); }
    &.error   { background: rgba(120, 20, 20, 0.85); border: 1px solid rgba(239,68,68,0.4); }


    .close {
        background: none;
        border: none;
        color: rgba(255,255,255,0.6);
        cursor: pointer;
        font-size: 0.75rem;
        padding: 0;
        line-height: 1;

        &:hover { color: white; }
    }
}

.error-description {
    opacity: 0.6;
    
    font-size: 0.8rem;
    font-style: italic;
}
</style>
