<script lang="ts">
    import { fade } from "svelte/transition";
import { toasts, dismissToast } from "./toast.svelte.ts";
</script>

<div class="toast-container">
    {#each toasts as toast (toast.id)}
        <div
            class="toast {toast.kind}"
            role="alert"
            out:fade={{duration: 200}}
        >
            <span class="message">{toast.message}</span>
            <button class="close" onclick={() => dismissToast(toast.id)} aria-label="Dismiss">✕</button>
        </div>
    {/each}
</div>

<style lang="scss">
.toast-container {
    position: fixed;
    bottom: 1.5rem;
    left: 50%;
    transform: translateX(-50%);
    z-index: 1000;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    align-items: center;
    pointer-events: none;
}

.toast {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 1rem;
    border-radius: 8px;
    font-family: sans-serif;
    font-size: 0.875rem;
    color: white;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    pointer-events: all;
    max-width: 480px;
    animation: slide-in 0.2s ease;

    &.info    { background: rgba(30, 30, 50, 0.85); border: 1px solid rgba(255,255,255,0.15); }
    &.success { background: rgba(5, 100, 60, 0.85);  border: 1px solid rgba(16,185,129,0.4); }
    &.error   { background: rgba(120, 20, 20, 0.85); border: 1px solid rgba(239,68,68,0.4); }

    .message { flex: 1; }

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

@keyframes slide-in {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
