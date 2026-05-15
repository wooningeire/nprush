<script lang="ts">
import type { Snippet } from "svelte";
let {
    children,
}: {
    children?: Snippet,
} = $props();
</script>

<view-panel-control-box onpointerdown={event => event.stopPropagation()}>
    {@render children?.()}
</view-panel-control-box>

<style lang="scss">
view-panel-control-box {
    align-self: start;
    justify-self: start;

    padding: 0.5rem;
    display: flex;
    flex-direction: column;

    gap: 0.25rem;

    background: oklch(0.2 0.05 190 / 0.8);
    backdrop-filter: blur(8px);

    border: 1px oklch(1 0 0 / 0.25);
    border-style: none solid solid none;

    border-bottom-right-radius: 0.5rem;
}
</style>