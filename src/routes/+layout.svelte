<script lang="ts">
import "./index.scss";
import favicon from '$lib/assets/favicon.svg';
import { page } from '$app/state';
import Toasts from "$/viewer/Toasts.svelte";
import WorkspaceNav from "./WorkspaceNav.svelte";

let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

<main>
	<WorkspaceNav />

	<app-content>
		{@render children()}
	</app-content>

	<Toasts />
</main>

<style lang="scss">
main {
	width: 100vw;
	height: 100vh;

	display: flex;
	flex-direction: column;
	align-items: stretch;

	overflow: hidden;
}

app-content {
	flex: 1 1 auto;
	min-height: 0;

	display: flex;
	align-items: stretch;

	overflow: hidden;
}
</style>
