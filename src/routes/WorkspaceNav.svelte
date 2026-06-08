<script lang="ts">
import { page } from "$app/state";
import { resolve } from "$app/paths"; 
import type { RouteIdWithSearchOrHash } from "$app/types";

const toolRoutes: {
    href: RouteIdWithSearchOrHash,
    label: string,
}[] = [
	{ href: "/brushstroke-optimizer", label: "Brushstroke optimizer" },
	{ href: "/paint-modeler", label: "Paint Modeler" },
];
</script>


<tool-nav aria-label="Workspace tools">
    <nprush-logomark>nprush!</nprush-logomark>

    {#each toolRoutes as route}
        <a
            href={resolve(route.href)}
            aria-current={page.route.id === route.href ? "page" : undefined}
        >
            {route.label}
        </a>
    {/each}
</tool-nav>

<style lang="scss">
tool-nav {
	flex: 0 0 auto;

	display: flex;
	align-items: center;
	gap: 1.1rem;

	min-height: 2.65rem;
	padding: 0 1rem;
	border-bottom: 1px solid rgba(190, 231, 224, 0.12);
	background: rgba(5, 12, 14, 0.64);
	backdrop-filter: blur(10px);

	a {
		display: inline-flex;
		align-items: center;
		min-height: 2.65rem;
		border-bottom: 2px solid transparent;
		color: rgba(223, 244, 239, 0.68);
		text-decoration: none;
		white-space: nowrap;

		&:hover {
			color: rgba(247, 255, 253, 0.95);
		}

		&[aria-current="page"] {
			border-bottom-color: #7dddbd;
			color: white;
		}
	}
}


nprush-logomark {
	font-family: "Belanosima", "Averia Libre", sans-serif;
	font-weight: 600;
}
</style>