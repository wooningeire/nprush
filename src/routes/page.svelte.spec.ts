import { page } from 'vitest/browser';
import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';
import Page from './+page.svelte';

describe('/+page.svelte', () => {
	it('should render the top-level work modes', async () => {
		render(Page);
		
		await expect.element(page.getByRole('button', { name: 'Brushstroke optimizer' })).toBeInTheDocument();
		await expect.element(page.getByRole('button', { name: 'Materials' })).toBeInTheDocument();
		await expect.element(page.getByRole('button', { name: 'Contour Modeler' })).toBeInTheDocument();
		await expect.element(page.getByRole('button', { name: 'Paint Modeler' })).toBeInTheDocument();
	});
});
