import { page } from 'vitest/browser';
import { describe, expect, it } from 'vitest';
import { render } from 'vitest-browser-svelte';
import Page from './+page.svelte';

describe('/+page.svelte', () => {
	it('renders the brushstroke optimizer without legacy mode buttons', async () => {
		render(Page);
		
		await expect.element(page.getByText('Load a model (.glb)')).toBeInTheDocument();
		await expect.element(page.getByRole('button', { name: 'Materials' })).not.toBeInTheDocument();
		await expect.element(page.getByRole('button', { name: 'Contour Modeler' })).not.toBeInTheDocument();
		await expect.element(page.getByRole('button', { name: 'Paint Modeler' })).not.toBeInTheDocument();
	});
});
