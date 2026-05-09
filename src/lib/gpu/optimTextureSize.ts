/**
 * Pixel dimensions for the optimization render targets: short side is fixed,
 * long side follows the panel aspect ratio so pixel proportions match the visible panel.
 */
export function computeOptimTextureSize(
	shortSide: number,
	panelAspect: number,
): { width: number; height: number } {
	if (panelAspect >= 1) {
		return {
			width: Math.round(shortSide * panelAspect),
			height: shortSide,
		};
	}
	return {
		width: shortSide,
		height: Math.round(shortSide / panelAspect),
	};
}
