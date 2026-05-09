import { describe, expect, it } from 'vitest';
import { constants } from './constants';
import { computeOptimTextureSize } from './optimTextureSize';

describe('computeOptimTextureSize', () => {
	const short = constants.OPTIM_SHORT;

	it('uses short side as height when aspect >= 1 (landscape / square)', () => {
		expect(computeOptimTextureSize(short, 1)).toEqual({ width: short, height: short });
		expect(computeOptimTextureSize(short, 2)).toEqual({ width: 256, height: short });
	});

	it('uses short side as width when aspect < 1 (portrait)', () => {
		expect(computeOptimTextureSize(short, 0.5)).toEqual({ width: short, height: 256 });
	});

	it('rounds the long side to integer pixels', () => {
		const { width, height } = computeOptimTextureSize(100, 1.333);
		expect(width).toBe(133);
		expect(height).toBe(100);
	});
});
