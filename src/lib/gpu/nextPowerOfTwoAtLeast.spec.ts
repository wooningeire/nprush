import { describe, expect, it } from 'vitest';
import { nextPowerOfTwoAtLeast } from './nextPowerOfTwoAtLeast';

describe('nextPowerOfTwoAtLeast', () => {
	it('returns 1 for non-positive n (matches legacy loop with count 0)', () => {
		expect(nextPowerOfTwoAtLeast(0)).toBe(1);
		expect(nextPowerOfTwoAtLeast(-5)).toBe(1);
	});

	it('returns n when n is already a power of two', () => {
		expect(nextPowerOfTwoAtLeast(1)).toBe(1);
		expect(nextPowerOfTwoAtLeast(32)).toBe(32);
		expect(nextPowerOfTwoAtLeast(1024)).toBe(1024);
	});

	it('returns the next power of two when n is not a power of two', () => {
		expect(nextPowerOfTwoAtLeast(3)).toBe(4);
		expect(nextPowerOfTwoAtLeast(17)).toBe(32);
		expect(nextPowerOfTwoAtLeast(513)).toBe(1024);
	});
});
