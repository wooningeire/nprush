/**
 * Smallest power of two that is >= n. Used for GPU sort buffer sizing (bitonic sort).
 * For n <= 0, returns 1 to match legacy `let n = 1; while (n < count) n <<= 1` when count is 0.
 */
export function nextPowerOfTwoAtLeast(n: number): number {
	if (n <= 0) return 1;
	let p = 1;
	while (p < n) p <<= 1;
	return p;
}
