import { describe, expect, it, vi } from 'vitest';
import { constants, injectWgslConstants } from './constants';

describe('constants', () => {
	it('PIXEL_LOSS_MAX covers at least the default square optimization resolution', () => {
		expect(constants.PIXEL_LOSS_MAX).toBeGreaterThanOrEqual(
			constants.OPTIMIZATION_SHORT * constants.OPTIMIZATION_SHORT,
		);
	});
});

describe('injectWgslConstants', () => {
	it('replaces {@KEY} placeholders with substitution values', () => {
		const out = injectWgslConstants('x={@A} y={@B}', { A: 1, B: 'two' });
		expect(out).toBe('x=1 y=two');
	});

	it('coerces numbers and booleans with String()', () => {
		expect(injectWgslConstants('{@N} {@F} {@T} {@F2}', { N: 0, F: 1.5, T: true, F2: false })).toBe(
			'0 1.5 true false',
		);
	});

	it('replaces multiple occurrences of the same key', () => {
		expect(injectWgslConstants('{@K}-{@K}', { K: 9 })).toBe('9-9');
	});

	it('ignores non-placeholder braces', () => {
		const src = 'const x = { a: 1 }; and {@OK}';
		expect(injectWgslConstants(src, { OK: 2 })).toBe('const x = { a: 1 }; and 2');
	});

	it('only matches WGSL-style tokens [A-Z0-9_]+ after {@', () => {
		// Lowercase letters are not part of the token, so the pattern does not match.
		expect(injectWgslConstants('{@low}', { low: 99 })).toBe('{@low}');
		// Numeric keys still work when present; when missing the placeholder is preserved.
		const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
		expect(injectWgslConstants('{@9}', {})).toBe('{@9}');
		expect(warn).toHaveBeenCalled();
		warn.mockRestore();
	});

	it('leaves unknown keys unchanged and warns', () => {
		const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
		const src = 'a={@KNOWN} b={@MISSING}';
		const out = injectWgslConstants(src, { KNOWN: 1 });
		expect(out).toBe('a=1 b={@MISSING}');
		expect(warn).toHaveBeenCalledOnce();
		expect(warn.mock.calls[0][0]).toContain('MISSING');
		warn.mockRestore();
	});

	it('uses own properties only (Object.hasOwn)', () => {
		const proto = { FROM_PROTO: 1 };
		const subs = Object.create(proto) as Record<string, number>;
		subs.DIRECT = 2;
		expect(injectWgslConstants('{@FROM_PROTO}{@DIRECT}', subs)).toBe('{@FROM_PROTO}2');
	});
});
