import { describe, expect, it } from "vitest";
import { PaintModelingState } from "./PaintModelingState.svelte.ts";

describe("PaintModelingState ribbon surface boundary", () => {
    it("does not keep CPU-only surface editing entry points", () => {
        const state = new PaintModelingState() as unknown as Record<string, unknown>;

        expect(state.raycastStrokeAt).toBeUndefined();
        expect(state.sculptStrokeAt).toBeUndefined();
        expect(state.addDeformationLine).toBeUndefined();
    });
});
