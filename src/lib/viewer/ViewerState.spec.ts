import { describe, it, expect, vi } from "vitest";
import { ViewerState } from "./ViewerState.svelte.ts";

describe("ViewerState Camera Separation", () => {
    it("should use browser window size for viewportCamera and render dims for backendCamera", () => {
        const state = new ViewerState();
        state.width = 800;
        state.height = 600;
        state.renderWidth = 512;
        state.renderHeight = 512;

        expect(state.viewportCamera.screenDims.width()).toBe(800);
        expect(state.viewportCamera.screenDims.height()).toBe(600);
        expect(state.backendCamera.screenDims.width()).toBe(512);
        expect(state.backendCamera.screenDims.height()).toBe(512);
        expect(state.viewportCamera.aspect).toBe(1);
        expect(state.backendCamera.aspect).toBe(1);

        // Modifying window size should NOT affect backend camera
        state.width = 1920;
        state.height = 1080;
        expect(state.viewportCamera.screenDims.width()).toBe(1920);
        expect(state.viewportCamera.screenDims.height()).toBe(1080);
        expect(state.backendCamera.screenDims.width()).toBe(512);
        expect(state.backendCamera.screenDims.height()).toBe(512);
        expect(state.viewportCamera.aspect).toBe(1);
        expect(state.backendCamera.aspect).toBe(1);
    });

    it("should keep viewportOrbit and backendOrbit independent during normal usage", () => {
        const state = new ViewerState();
        state.viewportOrbit.long = 1.0;
        state.viewportOrbit.lat = 0.5;
        state.viewportOrbit.radius = 2.0;

        expect(state.backendOrbit.long).not.toBe(1.0);
        expect(state.backendOrbit.lat).not.toBe(0.5);
        expect(state.backendOrbit.radius).not.toBe(2.0);
    });

    it("should initialize backendOrbit to viewportOrbit when toggling turntable training", () => {
        const state = new ViewerState();
        
        // Setup initial viewport orientation
        state.viewportOrbit.long = Math.PI;
        state.viewportOrbit.lat = Math.PI / 4;
        state.viewportOrbit.radius = 3.0;

        // Ensure backend starts differently
        state.backendOrbit.long = 0;
        state.backendOrbit.lat = 0;
        state.backendOrbit.radius = 1;

        // Mock the runner so it doesn't throw or do actual work
        state.runner = {
            prerenderDataset: vi.fn().mockResolvedValue(undefined)
        } as any;

        state.toggleTurntableTraining();

        // backendOrbit should now match viewportOrbit (initial turntable orientation)
        expect(state.backendOrbit.long).toBe(Math.PI);
        expect(state.backendOrbit.lat).toBe(Math.PI / 4);
        expect(state.backendOrbit.radius).toBe(3.0);

        // However, subsequent changes to viewportOrbit should NOT affect backendOrbit
        state.viewportOrbit.long = 0;
        expect(state.backendOrbit.long).toBe(Math.PI);
    });
});
