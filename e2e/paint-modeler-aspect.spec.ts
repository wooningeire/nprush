import { expect, test, type Page } from "playwright/test";

type ViewportGeometry = {
    viewportWidth: number,
    viewportHeight: number,
    canvasWidth: number,
    canvasHeight: number,
    documentOverflowX: number,
    documentOverflowY: number,
};

type ProjectionAspects = {
    wide: number,
    tall: number,
};

test("paint modeler projection follows viewport aspect without layout overflow", async ({ page }) => {
    const consoleProblems: string[] = [];
    page.on("console", message => {
        if (message.type() === "error" && !isExpectedStartupNoise(message.text())) {
            consoleProblems.push(message.text());
        }
    });
    page.on("pageerror", error => {
        if (!isExpectedStartupNoise(error.message)) consoleProblems.push(error.message);
    });

    await page.goto("/paint-modeler");
    await dismissStartupAlerts(page, 700);
    await expect(page.locator("paint-viewport")).toBeVisible();

    const geometry = await page.evaluate((): ViewportGeometry => {
        const viewport = document.querySelector("paint-viewport")?.getBoundingClientRect();
        const canvas = document.querySelector("canvas")?.getBoundingClientRect();
        if (!viewport || !canvas) throw new Error("Paint modeler viewport or canvas missing");

        return {
            viewportWidth: viewport.width,
            viewportHeight: viewport.height,
            canvasWidth: canvas.width,
            canvasHeight: canvas.height,
            documentOverflowX: document.documentElement.scrollWidth - document.documentElement.clientWidth,
            documentOverflowY: document.documentElement.scrollHeight - document.documentElement.clientHeight,
        };
    });

    expect(geometry.viewportWidth).toBeGreaterThan(320);
    expect(geometry.viewportHeight).toBeGreaterThan(320);
    expect(Math.abs(geometry.canvasWidth - geometry.viewportWidth)).toBeLessThan(1);
    expect(Math.abs(geometry.canvasHeight - geometry.viewportHeight)).toBeLessThan(1);
    expect(geometry.documentOverflowX).toBeLessThanOrEqual(1);
    expect(geometry.documentOverflowY).toBeLessThanOrEqual(1);

    const projectionAspects = await browserProjectionAspects(page);
    expect(projectionAspects.wide).toBeCloseTo(2, 5);
    expect(projectionAspects.tall).toBeCloseTo(0.5, 5);
    expect(consoleProblems).toEqual([]);
});

const browserProjectionAspects = async (page: Page): Promise<ProjectionAspects> => {
    return page.evaluate(async () => {
        const { PaintModelingState } = await import("/src/lib/paint-modeling/PaintModelingState.svelte.ts");
        const state = new PaintModelingState();
        const wideView = state.saveCurrentView(1200, 600, false);
        const tallView = state.saveCurrentView(600, 1200, false);

        const distance3 = (a: [number, number, number], b: [number, number, number]): number => {
            return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
        };

        const unprojectNdc = (
            viewProjInvMat: number[],
            x: number,
            y: number,
            z: number,
        ): [number, number, number] => {
            const worldX = viewProjInvMat[0] * x + viewProjInvMat[4] * y + viewProjInvMat[8] * z + viewProjInvMat[12];
            const worldY = viewProjInvMat[1] * x + viewProjInvMat[5] * y + viewProjInvMat[9] * z + viewProjInvMat[13];
            const worldZ = viewProjInvMat[2] * x + viewProjInvMat[6] * y + viewProjInvMat[10] * z + viewProjInvMat[14];
            const worldW = viewProjInvMat[3] * x + viewProjInvMat[7] * y + viewProjInvMat[11] * z + viewProjInvMat[15];
            if (!Number.isFinite(worldW) || Math.abs(worldW) <= 1e-6) {
                throw new Error("Cannot unproject NDC point");
            }
            return [worldX / worldW, worldY / worldW, worldZ / worldW];
        };

        const projectedSpanAspect = (viewProjInvMat: number[]): number => {
            const left = unprojectNdc(viewProjInvMat, -1, 0, 0.5);
            const right = unprojectNdc(viewProjInvMat, 1, 0, 0.5);
            const bottom = unprojectNdc(viewProjInvMat, 0, -1, 0.5);
            const top = unprojectNdc(viewProjInvMat, 0, 1, 0.5);

            return distance3(left, right) / distance3(bottom, top);
        };

        return {
            wide: projectedSpanAspect(wideView.viewProjInvMat),
            tall: projectedSpanAspect(tallView.viewProjInvMat),
        };
    });
};

const isExpectedStartupNoise = (text: string): boolean => {
    return text.includes("ResizeObserver loop completed")
        || text.includes("could not get gpu adapter");
};

const dismissStartupAlerts = async (page: Page, durationMs = 1_200): Promise<void> => {
    const deadline = Date.now() + durationMs;
    while (Date.now() < deadline) {
        const dismiss = page.getByLabel("Dismiss");
        if (await dismiss.count() > 0) {
            await dismiss.first().click({ timeout: 750 }).catch(() => {});
            await page.waitForTimeout(100);
        } else {
            await page.waitForTimeout(100);
        }
    }
};
