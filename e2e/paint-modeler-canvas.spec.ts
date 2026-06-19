import { expect, test, type Page } from "playwright/test";

test("paint modeler canvas supports the simplified paint tool surface", async ({ page }) => {
    const consoleProblems: string[] = [];
    page.on("console", message => {
        if (message.type() === "error" && !isExpectedStartupNoise(message.text())) {
            consoleProblems.push(message.text());
        }
    });
    page.on("pageerror", error => {
        if (!isExpectedStartupNoise(error.message)) consoleProblems.push(error.message);
    });

    await forbidWebglContexts(page);

    await page.goto("/paint-modeler");
    await dismissStartupAlerts(page, 700);
    await expect(page.locator("paint-viewport")).toBeVisible();
    await page.getByRole("button", { name: "Add Object" }).click();
    await expect(page.getByRole("button", { name: "Depth Brush" })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Depth Pull" })).toHaveCount(0);
    await expect(page.getByText("Placement", { exact: true })).toHaveCount(0);
    await expect(page.getByText("Projection", { exact: true })).toHaveCount(0);
    await expect(page.getByText("Render", { exact: true })).toHaveCount(0);
    await expect(page.getByText("Tool", { exact: true })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Seam" })).toHaveCount(0);
    await expect(page.getByLabel("Opacity")).toHaveCount(0);
    await expect(page.getByLabel("Depth rate")).toHaveCount(0);
    await expect(page.getByLabel("Brush lattice")).toHaveCount(0);
    await expect(page.getByLabel("Seam size")).toHaveCount(0);
    await expect(page.getByLabel("Surface field")).not.toBeChecked();

    await page.getByLabel("Width").evaluate(element => {
        const input = element as HTMLInputElement;
        input.value = "66";
        input.dispatchEvent(new Event("input", { bubbles: true }));
    });

    const viewport = page.locator("paint-viewport");
    await expect(viewport).toBeVisible();
    const box = await viewport.boundingBox();
    if (!box) throw new Error("paint viewport missing");
    const cx = box.x + box.width * 0.56;
    const cy = box.y + box.height * 0.52;

    await page.mouse.move(cx - 150, cy - 20);
    await page.mouse.down();
    for (const [dx, dy] of [[-95, 10], [-35, -12], [40, 8], [105, -10], [155, 12]]) {
        await page.mouse.move(cx + dx, cy + dy, { steps: 5 });
    }
    await page.mouse.up();

    await expect(page.getByText("Charts 1")).toBeVisible();

    const withoutSurfaceField = await page.locator("canvas").screenshot();
    await page.getByLabel("Surface field").check();
    await expect(page.getByLabel("Surface field")).toBeChecked();
    await waitForAnimationFrame(page);
    const withSurfaceField = await page.locator("canvas").screenshot();
    expect(withSurfaceField.equals(withoutSurfaceField)).toBe(false);
    await page.getByLabel("Surface field").uncheck();

    await page.mouse.move(cx - 130, cy + 74);
    await page.mouse.down();
    await page.mouse.move(cx + 42, cy + 84, { steps: 6 });
    await expect(page.locator(".draft")).toHaveCount(0);
    await page.mouse.up();

    expect(consoleProblems).toEqual([]);
});

function isExpectedStartupNoise(text: string): boolean {
    return text.includes("ResizeObserver loop completed")
        || text.includes("could not get gpu adapter");
}

async function dismissStartupAlerts(page: Page, durationMs = 1_200) {
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
}

async function waitForAnimationFrame(page: Page) {
    await page.evaluate(() => new Promise<void>(resolve => {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
    }));
}

async function forbidWebglContexts(page: Page) {
    await page.addInitScript(() => {
        const originalGetContext = HTMLCanvasElement.prototype.getContext;
        HTMLCanvasElement.prototype.getContext = function(...args) {
            const [contextId] = args;
            if (
                contextId === "webgl"
                || contextId === "webgl2"
                || contextId === "experimental-webgl"
            ) {
                throw new Error(`Paint Modeler requested ${contextId}; WebGPU only`);
            }
            return originalGetContext.apply(this, args);
        } as typeof HTMLCanvasElement.prototype.getContext;
    });
}
