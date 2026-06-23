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
    const paintLayers = page.locator("section").filter({ hasText: "Paint Layers" });
    await expect(paintLayers.getByRole("button", { name: /Layer 1\s+0s/ })).toHaveClass(/active/);
    await paintLayers.getByRole("button", { name: "Add" }).click();
    await expect(paintLayers.getByRole("button", { name: /Layer 2\s+0s/ })).toHaveClass(/active/);
    await paintLayers.getByRole("button", { name: /Layer 1\s+0s/ }).click();
    await expect(paintLayers.getByRole("button", { name: /Layer 1\s+0s/ })).toHaveClass(/active/);

    const layerBox = await paintLayers.boundingBox();
    const controlPanelBox = await page.locator(".control-panel").boundingBox();
    if (!layerBox || !controlPanelBox) throw new Error("paint layer controls missing");
    expect(layerBox.x).toBeGreaterThanOrEqual(controlPanelBox.x - 1);
    expect(layerBox.x + layerBox.width).toBeLessThanOrEqual(controlPanelBox.x + controlPanelBox.width + 1);

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
    await expect(page.getByRole("group", { name: "Brush mode" })).toBeVisible();
    await expect(page.getByRole("group", { name: "Stroke geometry" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Color" })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByRole("button", { name: "Billboard" })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByRole("button", { name: "Ribbon" })).toBeEnabled();
    await expect(page.getByLabel("Width")).toHaveValue("18");
    await page.getByRole("button", { name: "Surface" }).click();
    await expect(page.getByRole("button", { name: "Surface" })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByLabel("Width")).toHaveValue("72");
    await expect(page.getByLabel("Color")).toBeDisabled();
    await expect(page.getByRole("button", { name: "Ribbon" })).toBeDisabled();

    await page.getByLabel("Width").evaluate(element => {
        const input = element as HTMLInputElement;
        input.value = "66";
        input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await expect(page.getByLabel("Width")).toHaveValue("66");

    await page.getByRole("button", { name: "Depth" }).click();
    await expect(page.getByRole("button", { name: "Depth" })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByLabel("Width")).toHaveValue("36");
    await expect(page.getByLabel("Color")).toBeDisabled();
    await page.getByRole("button", { name: "Surface" }).click();
    await expect(page.getByRole("button", { name: "Surface" })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByLabel("Width")).toHaveValue("66");

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
    await expect(page.getByText("1c 0s")).toBeVisible();

    await page.mouse.move(cx, cy);
    await page.mouse.down({ button: "middle" });
    await page.mouse.move(cx + 90, cy + 24, { steps: 6 });
    await page.mouse.up({ button: "middle" });
    await waitForAnimationFrame(page);

    const rendererUnavailable = await page.getByText("WebGPU unavailable").isVisible().catch(() => false);
    if (rendererUnavailable) {
        await expect(page.getByText("WebGPU unavailable")).toBeVisible();
        await page.getByLabel("Surface field").check();
        await expect(page.getByLabel("Surface field")).toBeChecked();
    } else {
        const withChartWire = await page.locator("canvas").screenshot();
        await page.getByLabel("Chart wire").uncheck();
        await expect(page.getByLabel("Chart wire")).not.toBeChecked();
        await waitForAnimationFrame(page);
        const withoutChartWire = await page.locator("canvas").screenshot();
        expect(withoutChartWire.equals(withChartWire)).toBe(false);
        await page.getByLabel("Chart wire").check();
        await expect(page.getByLabel("Chart wire")).toBeChecked();
        await waitForAnimationFrame(page);

        const withoutSurfaceField = await page.locator("canvas").screenshot();
        await page.getByLabel("Surface field").check();
        await expect(page.getByLabel("Surface field")).toBeChecked();
        await waitForAnimationFrame(page);
        const withSurfaceField = await page.locator("canvas").screenshot();
        expect(withSurfaceField.equals(withoutSurfaceField)).toBe(false);
    }
    await page.getByLabel("Surface field").uncheck();
    await page.getByRole("button", { name: "Color" }).click();
    await expect(page.getByRole("button", { name: "Color" })).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByLabel("Width")).toHaveValue("18");
    await expect(page.getByLabel("Color")).toBeEnabled();
    await expect(page.getByRole("button", { name: "Ribbon" })).toBeEnabled();
    await page.getByRole("button", { name: "Ribbon" }).click();
    await expect(page.getByRole("button", { name: "Ribbon" })).toHaveAttribute("aria-pressed", "true");

    await page.mouse.move(cx - 130, cy + 74);
    await page.mouse.down();
    await page.mouse.move(cx + 42, cy + 84, { steps: 6 });
    await expect(page.locator(".draft")).toHaveCount(0);
    await page.mouse.up();
    await expect(page.getByText(/\d+c 1s/)).toBeVisible();
    await expect(paintLayers.getByRole("button", { name: /Layer 1\s+1s/ })).toBeVisible();

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
