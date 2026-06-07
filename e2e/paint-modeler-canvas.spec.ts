import { expect, test, type Page } from "playwright/test";

test("paint modeler canvas supports paint, fast depth hover, and depth pull", async ({ page }) => {
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
    await page.getByRole("button", { name: "Add Object" }).click();
    await page.getByRole("button", { name: "New" }).click();
    await page.getByRole("button", { name: "View Plane" }).click();
    await page.getByRole("button", { name: "Surface" }).click();

    await page.locator('input[type="range"]').nth(0).evaluate(element => {
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
    await expect(page.getByText("View Plane")).toBeVisible();
    await expect(page.getByRole("button", { name: "Surface" })).toHaveClass(/active/);

    await page.getByRole("button", { name: "Paint", exact: true }).click();
    await page.mouse.move(cx - 130, cy + 74);
    await page.mouse.down();
    await page.mouse.move(cx + 42, cy + 84, { steps: 6 });
    await expect(page.locator(".draft")).toHaveCount(0);
    await page.mouse.up();

    await page.getByLabel("Brush lattice").uncheck();
    await page.getByRole("button", { name: "Depth Brush" }).click();
    await expect(page.getByRole("button", { name: "Raise" })).toHaveClass(/active/);

    const hoverStart = Date.now();
    for (let i = 0; i < 32; i++) {
        await page.mouse.move(cx - 120 + i * 7, cy + Math.sin(i / 3) * 18);
    }
    const hoverMs = Date.now() - hoverStart;
    expect(hoverMs).toBeLessThan(5_000);
    await expect(page.locator(".depth-brush-cursor.brush")).toBeVisible();

    await page.getByLabel("Brush lattice").check();
    await page.mouse.move(cx + 12, cy + 4, { steps: 4 });
    await expect(page.locator(".depth-brush-cursor.brush")).toBeVisible();
    const latticeLines = page.locator(".brush-lattice line");
    await expect(latticeLines).not.toHaveCount(0);
    const firstLatticeLength = await latticeLines.first().evaluate(element => {
        const line = element as SVGLineElement;
        return Math.hypot(
            line.x1.baseVal.value - line.x2.baseVal.value,
            line.y1.baseVal.value - line.y2.baseVal.value,
        );
    });
    expect(firstLatticeLength).toBeGreaterThan(1);

    await page.getByRole("button", { name: "Depth Pull" }).click();
    await page.mouse.move(cx + 8, cy + 2);
    await page.mouse.down();
    await page.mouse.move(cx + 8, cy + 84, { steps: 6 });
    await expect(page.locator(".depth-brush-cursor.pull")).toBeVisible();
    await expect(page.locator(".depth-pull-anchor")).toBeVisible();
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
