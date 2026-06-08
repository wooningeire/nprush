import { expect, test, type Page } from "playwright/test";

test("paint modeler snap preview stays responsive after the first stroke", async ({ page }) => {
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
    await page.getByRole("button", { name: "Add Object" }).click();
    await page.getByRole("button", { name: "New" }).click();
    await page.getByRole("button", { name: "View Plane" }).click();

    const viewport = page.locator("paint-viewport");
    await expect(viewport).toBeVisible();
    const box = await viewport.boundingBox();
    if (!box) throw new Error("paint viewport missing");
    const center = {
        x: box.x + box.width * 0.55,
        y: box.y + box.height * 0.52,
    };

    await page.mouse.move(center.x - 180, center.y - 18);
    await page.mouse.down();
    await page.mouse.move(center.x + 180, center.y + 18, { steps: 8 });
    await page.mouse.up();
    await expect(page.getByText("Charts 1")).toBeVisible();

    await page.getByRole("button", { name: "Snap" }).click();
    await page.mouse.move(center.x - 210, center.y + 60);
    await page.mouse.down();
    const start = Date.now();
    for (let i = 1; i <= 96; i++) {
        const t = i / 96;
        await page.mouse.move(
            center.x - 210 + t * 420,
            center.y + 60 + Math.sin(t * Math.PI * 8) * 58,
        );
    }
    const moveMs = Date.now() - start;
    await page.mouse.up();

    expect(moveMs).toBeLessThan(5_000);
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
