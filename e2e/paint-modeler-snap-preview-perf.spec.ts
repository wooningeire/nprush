import { expect, test, type Page } from "playwright/test";

test("paint modeler ribbon preview stays responsive after the first stroke", async ({ page }) => {
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
    await dismissStartupAlerts(page);
    await waitForAnimationFrame(page);
    await addObjectAndWait(page);

    const viewport = page.locator("paint-viewport");
    await expect(viewport).toBeVisible();
    const box = await viewport.boundingBox();
    if (!box) throw new Error("paint viewport missing");
    const center = {
        x: box.x + box.width * 0.55,
        y: box.y + box.height * 0.52,
    };

    await drawStraightStrokeUntilCount(page, center.x, center.y, 1);

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
    await expect(page.getByText("Strokes 2")).toBeVisible();
    expect(consoleProblems).toEqual([]);
});

const addObjectAndWait = async (page: Page): Promise<void> => {
    const row = page.locator("section").filter({ hasText: "Objects" }).getByRole("button", { name: /Object 1\s+0s/ });
    for (let attempt = 0; attempt < 4; attempt++) {
        if (await row.count() > 0) return;
        await page.getByRole("button", { name: "Add Object" }).click();
        await waitForAnimationFrame(page);
    }
    await expect(row).toBeVisible();
};

const drawStraightStrokeUntilCount = async (
    page: Page,
    cx: number,
    cy: number,
    strokeCount: number,
): Promise<void> => {
    for (let attempt = 0; attempt < 3; attempt++) {
        await page.mouse.move(cx - 180, cy - 18);
        await page.mouse.down();
        await page.mouse.move(cx + 180, cy + 18, { steps: 8 });
        await page.mouse.up();
        await waitForAnimationFrame(page);
        if (await page.getByText(`Strokes ${strokeCount}`).isVisible({ timeout: 1_500 }).catch(() => false)) return;
    }
    await expect(page.getByText(`Strokes ${strokeCount}`)).toBeVisible();
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

const waitForAnimationFrame = async (page: Page): Promise<void> => {
    await page.evaluate(() => new Promise<void>(resolve => {
        requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
    }));
};