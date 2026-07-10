import { expect, test, type Locator, type Page } from "playwright/test";

type Vec3 = [number, number, number];

type StrokePlacementDebug = {
    positions: Vec3[],
    provenance: string[],
};

test("surface placement bridges GPU hit gaps with continuous tangents", async ({ page }) => {
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
    await addObjectAndWait(page);

    const rendererUnavailable = await page.getByText("WebGPU unavailable").isVisible().catch(() => false);
    test.skip(rendererUnavailable, "WebGPU is required for surface bridge verification");

    const viewport = page.locator("paint-viewport");
    const box = await viewport.boundingBox();
    if (!box) throw new Error("paint viewport missing");
    const center = {
        x: box.x + box.width * 0.5,
        y: box.y + box.height * 0.52,
    };

    const paintOn = page.getByLabel("Paint on");
    await paintOn.selectOption("construction-plane");
    const planeEditor = page.getByLabel("Construction plane");
    await planeEditor.getByRole("button", { name: "View" }).click();
    const depth = page.getByLabel("Construction plane depth");

    await setNumericInput(depth, "0.75");
    await drawStroke(page, center.x - 310, center.x - 90, center.y);
    await expect(page.getByText("Strokes 1")).toBeVisible();

    await setNumericInput(depth, "1.25");
    await drawStroke(page, center.x + 90, center.x + 310, center.y);
    await expect(page.getByText("Strokes 2")).toBeVisible();

    await paintOn.selectOption("surface");
    await drawStroke(page, center.x - 340, center.x + 340, center.y);
    await expect(page.getByText("Strokes 3")).toBeVisible();

    const placement = await readLastStrokePlacement(page);
    if (!placement) throw new Error("stroke placement diagnostics missing");

    expect(placement.positions).toHaveLength(placement.provenance.length);
    expect(placement.provenance).toContain("surface");
    expect(placement.provenance).toContain("bridge");
    expect(placement.positions.flat().every(Number.isFinite)).toBe(true);

    const transitionDots = transitionTangentDots(placement);
    expect(transitionDots.length).toBeGreaterThanOrEqual(2);
    for (const dot of transitionDots) {
        expect(dot).toBeGreaterThan(0.8);
    }

    expect(consoleProblems).toEqual([]);
});

const readLastStrokePlacement = async (page: Page): Promise<StrokePlacementDebug | null> => (
    await page.evaluate(() => {
        const debug = (window as typeof window & {
            __paintModelerDebug?: {
                readLastStrokePlacement: () => StrokePlacementDebug | null,
            },
        }).__paintModelerDebug;
        return debug?.readLastStrokePlacement() ?? null;
    })
);

const transitionTangentDots = ({
    positions,
    provenance,
}: StrokePlacementDebug): number[] => {
    const dots: number[] = [];
    for (let index = 1; index < provenance.length; index++) {
        if (provenance[index] === provenance[index - 1]) continue;
        if (provenance[index] !== "bridge" && provenance[index - 1] !== "bridge") continue;

        const boundary = provenance[index] === "bridge" ? index - 1 : index;
        if (boundary < 1 || boundary + 1 >= positions.length) continue;

        const incoming = normalize3(sub3(positions[boundary], positions[boundary - 1]));
        const outgoing = normalize3(sub3(positions[boundary + 1], positions[boundary]));
        dots.push(dot3(incoming, outgoing));
    }
    return dots;
};

const drawStroke = async (
    page: Page,
    fromX: number,
    toX: number,
    y: number,
): Promise<void> => {
    await page.mouse.move(fromX, y);
    await page.mouse.down();
    await page.mouse.move(toX, y, { steps: 28 });
    await page.mouse.up();
    await waitForAnimationFrame(page);
};

const setNumericInput = async (
    input: Locator,
    value: string,
): Promise<void> => {
    await input.fill(value);
    await input.press("Tab");
    expect(Number(await input.inputValue())).toBeCloseTo(Number(value), 6);
};

const sub3 = (a: Vec3, b: Vec3): Vec3 => [
    a[0] - b[0],
    a[1] - b[1],
    a[2] - b[2],
];

const dot3 = (a: Vec3, b: Vec3): number => (
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
);

const normalize3 = (value: Vec3): Vec3 => {
    const length = Math.hypot(...value);
    if (length <= 0.000001) return [0, 0, 0];
    return [value[0] / length, value[1] / length, value[2] / length];
};

const addObjectAndWait = async (page: Page): Promise<void> => {
    const row = page.locator("section").filter({ hasText: "Objects" }).getByRole("button", { name: /Object 1\s+0s/ });
    for (let attempt = 0; attempt < 4; attempt++) {
        if (await row.count() > 0) return;
        await page.getByRole("button", { name: "Add Object" }).click();
        await waitForAnimationFrame(page);
    }
    await expect(row).toBeVisible();
};

const isExpectedStartupNoise = (text: string): boolean => {
    return text === "Failed to load resource: net::ERR_CONNECTION_CLOSED"
        || text === "Failed to load resource: net::ERR_TIMED_OUT"
        || text.includes("ResizeObserver loop completed")
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