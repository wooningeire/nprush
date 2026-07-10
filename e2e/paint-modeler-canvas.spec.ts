import { expect, test, type Locator, type Page } from "playwright/test";

test("paint modeler canvas supports stroke-owned ribbon surfaces", async ({ page }) => {
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
    await dismissStartupAlerts(page);
    await expect(page.locator("paint-viewport")).toBeVisible();
    await waitForAnimationFrame(page);
    await expect(page.getByText("Stroke-owned ribbon prototype")).toBeVisible();

    await expect(page.getByRole("button", { name: "Surface" })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Depth" })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Snap" })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "View Plane" })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Chart wire" })).toHaveCount(0);
    await expect(page.getByLabel("Surface field")).toHaveCount(0);
    await expect(page.getByLabel("Opacity")).toHaveCount(0);
    await expect(page.getByRole("group", { name: "Brush mode" })).toHaveCount(0);
    await expect(page.getByRole("group", { name: "Stroke geometry" })).toHaveCount(0);
    await expect(page.getByText(/Charts \d+/)).toHaveCount(0);

    await addObjectAndWait(page, "Object 1");

    const paintLayers = page.locator("section").filter({ hasText: "Paint Layers" });
    await expect(paintLayers.getByRole("button", { name: /Layer 1\s+0s/ })).toHaveClass(/active/);
    await paintLayers.getByRole("button", { name: "Add" }).click();
    await expect(paintLayers.getByRole("button", { name: /Layer 2\s+0s/ })).toHaveClass(/active/);
    await paintLayers.getByRole("button", { name: "Add" }).click();
    await expect(paintLayers.getByRole("button", { name: /Layer 3\s+0s/ })).toHaveClass(/active/);

    await dragRowBefore(
        page,
        paintLayers.locator(".layer-row").filter({ hasText: "Layer 3" }),
        paintLayers.locator(".layer-row").filter({ hasText: "Layer 1" }),
    );
    await expect(paintLayers.locator(".layer-row").first()).toContainText("Layer 3");
    await paintLayers.getByRole("button", { name: /Layer 1\s+0s/ }).click();
    await expect(paintLayers.getByRole("button", { name: /Layer 1\s+0s/ })).toHaveClass(/active/);

    const layerBox = await paintLayers.boundingBox();
    const controlPanelBox = await page.locator(".control-panel").boundingBox();
    if (!layerBox || !controlPanelBox) throw new Error("paint layer controls missing");
    expect(layerBox.x).toBeGreaterThanOrEqual(controlPanelBox.x - 1);
    expect(layerBox.x + layerBox.width).toBeLessThanOrEqual(controlPanelBox.x + controlPanelBox.width + 1);

    const viewport = page.locator("paint-viewport");
    const firstBox = await viewport.boundingBox();
    if (!firstBox) throw new Error("paint viewport missing");
    const firstCenter = {
        x: firstBox.x + firstBox.width * 0.5,
        y: firstBox.y + firstBox.height * 0.5,
    };

    await expect(viewport).toHaveCSS("cursor", "crosshair");
    await page.mouse.move(firstCenter.x, firstCenter.y);
    await waitForAnimationFrame(page);
    await expect(viewport).toHaveCSS("cursor", "none");
    await page.mouse.down({ button: "middle" });
    await page.mouse.move(firstCenter.x + 64, firstCenter.y + 18, { steps: 4 });
    await page.mouse.up({ button: "middle" });
    await addObjectAndWait(page, "Object 2");

    const objects = page.locator("section").filter({ hasText: "Objects" });
    await dragRowBefore(
        page,
        objects.locator(".list-row").filter({ hasText: "Object 2" }),
        objects.locator(".list-row").filter({ hasText: "Object 1" }),
    );
    await expect(objects.locator(".list-row").first()).toContainText("Object 2");

    const views = page.locator("section").filter({ hasText: "Views" });
    await dragRowBefore(
        page,
        views.locator(".list-row").filter({ hasText: "View 2" }),
        views.locator(".list-row").filter({ hasText: "View 1" }),
    );
    await expect(views.locator(".list-row").first()).toContainText("View 2");

    await expect(page.getByLabel("Shade ribbons")).toBeChecked();
    await expect(page.getByLabel("Shade ribbons")).toBeEnabled();
    const paintOn = page.getByLabel("Paint on");
    await expect(paintOn).toHaveValue("view");
    await expect(paintOn.locator("option")).toHaveText([
        "View",
        "Start depth",
        "Start plane",
        "Surface",
        "Construction plane",
    ]);
    await paintOn.selectOption("start-depth");
    await expect(paintOn).toHaveValue("start-depth");
    await paintOn.selectOption("start-plane");
    await expect(paintOn).toHaveValue("start-plane");
    await paintOn.selectOption("view");
    await expect(page.getByLabel("Width")).toHaveValue("18");
    await page.getByLabel("Width").evaluate(element => {
        const input = element as HTMLInputElement;
        input.value = "42";
        input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await expect(page.getByLabel("Width")).toHaveValue("42");
    await page.getByLabel("Shade ribbons").uncheck();
    await expect(page.getByLabel("Shade ribbons")).not.toBeChecked();
    await page.getByLabel("Shade ribbons").check();
    await expect(page.getByLabel("Shade ribbons")).toBeChecked();

    const strokeBox = await viewport.boundingBox();
    if (!strokeBox) throw new Error("paint viewport missing");
    const center = {
        x: strokeBox.x + strokeBox.width * 0.56,
        y: strokeBox.y + strokeBox.height * 0.52,
    };

    await drawRibbonUntilStrokeCount(page, center.x, center.y, 1);
    await expect(paintLayers.getByRole("button", { name: /Layer 1\s+1s/ })).toBeVisible();
    await expect(objects.getByRole("button", { name: /Object 2\s+1s/ })).toBeVisible();

    await paintOn.selectOption("surface");
    await expect(paintOn).toHaveValue("surface");

    const rendererUnavailable = await page.getByText("WebGPU unavailable").isVisible().catch(() => false);
    if (!rendererUnavailable) {
        await page.mouse.move(center.x + 40, center.y + 8);
        await waitForAnimationFrame(page);
        const placement = await readBrushPlacement(page);
        expect(placement?.provenance).toBe("surface");
        expect(placement?.snapped).toBe(true);
        if (!placement) throw new Error("brush placement missing");
        expectGuideBasisMatchesNormal(placement);

        const beforeOrbit = await page.locator("canvas").screenshot();
        await page.mouse.move(center.x, center.y);
        await page.mouse.down({ button: "middle" });
        await page.mouse.move(center.x + 120, center.y + 24, { steps: 8 });
        await page.mouse.up({ button: "middle" });
        await waitForAnimationFrame(page);
        const tiltedSurface = await findSurfaceBrushPlacement(page, viewport);
        expectGuideBasisMatchesNormal(tiltedSurface.placement);
        const afterOrbit = await page.locator("canvas").screenshot();
        expect(afterOrbit.equals(beforeOrbit)).toBe(false);

        await paintOn.selectOption("construction-plane");
        await expect(page.getByLabel("Construction plane")).toBeVisible();
        const depthInput = page.getByLabel("Construction plane depth");
        await expect(depthInput).toBeVisible();
        await depthInput.fill("1.35");
        await depthInput.press("Tab");
        expect(Number(await depthInput.inputValue())).toBeCloseTo(1.35, 6);

        await page.getByLabel("Align construction plane to Z axis").click();
        await page.mouse.move(tiltedSurface.point.x, tiltedSurface.point.y);
        await waitForAnimationFrame(page);
        const constructionPlacement = await readBrushPlacement(page);
        expect(constructionPlacement?.provenance).toBe("construction-plane");
        expect(constructionPlacement?.normal[0]).toBeCloseTo(0, 5);
        expect(constructionPlacement?.normal[1]).toBeCloseTo(0, 5);
        expect(constructionPlacement?.normal[2]).toBeCloseTo(1, 5);
        const withConstructionGrid = await page.locator("canvas").screenshot();
        expect(withConstructionGrid.equals(afterOrbit)).toBe(false);

        const pickButton = page.getByRole("button", { name: "Pick" });
        await pickButton.click();
        await expect(pickButton).toHaveAttribute("aria-pressed", "true");
        await page.mouse.click(tiltedSurface.point.x, tiltedSurface.point.y);
        await expect(pickButton).toHaveAttribute("aria-pressed", "false");
    }

    await expect(page.getByRole("button", { name: "Chart wire" })).toHaveCount(0);
    await expect(page.getByLabel("Surface field")).toHaveCount(0);
    expect(consoleProblems).toEqual([]);
});


type Vec3Tuple = [number, number, number];

type BrushPlacementDebugResult = {
    normal: Vec3Tuple,
    tangent: Vec3Tuple,
    bitangent: Vec3Tuple,
    provenance: string,
    snapped: boolean,
};

type FoundBrushPlacement = {
    placement: BrushPlacementDebugResult,
    point: { x: number, y: number },
};

const readBrushPlacement = async (page: Page): Promise<BrushPlacementDebugResult | null> => {
    return await page.evaluate(async () => {
        const debug = (window as typeof window & {
            __paintModelerDebug?: {
                readBrushPlacement: () => Promise<BrushPlacementDebugResult | null>,
            },
        }).__paintModelerDebug;
        return await debug?.readBrushPlacement() ?? null;
    });
};
const findSurfaceBrushPlacement = async (page: Page, viewport: Locator): Promise<FoundBrushPlacement> => {
    const box = await viewport.boundingBox();
    if (!box) throw new Error("paint viewport missing");

    const fractions = [0.34, 0.42, 0.50, 0.58, 0.66];
    for (const yFraction of fractions) {
        for (const xFraction of fractions) {
            const point = {
                x: box.x + box.width * xFraction,
                y: box.y + box.height * yFraction,
            };
            await page.mouse.move(point.x, point.y);
            await waitForAnimationFrame(page);
            const placement = await readBrushPlacement(page);
            if (placement?.provenance === "surface") {
                return { placement, point };
            }
        }
    }

    throw new Error("Could not find surface brush placement after orbit");
};

const expectGuideBasisMatchesNormal = (placement: BrushPlacementDebugResult): void => {
    const normal = normalize3(placement.normal);
    const tangent = placement.tangent;
    const bitangent = placement.bitangent;
    expect(length3(tangent)).toBeGreaterThan(0.000001);
    expect(length3(bitangent)).toBeGreaterThan(0.000001);
    expect(Math.abs(dot3(normal, tangent))).toBeLessThan(0.001);
    expect(Math.abs(dot3(normal, bitangent))).toBeLessThan(0.001);
    expect(dot3(normalize3(cross3(tangent, bitangent)), normal)).toBeGreaterThan(0.98);
};

const dot3 = (a: Vec3Tuple, b: Vec3Tuple): number => (
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
);

const cross3 = (a: Vec3Tuple, b: Vec3Tuple): Vec3Tuple => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
];

const length3 = (value: Vec3Tuple): number => Math.hypot(value[0], value[1], value[2]);

const normalize3 = (value: Vec3Tuple): Vec3Tuple => {
    const length = length3(value);
    if (length <= 0.000001) return [0, 0, 0];
    return [value[0] / length, value[1] / length, value[2] / length];
};
const addObjectAndWait = async (page: Page, objectName: string): Promise<void> => {
    const objects = page.locator("section").filter({ hasText: "Objects" });
    const row = objects.getByRole("button", { name: new RegExp(`${objectName}\\s+0s`) });
    for (let attempt = 0; attempt < 4; attempt++) {
        if (await row.count() > 0) return;
        await page.getByRole("button", { name: "Add Object" }).click();
        await waitForAnimationFrame(page);
    }
    await expect(row).toBeVisible();
};

const drawRibbonUntilStrokeCount = async (
    page: Page,
    cx: number,
    cy: number,
    strokeCount: number,
): Promise<void> => {
    for (let attempt = 0; attempt < 3; attempt++) {
        await drawRibbon(page, cx, cy);
        if (await isStrokeCountVisible(page, strokeCount)) return;
    }
    await expect(page.getByText(`Strokes ${strokeCount}`)).toBeVisible();
};

const drawRibbon = async (page: Page, cx: number, cy: number): Promise<void> => {
    await page.mouse.move(cx - 150, cy - 20);
    await page.mouse.down();
    for (const [dx, dy] of [
        [-95, 10],
        [-35, -12],
        [40, 8],
        [105, -10],
        [155, 12],
    ]) {
        await page.mouse.move(cx + dx, cy + dy, { steps: 5 });
    }
    await page.mouse.up();
    await waitForAnimationFrame(page);
};

const isStrokeCountVisible = async (page: Page, count: number): Promise<boolean> => {
    return page.getByText(`Strokes ${count}`).isVisible({ timeout: 1_500 }).catch(() => false);
};

const dragRowBefore = async (page: Page, source: Locator, target: Locator): Promise<void> => {
    await source.scrollIntoViewIfNeeded();
    await target.scrollIntoViewIfNeeded();
    await source.dragTo(target);
    await waitForAnimationFrame(page);
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

const forbidWebglContexts = async (page: Page): Promise<void> => {
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
};