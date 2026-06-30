import { inflateSync } from "node:zlib";
import { expect, test, type Page } from "playwright/test";

type CanvasGuideSample = {
    width: number,
    height: number,
    nonBackground: number,
    redDominant: number,
    greenDominant: number,
};

type DecodedPng = {
    width: number,
    height: number,
    channels: number,
    pixels: Uint8Array,
};

test("paint modeler ground grid and axes stay visible while orbiting", async ({ page }) => {
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
    await expect(page.locator("paint-viewport")).toBeVisible();
    await page.waitForTimeout(750);

    const rendererUnavailable = await page.getByText("WebGPU unavailable").isVisible().catch(() => false);
    if (rendererUnavailable) {
        await expect(page.getByText("WebGPU unavailable")).toBeVisible();
        expect(consoleProblems).toEqual([]);
        return;
    }

    const viewport = page.locator("paint-viewport");
    const box = await viewport.boundingBox();
    if (!box) throw new Error("paint viewport missing");
    const center = {
        x: box.x + box.width * 0.5,
        y: box.y + box.height * 0.5,
    };

    for (const movement of [
        { x: 0, y: 0 },
        { x: 210, y: 26 },
        { x: -320, y: -44 },
        { x: 160, y: 72 },
    ]) {
        if (movement.x !== 0 || movement.y !== 0) {
            await page.mouse.move(center.x, center.y);
            await page.mouse.down({ button: "middle" });
            await page.mouse.move(center.x + movement.x, center.y + movement.y, { steps: 12 });
            await page.mouse.up({ button: "middle" });
        }
        await page.waitForTimeout(120);

        const sample = await sampleCanvasGuidePixels(page);
        expect(sample.nonBackground / (sample.width * sample.height)).toBeGreaterThan(0.003);
        expect(sample.redDominant + sample.greenDominant).toBeGreaterThan(80);
        expect(Math.max(sample.redDominant, sample.greenDominant)).toBeGreaterThan(40);
    }

    expect(consoleProblems).toEqual([]);
});

async function sampleCanvasGuidePixels(page: Page): Promise<CanvasGuideSample> {
    const png = await page.locator("canvas").screenshot();
    const image = decodePng(png);
    let nonBackground = 0;
    let redDominant = 0;
    let greenDominant = 0;
    for (let i = 0; i < image.pixels.length; i += image.channels) {
        const r = image.pixels[i];
        const g = image.pixels[i + 1];
        const b = image.pixels[i + 2];
        const backgroundDelta = Math.abs(r - 9) + Math.abs(g - 11) + Math.abs(b - 12);
        if (backgroundDelta > 12) nonBackground++;
        if (r > g + 16 && r > b + 16) redDominant++;
        if (g > r + 16 && g > b + 16) greenDominant++;
    }

    return {
        width: image.width,
        height: image.height,
        nonBackground,
        redDominant,
        greenDominant,
    };
}

function decodePng(png: Buffer): DecodedPng {
    let offset = 8;
    let width = 0;
    let height = 0;
    let channels = 0;
    const idatChunks: Buffer[] = [];

    while (offset < png.length) {
        const length = png.readUInt32BE(offset);
        const type = png.toString("ascii", offset + 4, offset + 8);
        const dataStart = offset + 8;
        const dataEnd = dataStart + length;
        const data = png.subarray(dataStart, dataEnd);
        if (type === "IHDR") {
            width = data.readUInt32BE(0);
            height = data.readUInt32BE(4);
            const bitDepth = data[8];
            const colorType = data[9];
            if (bitDepth !== 8 || (colorType !== 2 && colorType !== 6)) {
                throw new Error(`Unsupported PNG format: bit depth ${bitDepth}, color type ${colorType}`);
            }
            channels = colorType === 6 ? 4 : 3;
        } else if (type === "IDAT") {
            idatChunks.push(data);
        } else if (type === "IEND") {
            break;
        }
        offset = dataEnd + 4;
    }

    if (width <= 0 || height <= 0 || channels <= 0) throw new Error("PNG header missing");
    const inflated = inflateSync(Buffer.concat(idatChunks));
    const stride = width * channels;
    const pixels = new Uint8Array(width * height * channels);
    let source = 0;
    let target = 0;

    for (let y = 0; y < height; y++) {
        const filter = inflated[source++];
        const rowStart = target;
        for (let x = 0; x < stride; x++) {
            const raw = inflated[source++];
            const left = x >= channels ? pixels[target - channels] : 0;
            const up = y > 0 ? pixels[target - stride] : 0;
            const upLeft = y > 0 && x >= channels ? pixels[target - stride - channels] : 0;
            pixels[target++] = unfilterByte(filter, raw, left, up, upLeft);
        }
        if (target - rowStart !== stride) throw new Error("PNG row decode mismatch");
    }

    return { width, height, channels, pixels };
}

function unfilterByte(filter: number, raw: number, left: number, up: number, upLeft: number): number {
    if (filter === 0) return raw;
    if (filter === 1) return (raw + left) & 255;
    if (filter === 2) return (raw + up) & 255;
    if (filter === 3) return (raw + Math.floor((left + up) / 2)) & 255;
    if (filter === 4) return (raw + paeth(left, up, upLeft)) & 255;
    throw new Error(`Unsupported PNG filter ${filter}`);
}

function paeth(left: number, up: number, upLeft: number): number {
    const p = left + up - upLeft;
    const pa = Math.abs(p - left);
    const pb = Math.abs(p - up);
    const pc = Math.abs(p - upLeft);
    if (pa <= pb && pa <= pc) return left;
    if (pb <= pc) return up;
    return upLeft;
}

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
