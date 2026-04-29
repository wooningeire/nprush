/**
 * Loads a Radiance HDR (.hdr) file and uploads it to the GPU as an rgba16float texture.
 *
 * The Radiance RGBE format stores each pixel as (R, G, B, E) where E is a shared
 * exponent. Each channel value is decoded as: channel = mantissa * 2^(E - 128 - 8)
 */
export async function loadHdrTexture(device: GPUDevice, url: string): Promise<GPUTexture> {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    const bytes = new Uint8Array(buffer);

    const { width, height, pixels } = parseHdr(bytes);

    // Upload as rgba16float — sufficient precision for an environment map and
    // avoids the need for the float32-filterable feature.
    const texture = device.createTexture({
        label: url,
        size: [width, height],
        format: "rgba16float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    // Pack decoded HDR floats into Float16 (via Float32Array → Uint16Array conversion).
    const f16data = toFloat16Array(pixels);

    device.queue.writeTexture(
        { texture },
        f16data,
        { bytesPerRow: width * 4 * 2 }, // 4 channels × 2 bytes per f16
        [width, height],
    );

    return texture;
}

// ---------------------------------------------------------------------------
// Radiance HDR / RGBE parser
// ---------------------------------------------------------------------------

interface HdrImage {
    width: number;
    height: number;
    /** Decoded linear-light RGBA floats, length = width * height * 4 */
    pixels: Float32Array;
}

function parseHdr(bytes: Uint8Array): HdrImage {
    let pos = 0;

    // --- ASCII header ---
    // Must start with "#?RADIANCE" or "#?RGBE"
    const headerEnd = findHeaderEnd(bytes);
    const headerText = new TextDecoder().decode(bytes.subarray(0, headerEnd));
    if (!headerText.startsWith("#?RADIANCE") && !headerText.startsWith("#?RGBE")) {
        throw new Error("Not a valid Radiance HDR file");
    }
    pos = headerEnd;

    // Skip the blank line that terminates the header
    while (pos < bytes.length && bytes[pos] !== 0x0a) pos++;
    pos++; // consume the newline

    // --- Resolution string, e.g. "-Y 1024 +X 2048\n" ---
    const resLineStart = pos;
    while (pos < bytes.length && bytes[pos] !== 0x0a) pos++;
    const resLine = new TextDecoder().decode(bytes.subarray(resLineStart, pos));
    pos++; // consume newline

    const resMatch = resLine.match(/[-+][YX]\s+(\d+)\s+[-+][YX]\s+(\d+)/);
    if (!resMatch) throw new Error(`Cannot parse HDR resolution line: "${resLine}"`);

    // Standard orientation is "-Y H +X W"
    let width: number, height: number;
    if (resLine.trimStart().startsWith("-Y")) {
        height = parseInt(resMatch[1], 10);
        width  = parseInt(resMatch[2], 10);
    } else {
        width  = parseInt(resMatch[1], 10);
        height = parseInt(resMatch[2], 10);
    }

    const pixels = new Float32Array(width * height * 4);

    // --- Scanline data (new RLE encoding) ---
    const scanline = new Uint8Array(width * 4);
    let pixelOffset = 0;

    for (let y = 0; y < height; y++) {
        // Each scanline starts with a 4-byte marker
        if (pos + 4 > bytes.length) break;

        const r0 = bytes[pos];
        const r1 = bytes[pos + 1];
        const r2 = bytes[pos + 2];
        const r3 = bytes[pos + 3];

        if (r0 === 2 && r1 === 2 && (r2 & 0x80) === 0) {
            // New RLE format
            const scanWidth = (r2 << 8) | r3;
            if (scanWidth !== width) throw new Error("HDR scanline width mismatch");
            pos += 4;

            // Decode 4 channels separately
            for (let ch = 0; ch < 4; ch++) {
                let x = 0;
                while (x < width) {
                    if (pos >= bytes.length) throw new Error("Unexpected end of HDR data");
                    const code = bytes[pos++];
                    if (code > 128) {
                        // RLE run
                        const count = code - 128;
                        const val = bytes[pos++];
                        for (let i = 0; i < count; i++) scanline[x++ * 4 + ch] = val;
                    } else {
                        // Non-run
                        for (let i = 0; i < code; i++) scanline[x++ * 4 + ch] = bytes[pos++];
                    }
                }
            }
        } else {
            // Old/uncompressed format — read raw RGBE quads
            for (let x = 0; x < width; x++) {
                scanline[x * 4 + 0] = bytes[pos++];
                scanline[x * 4 + 1] = bytes[pos++];
                scanline[x * 4 + 2] = bytes[pos++];
                scanline[x * 4 + 3] = bytes[pos++];
            }
        }

        // Decode RGBE → linear float
        for (let x = 0; x < width; x++) {
            const ri = scanline[x * 4 + 0];
            const gi = scanline[x * 4 + 1];
            const bi = scanline[x * 4 + 2];
            const ei = scanline[x * 4 + 3];

            if (ei === 0) {
                pixels[pixelOffset++] = 0;
                pixels[pixelOffset++] = 0;
                pixels[pixelOffset++] = 0;
                pixels[pixelOffset++] = 1;
            } else {
                const scale = Math.pow(2, ei - 128 - 8);
                pixels[pixelOffset++] = ri * scale;
                pixels[pixelOffset++] = gi * scale;
                pixels[pixelOffset++] = bi * scale;
                pixels[pixelOffset++] = 1.0;
            }
        }
    }

    return { width, height, pixels };
}

/** Find the byte offset of the end of the Radiance HDR ASCII header.
 *  The header ends with a blank line (two consecutive newlines). */
function findHeaderEnd(bytes: Uint8Array): number {
    for (let i = 0; i < bytes.length - 1; i++) {
        if (bytes[i] === 0x0a && bytes[i + 1] === 0x0a) {
            return i + 1; // include the first newline, stop before the second
        }
    }
    throw new Error("Could not find end of HDR header");
}

// ---------------------------------------------------------------------------
// Float32 → Float16 conversion
// ---------------------------------------------------------------------------

/** Convert a Float32Array to a Uint16Array of IEEE 754 half-precision values. */
function toFloat16Array(f32: Float32Array): Uint16Array {
    const f16 = new Uint16Array(f32.length);
    for (let i = 0; i < f32.length; i++) {
        f16[i] = floatToHalf(f32[i]);
    }
    return f16;
}

function floatToHalf(val: number): number {
    if (val === 0) return 0;
    if (isNaN(val)) return 0x7e00; // NaN
    if (!isFinite(val)) return val > 0 ? 0x7c00 : 0xfc00; // ±Inf

    const sign = val < 0 ? 1 : 0;
    val = Math.abs(val);

    // Clamp to max half-float value (~65504)
    if (val > 65504) val = 65504;

    // Subnormal range
    if (val < 6.103515625e-5) {
        // Encode as subnormal
        const mantissa = Math.round(val / 5.960464477539063e-8);
        return (sign << 15) | mantissa;
    }

    // Use DataView trick for reliable bit manipulation
    const buf = new ArrayBuffer(4);
    const view = new DataView(buf);
    view.setFloat32(0, val, false);
    const bits = view.getUint32(0, false);

    const exp32 = (bits >>> 23) & 0xff;
    const mant32 = bits & 0x7fffff;

    const exp16 = exp32 - 127 + 15;
    if (exp16 >= 31) return (sign << 15) | 0x7c00; // overflow → Inf
    if (exp16 <= 0) {
        // Subnormal half
        const m = (mant32 | 0x800000) >> (1 - exp16);
        return (sign << 15) | (m >> 13);
    }

    return (sign << 15) | (exp16 << 10) | (mant32 >> 13);
}
