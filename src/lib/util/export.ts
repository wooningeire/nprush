/**
 * Utilities for exporting data from the browser (downloads, folder saves).
 */

/**
 * Triggers a browser download for a Blob.
 */
export function downloadBlob(blob: Blob, filename: string) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

/**
 * Converts an ImageData to a PNG Blob via an OffscreenCanvas (no DOM required).
 */
export async function imageDataToPngBlob(imageData: ImageData): Promise<Blob> {
    const canvas = new OffscreenCanvas(imageData.width, imageData.height);
    const ctx = canvas.getContext("2d")!;
    ctx.putImageData(imageData, 0, 0);
    return canvas.convertToBlob({ type: "image/png" });
}

/**
 * Asks the user to pick an output folder via the File System Access API, then
 * returns a writer that saves ImageData frames as sequentially-numbered PNGs
 * directly into that folder — one file write per frame, no memory buffering.
 *
 * When the FSA API is unavailable (non-Chromium browsers), falls back to a
 * ZIP writer that collects all frames and downloads a single .zip on close(),
 * avoiding the per-frame download-dialog spam of the old approach.
 *
 * Returns null only if the user explicitly cancels the folder picker.
 */
export async function openFrameWriter(): Promise<FrameWriter | null> {
    if ("showDirectoryPicker" in window) {
        let dirHandle: FileSystemDirectoryHandle;
        try {
            dirHandle = await (window as any).showDirectoryPicker({
                id: "nprush-export",
                mode: "readwrite",
                startIn: "downloads",
            });
        } catch (e: any) {
            if (e?.name !== "AbortError") console.error("Directory picker failed:", e);
            return null;
        }
        return new FsaFrameWriter(dirHandle);
    }

    // FSA unavailable — collect frames and zip them into a single download.
    return new ZipFrameWriter();
}

export interface FrameWriter {
    /** Save one frame. Resolves when the write is complete. */
    write(frame: ImageData): Promise<void>;
    /** Total frames written so far. */
    readonly count: number;
    /**
     * Finalise the export. For the ZIP fallback this triggers the single
     * download; for the FSA writer it is a no-op.
     */
    close(): Promise<void>;
}

// ---------------------------------------------------------------------------
// FSA writer — writes each frame directly to disk as it arrives.
// ---------------------------------------------------------------------------

class FsaFrameWriter implements FrameWriter {
    private readonly dir: FileSystemDirectoryHandle;
    count = 0;

    constructor(dir: FileSystemDirectoryHandle) {
        this.dir = dir;
    }

    async write(frame: ImageData): Promise<void> {
        const name = `frame_${String(this.count).padStart(4, "0")}.png`;
        const blob = await imageDataToPngBlob(frame);
        const fileHandle = await this.dir.getFileHandle(name, { create: true });
        const writable = await fileHandle.createWritable();
        await writable.write(blob);
        await writable.close();
        this.count++;
    }

    async close() { /* nothing to do */ }
}

// ---------------------------------------------------------------------------
// ZIP fallback — accumulates frames then downloads a single .zip on close().
// ---------------------------------------------------------------------------

class ZipFrameWriter implements FrameWriter {
    count = 0;
    private readonly entries: Array<{ name: string; data: Uint8Array }> = [];

    async write(frame: ImageData): Promise<void> {
        const name = `frame_${String(this.count).padStart(4, "0")}.png`;
        const blob = await imageDataToPngBlob(frame);
        const data = new Uint8Array(await blob.arrayBuffer());
        this.entries.push({ name, data });
        this.count++;
    }

    async close(): Promise<void> {
        if (this.entries.length === 0) return;
        const zip = buildUncompressedZip(this.entries);
        downloadBlob(new Blob([zip], { type: "application/zip" }), `nprush-turntable-${Date.now()}.zip`);
    }
}

// ---------------------------------------------------------------------------
// Minimal uncompressed ZIP builder (no dependencies).
// Spec ref: https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT
// Uncompressed (method 0) is correct for PNGs — they are already compressed.
// ---------------------------------------------------------------------------

function buildUncompressedZip(entries: Array<{ name: string; data: Uint8Array }>): Uint8Array {
    const enc = new TextEncoder();

    // Collect local-file records and central-directory entries.
    const localParts: Uint8Array[] = [];
    const centralParts: Uint8Array[] = [];
    const offsets: number[] = [];
    let offset = 0;

    for (const { name, data } of entries) {
        const nameBytes = enc.encode(name);
        const crc = crc32(data);
        const size = data.length;

        // Local file header (30 bytes) + filename + data
        const local = new Uint8Array(30 + nameBytes.length + size);
        const lv = new DataView(local.buffer);
        lv.setUint32(0,  0x04034b50, true); // signature
        lv.setUint16(4,  20,         true); // version needed
        lv.setUint16(6,  0,          true); // flags
        lv.setUint16(8,  0,          true); // compression: stored
        lv.setUint16(10, 0,          true); // mod time
        lv.setUint16(12, 0,          true); // mod date
        lv.setUint32(14, crc,        true); // crc-32
        lv.setUint32(18, size,       true); // compressed size
        lv.setUint32(22, size,       true); // uncompressed size
        lv.setUint16(26, nameBytes.length, true); // filename length
        lv.setUint16(28, 0,          true); // extra field length
        local.set(nameBytes, 30);
        local.set(data, 30 + nameBytes.length);

        offsets.push(offset);
        localParts.push(local);
        offset += local.length;

        // Central directory entry (46 bytes) + filename
        const central = new Uint8Array(46 + nameBytes.length);
        const cv = new DataView(central.buffer);
        cv.setUint32(0,  0x02014b50, true); // signature
        cv.setUint16(4,  20,         true); // version made by
        cv.setUint16(6,  20,         true); // version needed
        cv.setUint16(8,  0,          true); // flags
        cv.setUint16(10, 0,          true); // compression: stored
        cv.setUint16(12, 0,          true); // mod time
        cv.setUint16(14, 0,          true); // mod date
        cv.setUint32(16, crc,        true); // crc-32
        cv.setUint32(20, size,       true); // compressed size
        cv.setUint32(24, size,       true); // uncompressed size
        cv.setUint16(28, nameBytes.length, true); // filename length
        cv.setUint16(30, 0,          true); // extra field length
        cv.setUint16(32, 0,          true); // comment length
        cv.setUint16(34, 0,          true); // disk number start
        cv.setUint16(36, 0,          true); // internal attributes
        cv.setUint32(38, 0,          true); // external attributes
        cv.setUint32(42, offsets[offsets.length - 1], true); // local header offset
        central.set(nameBytes, 46);
        centralParts.push(central);
    }

    // End of central directory record (22 bytes)
    const cdSize = centralParts.reduce((s, p) => s + p.length, 0);
    const eocd = new Uint8Array(22);
    const ev = new DataView(eocd.buffer);
    ev.setUint32(0,  0x06054b50,          true); // signature
    ev.setUint16(4,  0,                   true); // disk number
    ev.setUint16(6,  0,                   true); // disk with cd
    ev.setUint16(8,  entries.length,      true); // entries on disk
    ev.setUint16(10, entries.length,      true); // total entries
    ev.setUint32(12, cdSize,              true); // cd size
    ev.setUint32(16, offset,              true); // cd offset
    ev.setUint16(20, 0,                   true); // comment length

    // Concatenate everything
    const total = offset + cdSize + eocd.length;
    const out = new Uint8Array(total);
    let pos = 0;
    for (const p of localParts)   { out.set(p, pos); pos += p.length; }
    for (const p of centralParts) { out.set(p, pos); pos += p.length; }
    out.set(eocd, pos);
    return out;
}

/** CRC-32 using the standard polynomial (ISO 3309). */
function crc32(data: Uint8Array): number {
    let crc = 0xffffffff;
    for (let i = 0; i < data.length; i++) {
        crc ^= data[i];
        for (let j = 0; j < 8; j++) {
            crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
        }
    }
    return (crc ^ 0xffffffff) >>> 0;
}
