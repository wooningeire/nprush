import { defineConfig } from "playwright/test";

export default defineConfig({
    testDir: "./e2e",
    timeout: 60_000,
    expect: {
        timeout: 10_000,
    },
    use: {
        baseURL: "http://127.0.0.1:5180",
        browserName: "chromium",
        trace: "retain-on-failure",
        viewport: { width: 1342, height: 967 },
    },
    webServer: {
        command: "npm run dev -- --host 127.0.0.1 --port 5180",
        url: "http://127.0.0.1:5180",
        reuseExistingServer: true,
        timeout: 120_000,
    },
});
