import path from "node:path";

import { defineConfig, devices } from "@playwright/test";

// End-to-end test against the REAL stack: Next.js -> FastAPI -> database -> trained model bundle.
// Prerequisites (CI does these; see .github/workflows/ci.yml):
//   tg data prepare synthetic && tg train synthetic --budget smoke && tg db upgrade
//   tg demo seed --n 500 --password "$E2E_PASSWORD"
// Existing servers on :8000 / :3000 are reused; otherwise Playwright starts them.
// The API command runs with cwd = repo root, so the venv path is relative to the root.
const python = path.join(".venv", process.platform === "win32" ? "Scripts" : "bin", "uvicorn");

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"], ["html", { open: "never" }]],
  use: { baseURL: "http://localhost:3000", trace: "retain-on-failure", ...devices["Desktop Chrome"] },
  webServer: [
    {
      command: `${python} transactionguard.api.main:app --port 8000`,
      cwd: "..",
      url: "http://127.0.0.1:8000/api/v1/health",
      reuseExistingServer: true,
      timeout: 120_000,
    },
    { command: "npm run start", url: "http://localhost:3000/login", reuseExistingServer: true, timeout: 120_000 },
  ],
});
