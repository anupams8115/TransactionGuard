import { expect, test } from "@playwright/test";

const EMAIL = process.env.E2E_EMAIL ?? "analyst@demo.example";
const PASSWORD = process.env.E2E_PASSWORD ?? "demo-password-2026";

test("analyst scores a transaction, reads the explanation and resolves the review", async ({ page }) => {
  // 1. Sign in (cookie session via the Next.js proxy)
  await page.goto("/login");
  await page.getByLabel("Email").fill(EMAIL);
  await page.getByLabel("Password").fill(PASSWORD);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("heading", { name: "Overview" })).toBeVisible();

  // 2. Score a synthetic transaction built from a stored example (fresh id)
  await page.getByRole("link", { name: "Score transaction" }).click();
  await page.getByRole("button", { name: "Fill with an example" }).click();
  await expect(page.locator("#f-transaction_id")).toHaveValue(/-copy-/);
  await page.locator("form").getByRole("button", { name: "Score transaction" }).click();

  // 3. The result shows a score, the threshold, and a real SHAP explanation with its scale
  await expect(page.getByText("Threshold", { exact: true })).toBeVisible();
  await expect(page.getByText(/calibrated log-odds/).first()).toBeVisible();
  await expect(page.getByRole("list", { name: "Feature contributions" }).getByRole("listitem").first()).toBeVisible();
  await expect(page.getByText(/not causes and do not prove fraud/)).toBeVisible();

  // 4. Open the full trace: model version, preprocessing and bundle checksum are recorded
  await page.getByRole("link", { name: "Open full trace and review" }).click();
  await expect(page.getByRole("heading", { name: "Trace" })).toBeVisible();
  await expect(page.getByText("bundle_sha256")).toBeVisible();
  await expect(page.getByText("preprocessing_version")).toBeVisible();

  // 5. Review: resolving without an assessment is refused; with one it is recorded in history
  await page.getByLabel("Status").selectOption("resolved");
  await page.getByRole("button", { name: "Save review" }).click();
  await expect(page.getByText(/requires an assessment/)).toBeVisible();
  await page.getByLabel(/Assessment/).selectOption("inconclusive");
  await page.getByLabel("Note").fill("E2E check: pattern resembles normal travel spend.");
  await page.getByRole("button", { name: "Save review" }).click();
  await expect(page.getByText("Saved", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Review history" }).locator("..").locator("..")).toContainText("resolved");
});

test("unauthenticated visitors are sent to sign-in", async ({ page }) => {
  await page.goto("/transactions");
  await expect(page).toHaveURL(/\/login/);
});
