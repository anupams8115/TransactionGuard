# Testing

Last local run (2026-10-14, Windows 11):

| Suite | Result | Runtime |
|---|---|---|
| `uv run pytest` (unit + integration) | **95 passed, 2 skipped** (the skipped tests need real PostgreSQL / Redis) | 2.5 min |
| `npx playwright test` (browser end-to-end) | **2 passed** | 11 s |
| `ruff check` / `ruff format --check` | clean | |
| `npm run lint` / `npm run typecheck` / `npm run build` | clean | |

In CI (`.github/workflows/ci.yml`) the two skipped tests run against PostgreSQL 17 and Redis 7.4
service containers. **CI has not run yet**, because the repository has not been pushed.

## What runs against what

| Layer | Real | Replaced |
|---|---|---|
| Unit (`tests/unit`) | pandas, sklearn, XGBoost, SHAP, real training (tiny data) | nothing (no network, no services) |
| Integration (`tests/integration`) | FastAPI app, HTTP via TestClient, SQLite migrated by **Alembic**, real model bundles, real Evidently | Celery broker → **eager mode** (in-process), so this is a mocked broker |
| Real services (`test_real_services.py`, `test_migrations.py::postgres`) | **Redis broker + a real Celery worker thread**, **PostgreSQL** migrations | nothing; skipped unless `TG_TEST_REDIS_URL` / `TG_TEST_DATABASE_URL` are set |
| End-to-end (`frontend/e2e`) | Chromium → Next.js → FastAPI → DB → trained model | nothing |

## Coverage of the required test areas

| Area | Tests |
|---|---|
| Input validation | `test_contract_validation.py` (types, missing, NaN/inf, duplicates, categories, ranges, future timestamps, strict request model); `test_api_scoring.py::test_invalid_transactions_rejected`, `test_missing_field_is_rejected_not_defaulted` |
| Leakage prevention | `test_features_leakage.py` (future rows change nothing, labels don't affect features, strictly-past history, same-timestamp rows); `test_training_and_bundles.py::test_test_labels_cannot_change_any_decision` |
| Preprocessing consistency | single-row equals batch transform; training features equal online features rebuilt from DB history; bundle round-trip reproduces scores |
| Unseen categories | `test_unseen_category_is_handled_and_detectable` (all-zeros encoding, reported as a warning) |
| Model loading | tampered checksum refused, unregistered version refused, path escape refused |
| Threshold behaviour | `>=` rule, min-precision, capacity, cost policies, infeasible fallback is reported |
| Explanation output | additivity for XGBoost / LogReg / IForest, one-hot grouping, calibrated scale, explainer failure is reported (never invented) |
| AuthN / AuthZ | Argon2 hashes, forged / unsigned / expired JWT rejected, role matrix, CSRF header for cookie writes, workspace isolation, job ownership |
| Duplicate submissions | same payload → same prediction (200); changed payload → 409; batch Idempotency-Key; re-uploaded rows reuse predictions; race on the unique constraint |
| Batch retries | crash mid-job → automatic retry with no duplicates; bounded retries → visible `failed`; manual retry; non-transient errors fail fast |
| Migrations | upgrade → downgrade → upgrade, and ORM-vs-migration diff is empty (SQLite here; PostgreSQL in CI) |
| Failed dependencies | DB down → `/ready` 503; no active model → 503 on scoring and readiness |
| Frontend → backend workflow | Playwright: sign in → score → explanation → trace → review rules → resolve |
| Monitoring | report built with drift + quality + score stats; performance withheld without labels; drift statistics read in the right direction |

## Known gaps

- No load test against PostgreSQL + multiple workers (see [benchmarks](benchmarks.md)).
- No property-based or fuzz testing of the CSV parser.
- The frontend has no component-level unit tests; only the end-to-end flow is tested.
- No test of token revocation (tokens are valid until expiry; there is no deny-list).
- Docker images are built in CI but the composed stack is not exercised end-to-end there.
- No rate limiting, so it is not tested either.
