# PROJECT_STATE

_Last updated: 2026-10-14_

## Milestones

| # | Milestone | Status | Evidence |
|---|---|---|---|
| 1 | Foundation: layout, uv lockfile, typed settings, structlog, `.env.example`, architecture | ✅ | `docs/architecture.md` |
| 2 | Dataset + data contract: OpenML 1597 (MD5-verified), synthetic generator, validation | ✅ | `docs/dataset.md`, `tests/unit/test_contract_validation.py` |
| 3 | Leakage-safe features: past-only history, train-only preprocessing | ✅ | `tests/unit/test_features_leakage.py` |
| 4 | Training: dummy, LogReg, XGBoost, Isolation Forest, bounded Optuna | ✅ | `tg train …` |
| 5 | Evaluation: AP + full metrics, bootstrap CIs, Platt calibration, validation-only thresholds, MLflow | ✅ | `docs/model_card.md` |
| 6 | SHAP: local + global, grouped one-hot, calibrated scale, additivity check | ✅ | `inference/explain.py` |
| 7 | Backend: FastAPI, SQLAlchemy 2, Alembic, JWT/Argon2/RBAC, idempotency, Celery | ✅ | 36+ integration tests |
| 8 | Dashboard: Next.js 16 / TypeScript / Tailwind 4, all 7 pages | ✅ | Playwright e2e |
| 9 | Monitoring: Evidently drift + quality, labels-gated performance, request metrics | ✅ | `tg monitor report` |
| 10 | Security + testing | ✅ | `docs/testing.md` (95 passed, 2 real-service tests skipped locally) |
| 11 | Docker Compose, GitHub Actions, measured benchmarks | ✅ written · ⚠️ Compose not run locally (no Docker); CI not yet run (repo not pushed) | `docs/benchmarks.md` |
| opt | Kafka streaming | ⏳ not started | |

## Commands verified on this machine (Windows 11, i7-12700H)

```bash
uv sync
uv run tg data prepare synthetic        # 9 s
uv run tg data prepare creditcard       # ~4 min download + MD5 check
uv run tg train creditcard --budget small   # 157 s
uv run tg train synthetic --budget small
uv run tg db upgrade
uv run tg demo seed --contract synthetic --n 8000 --password "…"
uv run tg demo seed --contract creditcard --n 20000
uv run tg monitor report --contract creditcard --days 7
uv run pytest                           # 95 passed, 2 skipped
uv run uvicorn transactionguard.api.main:app --port 8000
cd frontend && npm ci && npm run build && npm run start && npx playwright test   # 2 passed
```

## Current active models

- `creditcard-xgboost-20261014T115618Z`: test AP 0.789 [0.692, 0.872], threshold 0.0860
- `synthetic-xgboost-20261014T114016Z`: synthetic, demo only

## Known limitations

- Validation has 57 frauds and test has 75, so thresholds, calibration and model ranking are
  noisy. XGBoost versus LogReg is not statistically separated.
- `acct_prior_txn_count` (synthetic) is cumulative and drifts by construction. It should be windowed.
- SQLite limits concurrent writes; PostgreSQL throughput has not been measured.
- Single-row in-process latency (11–30 ms) is dominated by pandas overhead.
- Seeding/scoring history lookups reload account history per chunk from the DB (fine at demo
  scale; would need a feature store at real scale).
- JWTs cannot be revoked before expiry, and there is no rate limiting.
- Docker Compose is written and builds in CI, but the full stack has not been run end-to-end.
- `next start` warns because `output: "standalone"` is set for Docker. Locally it still serves
  correctly; in Docker the image runs `node server.js`.

## Next steps

1. Push to GitHub and confirm CI is green (real PostgreSQL + Redis tests, e2e, Docker build).
2. Optional Kafka ingestion profile (schema validation, idempotent consumer, DLQ, replay).
3. Replace cumulative history features with windowed ones; retrain and compare.
4. Measure throughput on Docker Compose (PostgreSQL, 2 API workers, Celery worker).
