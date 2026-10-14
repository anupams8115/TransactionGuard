# TransactionGuard

An explainable transaction-risk system. It scores card transactions, explains every score with
SHAP, routes high-risk ones to an analyst review queue, and monitors data drift.

> **Scores are risk signals for analyst review, not proof of fraud.** This is a portfolio project.
> It has not been deployed to production, and its scale has only been measured on one laptop
> ([benchmarks](docs/benchmarks.md)).

| | |
|---|---|
| ML | pandas · NumPy · scikit-learn · XGBoost · Isolation Forest · Optuna · SHAP · MLflow · Evidently |
| Backend | FastAPI · Pydantic v2 · SQLAlchemy 2.0 · Alembic · PostgreSQL (SQLite locally) · Celery · Redis |
| Frontend | Next.js 16 (App Router) · React 19 · TypeScript · Tailwind CSS 4 · TanStack Query · Recharts · Zod |
| Delivery | uv · Docker Compose · GitHub Actions · pytest · Playwright |

## What an analyst can do

1. Score one transaction. The form is generated from the model's data contract.
2. Upload a CSV for batch scoring and watch it progress (Celery).
3. See the risk score, the threshold it was compared with, and a SHAP explanation.
4. Work a review queue (unreviewed → under review → resolved, with a required assessment).
5. Compare trained models on an untouched test set, with bootstrap confidence intervals.
6. Generate data-quality and drift reports (Evidently).
7. Trace any prediction to its model version, preprocessing version, threshold policy,
   calibration, training run (MLflow), dataset fingerprint and bundle checksum.

## Results (real data)

Public ULB credit-card dataset (real transactions, PCA-anonymized), chronological split. The test
set is the last 20% of time: 56,962 transactions, 75 frauds (0.13%).

| Model | Test average precision (95% CI) | Precision | Recall | False-positive rate |
|---|---|---|---|---|
| Dummy (prevalence) | 0.001 | n/a | n/a | n/a |
| Logistic Regression | 0.748 (0.643–0.861) | 0.740 | 0.720 | 0.00033 |
| **XGBoost (selected on validation)** | **0.789** (0.692–0.872) | 0.740 | 0.760 | 0.00035 |
| Isolation Forest (unsupervised) | 0.073 (0.055–0.102) | 0.116 | 0.387 | 0.00388 |

The paired bootstrap interval for XGBoost minus Logistic Regression is −0.028 to +0.113. It
includes zero, so the test set does not show that XGBoost is better. Details are in the
[model card](docs/model_card.md).

## Quick start (no Docker needed)

Requirements: Python 3.12 via [uv](https://docs.astral.sh/uv/), Node.js 24. CPU only; no GPU,
paid API or cloud account.

**Windows PowerShell**

```powershell
cd TransactionGuard
py -m pip install --user "uv==0.12.19"          # once
uv sync                                          # Python deps (locked)
Copy-Item .env.example .env                      # then set TG_JWT_SECRET in .env
uv run tg data prepare synthetic                 # ~10 s, generated locally
uv run tg data prepare creditcard                # ~150 MB download from OpenML, MD5-verified
uv run tg train synthetic --budget small         # several minutes (history features)
uv run tg train creditcard --budget small        # ~3 min on an i7-12700H
uv run tg db upgrade                             # create tables (SQLite file)
uv run tg demo seed --contract synthetic --n 8000 --password "choose-a-password"
uv run tg demo seed --contract creditcard --n 20000
uv run uvicorn transactionguard.api.main:app --port 8000      # terminal 1
cd frontend; npm ci; npm run build; npm run start             # terminal 2
```

**macOS / Linux**

```bash
cd TransactionGuard
curl -LsSf https://astral.sh/uv/0.12.19/install.sh | sh        # once
uv sync
cp .env.example .env                                           # then set TG_JWT_SECRET
uv run tg data prepare synthetic
uv run tg data prepare creditcard
uv run tg train synthetic --budget small
uv run tg train creditcard --budget small
uv run tg db upgrade
uv run tg demo seed --contract synthetic --n 8000 --password "choose-a-password"
uv run tg demo seed --contract creditcard --n 20000
uv run uvicorn transactionguard.api.main:app --port 8000 &
(cd frontend && npm ci && npm run build && npm run start)
```

Open http://localhost:3000 and sign in as `admin@demo.example`, `analyst@demo.example` or
`viewer@demo.example` with the password you chose. The API docs are at http://localhost:8000/docs.

### Other commands (identical on both platforms)

| Task | Command |
|---|---|
| Run all Python tests | `uv run pytest` |
| Lint / format check | `uv run ruff check src tests` · `uv run ruff format --check src tests` |
| Frontend checks | `cd frontend` then `npm run lint`, `npm run typecheck`, `npm run build` |
| End-to-end browser test | `cd frontend` then `npx playwright install chromium` (once), then `npx playwright test` (set `E2E_PASSWORD`) |
| View MLflow experiments | `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db` → http://127.0.0.1:5000 |
| Monitoring report | `uv run tg monitor report --contract creditcard --days 7` |
| API latency benchmark | `uv run python scripts/benchmark_api.py --contract creditcard --n 300 --concurrency 1 4` |
| Validate a CSV | `uv run tg data validate my.csv --contract synthetic` |
| Create a user | `uv run tg users create --email you@bank.com --role analyst --workspace demo` |
| Real async worker (needs Redis) | set `TG_CELERY_EAGER=false`, then `uv run celery -A transactionguard.worker.celery_app worker -P solo -l info` |

### Docker Compose (PostgreSQL + Redis + worker)

```bash
cp .env.example .env        # set TG_JWT_SECRET and POSTGRES_PASSWORD
docker compose up --build -d
docker compose run --rm api tg data prepare synthetic
docker compose run --rm api tg train synthetic --budget small
docker compose run --rm api tg demo seed --password "choose-a-password"
# optional: docker compose --profile mlflow up -d mlflow   ->  http://localhost:5000
```

CI builds the Compose images. The full stack has not been run on the author's machine, which has
no Docker.

## Documentation

- [Architecture](docs/architecture.md): components, leakage protections, score semantics
- [Dataset provenance and data dictionary](docs/dataset.md)
- [Model card](docs/model_card.md): intended use, results, limitations
- [Benchmarks](docs/benchmarks.md): measured latency and throughput, with hardware details
- [Testing](docs/testing.md): what is tested, what is mocked, known gaps
- [Demo script](docs/demo_script.md): a 5-minute walkthrough
- [Interview guide](docs/interview_guide.md): questions and answers about the design
- [Project state](PROJECT_STATE.md): milestones, limitations, next steps

## Repository layout

```
src/transactionguard/
  core/        settings (pydantic-settings), structured logging (structlog)
  data/        data contract, validation, OpenML download, synthetic generator
  features/    leakage-safe feature engineering + fitted sklearn preprocessing
  training/    splits, models, Optuna, evaluation/thresholds/calibration, MLflow
  inference/   model bundle + trusted registry, SHAP explanations, scoring service
  monitoring/  Evidently reports, request metrics
  api/         FastAPI app, routers, schemas, auth
  db/          SQLAlchemy models + sessions        (migrations/ = Alembic)
  worker/      Celery app + tasks
frontend/      Next.js dashboard (+ e2e/ Playwright)
tests/         unit/ (no services) · integration/ (API + DB) · real-service tests (CI)
docs/          documentation
```

## License

Code: MIT. The credit-card dataset is **not** included or redistributed. It is downloaded from
OpenML under its own licence (see [docs/dataset.md](docs/dataset.md)).
