# Benchmarks (measured, not projected)

All numbers were measured on **one laptop** on 2026-10-14. They describe this setup only. No
production scale is claimed.

| | |
|---|---|
| CPU | Intel Core i7-12700H (14 cores / 20 threads) |
| RAM | 15.6 GB |
| OS | Windows 11 Home (10.0.26200) |
| Python / key libs | 3.12.15 · XGBoost 3.4.1 · scikit-learn 1.9.1 · SHAP 0.52 · FastAPI 0.142 |
| Server | `uvicorn`, **1 process**, SQLite database (the local default), Celery eager mode |

## 1. In-process model scoring (no HTTP, no database, no SHAP)

Measured during training on the test split (`training/train.py::_latency`): feature building,
preprocessing, model and calibration.

| Model (creditcard) | Single row p50 | Single row p95 | Batch throughput |
|---|---|---|---|
| Logistic Regression | 11.1 ms | 12.5 ms | ~854k rows/s (56,962-row batch) |
| XGBoost | 14.3 ms | 19.7 ms | ~620k rows/s |
| Isolation Forest | 28.9 ms | 32.4 ms | ~33k rows/s |

Single-row latency is dominated by pandas and sklearn overhead per call, not by the model. Batch
scoring amortises it.

## 2. End-to-end API (HTTP → validation → DB → features → model → DB insert)

`scripts/benchmark_api.py`. Each request scores a **new** transaction id. Python `httpx`
client on the same machine, after one warm-up request.

| Dataset | SHAP inline | Concurrency | Requests | p50 | p95 | p99 | Throughput |
|---|---|---|---|---|---|---|---|
| creditcard | no | 1 | 299 | 40.7 ms | 50.3 ms | 55.3 ms | 24.8 req/s |
| creditcard | no | 4 | 299 | 60.2 ms | 629 ms | 2,482 ms | 23.2 req/s |
| creditcard | yes | 1 | 199 | 56.8 ms | 71.8 ms | 74.3 ms | 17.3 req/s |
| synthetic (with DB history lookup) | no | 1 | 199 | 63.8 ms | 77.6 ms | 90.7 ms | 15.5 req/s |
| synthetic (with DB history lookup) | no | 4 | 199 | 59.1 ms | 1,227 ms | 2,845 ms | 17.3 req/s |

**What these show:**
- Inline SHAP explanations add about 16 ms at p50 for XGBoost.
- Account-history lookup (synthetic) adds about 23 ms at p50 over creditcard.
- **Concurrency does not increase throughput here.** SQLite allows one writer at a time and there
  is a single Python process, so concurrent requests queue (see the p95/p99 jump). The Docker
  Compose setup (PostgreSQL, two uvicorn workers) is designed to remove that bottleneck, but **it
  has not been measured**, so no number is claimed.

## 3. Batch and seeding

| Operation | Rows | Time | Notes |
|---|---|---|---|
| `tg demo seed --contract creditcard` | 5,000 | 33 s | 500-row chunks, inline SHAP for every row, SQLite |
| Training `tg train creditcard --budget small` | 284,807 | 157 s | 4 models, 15 Optuna trials each, SHAP global summaries, MLflow |
| Synthetic generation | 143,497 | ~9 s | deterministic, seed 42 |
| Feature building (history features) | 143,497 | 0.3 s | vectorised `searchsorted` |

An earlier version of seeding took 22 minutes for 20,000 rows. It kept one ORM session across all
chunks (quadratic flush cost) and wrapped each row in a SAVEPOINT. One transaction per chunk plus
a bulk insert with a row-by-row fallback only on conflict fixed it.

## Reproduce

```bash
uv run uvicorn transactionguard.api.main:app --port 8000        # terminal 1
uv run python scripts/benchmark_api.py --contract creditcard --n 300 --concurrency 1 4
uv run python scripts/benchmark_api.py --contract creditcard --n 200 --concurrency 1 --explain
uv run python scripts/benchmark_api.py --contract synthetic --n 200 --concurrency 1 4
```
