"""Shared fixtures.

Every test runs against an isolated temporary data/artifact directory and SQLite database, so
tests never touch your real datasets, bundles or MLflow store.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from transactionguard.core.config import get_settings
from transactionguard.data.synthetic import SyntheticConfig, generate

SMALL = SyntheticConfig(n_accounts=300, days=30, fraud_account_rate=0.35, seed=7)


@pytest.fixture(scope="session")
def small_synthetic() -> pd.DataFrame:
    """~13k rows with plenty of positives in every chronological split."""
    return generate(SMALL)


def _env_for(root: Path) -> dict[str, str]:
    return {
        "TG_ENVIRONMENT": "test",
        "TG_DATA_DIR": str(root / "data"),
        "TG_ARTIFACTS_DIR": str(root / "artifacts"),
        "TG_REPORTS_DIR": str(root / "reports"),
        "TG_DATABASE_URL": f"sqlite:///{(root / 'test.db').as_posix()}",
        "TG_MLFLOW_TRACKING_URI": f"sqlite:///{(root / 'mlflow.db').as_posix()}",
        "TG_JWT_SECRET": "test-secret-with-enough-length-for-hs256-0123456789",
        "TG_CELERY_EAGER": "true",
    }


@pytest.fixture
def isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for k, v in _env_for(tmp_path).items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture
def prepared_small(isolated_settings, small_synthetic):
    """Write the small synthetic dataset where ``sources.load('synthetic')`` expects it."""
    from transactionguard.data import sources

    sources._validate_and_write(small_synthetic, sources.SYNTHETIC, sources.processed_path("synthetic"))
    return isolated_settings


@pytest.fixture(scope="session")
def _trained_template(tmp_path_factory: pytest.TempPathFactory, small_synthetic):
    """Train ONCE per test session (smoke budget, no MLflow) into a template directory."""
    from transactionguard.data import sources
    from transactionguard.training.splits import SplitConfig
    from transactionguard.training.train import TrainConfig, train

    root = tmp_path_factory.mktemp("trained")
    mp = pytest.MonkeyPatch()
    for k, v in _env_for(root).items():
        mp.setenv(k, v)
    get_settings.cache_clear()
    try:
        sources._validate_and_write(small_synthetic, sources.SYNTHETIC, sources.processed_path("synthetic"))
        report = train(
            TrainConfig(
                dataset="synthetic",
                budget="smoke",
                log_mlflow=False,
                n_bootstrap=50,
                split=SplitConfig(seed=7),
                seed=7,
            )
        )
    finally:
        mp.undo()
        get_settings.cache_clear()
    return root, report


@pytest.fixture
def trained_small(_trained_template, isolated_settings, tmp_path: Path):
    """Per-test copy of the trained artifacts, so tests may tamper with files safely."""
    root, report = _trained_template
    for sub in ("data", "artifacts"):
        shutil.copytree(root / sub, tmp_path / sub, dirs_exist_ok=True)
    return report
