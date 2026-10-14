"""`tg` command-line interface. Run `uv run tg --help`."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from transactionguard.core.config import get_settings
from transactionguard.core.logging import configure_logging

app = typer.Typer(no_args_is_help=True, help="TransactionGuard: explainable transaction-risk signals.")
data_app = typer.Typer(no_args_is_help=True, help="Dataset preparation and validation.")
app.add_typer(data_app, name="data")


@app.callback()
def _setup() -> None:
    s = get_settings()
    configure_logging(s.log_level, s.log_json)


db_app = typer.Typer(no_args_is_help=True, help="Database migrations.")
users_app = typer.Typer(no_args_is_help=True, help="User management.")
demo_app = typer.Typer(no_args_is_help=True, help="Demo data.")
app.add_typer(db_app, name="db")
app.add_typer(users_app, name="users")
app.add_typer(demo_app, name="demo")


@db_app.command("upgrade")
def db_upgrade(revision: Annotated[str, typer.Argument()] = "head") -> None:
    """Apply Alembic migrations (creates/updates tables)."""
    from transactionguard.admin import upgrade_db

    upgrade_db(revision)
    typer.echo(f"Database at revision {revision}")


@users_app.command("create")
def users_create(
    email: Annotated[str, typer.Option()],
    role: Annotated[str, typer.Option(help="admin | analyst | viewer")] = "analyst",
    workspace: Annotated[str, typer.Option()] = "default",
    password: Annotated[str, typer.Option(prompt=True, hide_input=True, confirmation_prompt=True)] = "",
) -> None:
    """Create a user (password is prompted, never echoed)."""
    from transactionguard.admin import create_user
    from transactionguard.db.models import Role
    from transactionguard.db.session import session_scope

    with session_scope() as db:
        user = create_user(db, email=email, password=password, role=Role(role), workspace=workspace)
        typer.echo(f"Created {user.email} ({user.role.value}) in workspace '{workspace}'")


@demo_app.command("seed")
def demo_seed(
    contract: Annotated[str, typer.Option(help="synthetic | creditcard")] = "synthetic",
    workspace: Annotated[str, typer.Option()] = "demo",
    n: Annotated[int, typer.Option(help="Transactions to score")] = 2000,
    password: Annotated[str | None, typer.Option(help="Use this password for new demo users")] = None,
    no_labels: Annotated[bool, typer.Option(help="Do not replay dataset labels")] = False,
) -> None:
    """Create demo users and score recent transactions so every page has data."""
    from transactionguard.admin import seed_demo

    out = seed_demo(
        workspace=workspace, contract_name=contract, n_score=n, password=password, replay_labels=not no_labels
    )
    creds = out.pop("new_credentials")
    typer.echo(json.dumps(out, indent=2))
    for email, pw in creds.items():
        typer.echo(f"  login: {email}  password: {pw}")
    if not creds:
        typer.echo("  (demo users already existed; passwords unchanged)")


monitor_app = typer.Typer(no_args_is_help=True, help="Monitoring reports (Evidently).")
app.add_typer(monitor_app, name="monitor")


@monitor_app.command("report")
def monitor_report(
    contract: Annotated[str, typer.Option(help="synthetic | creditcard")] = "synthetic",
    workspace: Annotated[str, typer.Option()] = "demo",
    days: Annotated[int, typer.Option(help="Monitoring window in days")] = 7,
) -> None:
    """Build a drift/data-quality report for a workspace's recent predictions (runs synchronously)."""
    from sqlalchemy import select

    from transactionguard.db.models import MonitoringReport, Role, User, Workspace
    from transactionguard.db.session import session_scope
    from transactionguard.inference.service import MODEL_STORE, sync_model_versions
    from transactionguard.monitoring.drift import run_monitoring_report

    with session_scope() as db:
        sync_model_versions(db)
        ws = db.scalars(select(Workspace).where(Workspace.name == workspace)).one()
        admin = db.scalars(select(User).where(User.workspace_id == ws.id, User.role == Role.admin)).first()
        if admin is None:
            raise typer.BadParameter(f"workspace '{workspace}' has no admin user")
        report = MonitoringReport(
            workspace_id=ws.id,
            requested_by=admin.id,
            contract=contract,
            model_version=MODEL_STORE.active(contract).model_version,
            summary={"requested_window_days": days},
        )
        db.add(report)
        db.flush()
        report_id = report.id
    status = run_monitoring_report(report_id)
    with session_scope() as db:
        r = db.get(MonitoringReport, report_id)
        assert r is not None
        s = r.summary or {}
        typer.echo(f"status={status} report={report_id}")
        typer.echo(
            f"  result: {s.get('status')}  window rows={s.get('window', {}).get('rows')}  "
            f"reference rows={s.get('reference', {}).get('rows')}"
        )
        typer.echo(f"  drifted columns: {s.get('drift', {}).get('drifted_columns')}")
        typer.echo(f"  performance available: {s.get('performance', {}).get('available')}")
        typer.echo(f"  HTML: {r.html_path or r.error}")


@data_app.command("prepare")
def data_prepare(
    dataset: Annotated[str, typer.Argument(help="creditcard | synthetic")],
    force: Annotated[bool, typer.Option(help="Rebuild even if already prepared")] = False,
    accounts: Annotated[int, typer.Option(help="Synthetic only: number of accounts")] = 1500,
    days: Annotated[int, typer.Option(help="Synthetic only: days of history")] = 60,
) -> None:
    """Download (creditcard) or generate (synthetic), validate, and save as parquet."""
    from transactionguard.data import sources
    from transactionguard.data.synthetic import SyntheticConfig

    if dataset == "creditcard":
        path = sources.prepare_creditcard(force=force)
    elif dataset == "synthetic":
        cfg = SyntheticConfig(n_accounts=accounts, days=days, seed=get_settings().random_seed)
        path = sources.prepare_synthetic(cfg, force=force)
    else:
        raise typer.BadParameter("dataset must be 'creditcard' or 'synthetic'")
    typer.echo(path.with_suffix(".summary.json").read_text())


@data_app.command("validate")
def data_validate(
    csv_path: Annotated[str, typer.Argument(help="CSV file to validate")],
    contract: Annotated[str, typer.Option(help="creditcard | synthetic")] = "synthetic",
    with_target: Annotated[bool, typer.Option(help="Require the label column")] = False,
) -> None:
    """Validate a CSV against a data contract and print the report."""
    import pandas as pd

    from transactionguard.data.contract import get_contract
    from transactionguard.data.validation import validate_frame

    _, report = validate_frame(pd.read_csv(csv_path), get_contract(contract), require_target=with_target)
    typer.echo(json.dumps(report.to_dict(), indent=2))
    raise typer.Exit(code=0 if report.ok else 1)


@app.command("train")
def train_cmd(
    dataset: Annotated[str, typer.Argument(help="creditcard | synthetic")],
    budget: Annotated[str, typer.Option(help="Optuna budget: smoke | small | medium")] = "small",
    split: Annotated[str, typer.Option(help="chronological | group")] = "chronological",
    policy: Annotated[str, typer.Option(help="min_precision | review_capacity | cost")] = "min_precision",
    min_precision: Annotated[float, typer.Option(help="For policy=min_precision")] = 0.5,
    alerts_per_day: Annotated[float, typer.Option(help="For policy=review_capacity")] = 50.0,
    fp_cost: Annotated[float, typer.Option(help="For policy=cost")] = 1.0,
    fn_cost: Annotated[float, typer.Option(help="For policy=cost")] = 25.0,
    models: Annotated[
        str, typer.Option(help="Comma-separated subset")
    ] = "dummy,logreg,xgboost,isolation_forest",
    iforest_data: Annotated[str, typer.Option(help="mixed | normal_only")] = "mixed",
    no_activate: Annotated[bool, typer.Option(help="Do not make the selected model active")] = False,
    no_mlflow: Annotated[bool, typer.Option(help="Skip MLflow logging")] = False,
) -> None:
    """Train, tune, calibrate, select on validation; report on the untouched test set."""
    from transactionguard.training.evaluation import ThresholdPolicy
    from transactionguard.training.splits import SplitConfig
    from transactionguard.training.train import TrainConfig, train

    seed = get_settings().random_seed
    cfg = TrainConfig(
        dataset=dataset,
        models=tuple(m.strip() for m in models.split(",")),
        budget=budget,
        split=SplitConfig(strategy=split, seed=seed),  # type: ignore[arg-type]
        policy=ThresholdPolicy(
            name=policy,
            min_precision=min_precision,  # type: ignore[arg-type]
            alerts_per_day=alerts_per_day,
            fp_cost=fp_cost,
            fn_cost=fn_cost,
        ),
        iforest_training_data=iforest_data,
        seed=seed,
        activate=not no_activate,
        log_mlflow=not no_mlflow,
    )
    report = train(cfg)
    _print_results(report)


def _print_results(report: dict) -> None:
    label = "SYNTHETIC DATA - not benchmark evidence" if report["synthetic"] else "benchmark data"
    typer.echo(f"\nSession {report['session_id']} on {report['dataset']} ({label})")
    sp = report["split"]
    for part in ("train", "valid", "test"):
        typer.echo(
            f"  {part:<5} rows={sp[part]['rows']:>7}  positives={sp[part]['positives']:>4}  "
            f"prevalence={sp[part]['prevalence']:.4%}"
        )
    typer.echo(
        f"\n  {'model':<17}{'score':<24}{'val AP':>8}{'test AP':>9}{'prec':>7}{'recall':>8}"
        f"{'FPR':>9}{'ROC-AUC':>9}{'p50 ms':>8}"
    )
    for r in report["results"]:
        t, v, lat = r["test"], r["validation"], r["latency"] or {}
        mark = " *" if r["selected"] else ""
        typer.echo(
            f"  {r['model'] + mark:<17}{r['score_kind']:<24}{v['average_precision']:>8.3f}"
            f"{t['average_precision']:>9.3f}{t['precision']:>7.3f}{t['recall']:>8.3f}"
            f"{t['false_positive_rate']:>9.5f}{t['roc_auc']:>9.3f}"
            f"{lat.get('single_row_ms_p50', float('nan')):>8.2f}"
        )
    typer.echo(f"\n  * selected: {report['selection']['rule']}")
    typer.echo(f"  report: {report.get('report_path')}")


if __name__ == "__main__":
    app()
