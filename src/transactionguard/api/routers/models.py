"""Model metadata, data contracts and offline evaluation reports."""

from __future__ import annotations

import json
import re
from typing import Any

from fastapi import APIRouter, HTTPException, status

from transactionguard.api.deps import DB, Admin, CurrentUser
from transactionguard.core.config import get_settings
from transactionguard.data.contract import CONTRACTS, get_contract
from transactionguard.features.engineering import FEATURE_SPECS
from transactionguard.inference.bundle import SCORE_MEANINGS
from transactionguard.inference.service import MODEL_STORE, sync_model_versions

router = APIRouter(tags=["models"])
_SAFE_NAME = re.compile(r"^[a-z]+-\d{8}T\d{6}Z$")


@router.get("/contracts")
def list_contracts(_: CurrentUser) -> list[dict[str, Any]]:
    return [_contract_payload(name) for name in CONTRACTS]


@router.get("/contracts/{name}")
def contract_detail(name: str, _: CurrentUser) -> dict[str, Any]:
    if name not in CONTRACTS:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown contract")
    return _contract_payload(name)


def _contract_payload(name: str) -> dict[str, Any]:
    """Everything the frontend needs to render a form for exactly the fields the model uses."""
    c = get_contract(name)
    spec = FEATURE_SPECS[name]
    d = c.to_dict()
    d["input_fields"] = [f for f in d["fields"] if f["role"] != "target"]
    d["model_features"] = [{"name": f, "description": spec.descriptions.get(f, f)} for f in spec.all]
    d["active_model_version"] = MODEL_STORE.registry.active_version(name)
    return d


@router.get("/models")
def list_models(_: CurrentUser) -> list[dict[str, Any]]:
    reg = MODEL_STORE.registry.read()
    active = set(reg.get("active", {}).values())
    out = []
    for version, entry in sorted(
        reg.get("bundles", {}).items(), key=lambda kv: kv[1]["saved_at"], reverse=True
    ):
        out.append(
            {
                "model_version": version,
                "is_active": version in active,
                "score_meaning": SCORE_MEANINGS[entry["score_kind"]],
                **{k: entry.get(k) for k in _MODEL_FIELDS},
            }
        )
    return out


_MODEL_FIELDS = (
    "model_name",
    "contract",
    "score_kind",
    "threshold",
    "threshold_policy",
    "preprocessing_version",
    "calibration",
    "training_session",
    "dataset_fingerprint",
    "dataset_provenance",
    "mlflow_run_id",
    "mlflow_parent_run_id",
    "saved_at",
    "sha256",
    "test_metrics",
    "code_revision",
    "selected",
)


@router.get("/models/{model_version}")
def model_detail(model_version: str, _: CurrentUser) -> dict[str, Any]:
    reg = MODEL_STORE.registry.read()
    if model_version not in reg.get("bundles", {}):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown model version")
    bundle = MODEL_STORE.get(model_version)
    summary = bundle.summary()
    summary["global_importance"] = bundle.metadata.get("global_importance", [])
    summary["params"] = bundle.params
    summary["is_active"] = reg["active"].get(bundle.contract_name) == model_version
    summary["sha256"] = reg["bundles"][model_version]["sha256"]
    summary["mlflow_run_id"] = reg["bundles"][model_version].get("mlflow_run_id")
    return summary


@router.post("/models/{model_version}/activate")
def activate_model(model_version: str, _: Admin, db: DB) -> dict[str, Any]:
    """Switch the serving model (admin only). Only versions in the trusted registry are allowed."""
    try:
        MODEL_STORE.registry.activate(model_version)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown model version") from exc
    sync_model_versions(db)
    return {"active": MODEL_STORE.registry.read()["active"]}


@router.get("/evaluations")
def list_evaluations(_: CurrentUser) -> list[dict[str, Any]]:
    folder = get_settings().artifacts_dir / "evaluations"
    out = []
    for path in sorted(folder.glob("*.json"), reverse=True):
        r = json.loads(path.read_text())
        out.append(
            {
                "id": path.stem,
                "dataset": r["dataset"],
                "synthetic": r["synthetic"],
                "session_id": r["session_id"],
                "selected": r["selection"]["selected"],
                "created_at": r["created_at"],
                "test_ap": {x["model"]: x["test"]["average_precision"] for x in r["results"]},
            }
        )
    return out


@router.get("/evaluations/{evaluation_id}")
def evaluation_detail(evaluation_id: str, _: CurrentUser) -> dict[str, Any]:
    if not _SAFE_NAME.match(evaluation_id):  # no path traversal through the id
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown evaluation")
    path = get_settings().artifacts_dir / "evaluations" / f"{evaluation_id}.json"
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown evaluation")
    return json.loads(path.read_text())
