"""Input validation: data contract, batch validation report and the strict request model."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from transactionguard.data.contract import CREDITCARD, SYNTHETIC
from transactionguard.data.validation import validate_frame


def _codes(report) -> set[str]:
    return {i.code for i in report.issues}


def test_clean_synthetic_passes(small_synthetic):
    _, report = validate_frame(small_synthetic, SYNTHETIC, require_target=True)
    assert report.ok, report.to_dict()


def test_missing_required_column(small_synthetic):
    _, report = validate_frame(small_synthetic.drop(columns=["amount"]), SYNTHETIC, require_target=False)
    assert not report.ok and "missing_columns" in _codes(report)


def test_target_not_required_for_scoring(small_synthetic):
    _, report = validate_frame(small_synthetic.drop(columns=["is_fraud"]), SYNTHETIC, require_target=False)
    assert report.ok


@pytest.mark.parametrize(
    ("column", "value", "code"),
    [
        ("amount", "abc", "invalid_type"),
        ("amount", np.nan, "missing_values"),
        ("amount", np.inf, "non_finite"),
        ("amount", -5.0, "out_of_range"),
        ("amount", 10_000_000.0, "out_of_range"),
        ("channel", "carrier_pigeon", "invalid_category"),
        ("account_id", "", "missing_values"),
        ("timestamp", "not-a-date", "invalid_type"),
    ],
)
def test_row_level_errors_are_reported_with_ids(small_synthetic, column, value, code):
    df = small_synthetic.head(20).copy()
    df[column] = df[column].astype(object)
    df.loc[3, column] = value
    _, report = validate_frame(df, SYNTHETIC, require_target=False)
    assert code in _codes(report)
    assert df.loc[3, "transaction_id"] in report.invalid_row_ids
    assert len(report.invalid_row_ids) == 1  # other rows stay scorable


def test_future_timestamp_rejected(small_synthetic):
    df = small_synthetic.head(5).copy()
    df.loc[0, "timestamp"] = pd.Timestamp(datetime.now(UTC) + timedelta(days=2))
    _, report = validate_frame(df, SYNTHETIC, require_target=False)
    assert "future_timestamp" in _codes(report)


def test_duplicate_ids_rejected(small_synthetic):
    df = small_synthetic.head(10).copy()
    df.loc[5, "transaction_id"] = df.loc[4, "transaction_id"]
    _, report = validate_frame(df, SYNTHETIC, require_target=False)
    assert "duplicate_id" in _codes(report)
    assert report.to_dict()["n_invalid_rows"] == 1  # both rows share one id


def test_creditcard_time_must_be_non_negative():
    row = {"transaction_id": "x1", "Time": -1.0, "Amount": 10.0, **{f"V{i}": 0.0 for i in range(1, 29)}}
    _, report = validate_frame(pd.DataFrame([row]), CREDITCARD, require_target=False)
    assert "out_of_range" in _codes(report)


# ------------------------------------------------------------------ request model (API boundary)
def _valid_payload() -> dict:
    return {
        "transaction_id": "t-1",
        "timestamp": "2026-06-01T10:00:00Z",
        "account_id": "acct_1",
        "merchant_id": "m_groc_001",
        "amount": 12.5,
        "channel": "pos",
        "merchant_category": "grocery",
        "country": "IN",
    }


def test_request_model_accepts_valid_payload():
    model = SYNTHETIC.request_model()
    obj = model(**_valid_payload())
    assert obj.amount == 12.5


@pytest.mark.parametrize(
    "patch",
    [
        {"amount": float("inf")},
        {"amount": float("nan")},
        {"amount": -1},
        {"channel": "fax"},
        {"unexpected_field": 1},  # extra fields are forbidden, not ignored
        {"is_fraud": 1},  # clients cannot send the label
    ],
)
def test_request_model_rejects_bad_payloads(patch):
    with pytest.raises(ValidationError):
        SYNTHETIC.request_model()(**{**_valid_payload(), **patch})


def test_request_model_requires_every_contract_input():
    payload = _valid_payload()
    payload.pop("merchant_id")
    with pytest.raises(ValidationError):
        SYNTHETIC.request_model()(**payload)  # never silently defaulted


def test_request_model_never_includes_target():
    for contract in (SYNTHETIC, CREDITCARD):
        assert contract.target_column not in contract.request_model().model_fields
