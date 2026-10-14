"""Batch data validation against a :class:`DataContract`.

Used for (1) training data before any model sees it and (2) uploaded CSVs before batch scoring.
Returns a report instead of raising immediately so the UI can show *every* problem at once.
Single API requests are validated by the Pydantic model from ``contract.request_model()``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd

from transactionguard.data.contract import DataContract

Severity = Literal["error", "warning"]
MAX_EXAMPLES = 5
_BOOLS = {"true": True, "false": False, "1": True, "0": False}


@dataclass
class Issue:
    code: str
    severity: Severity
    column: str | None
    count: int
    message: str
    example_ids: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    contract: str
    n_rows: int
    issues: list[Issue] = field(default_factory=list)
    invalid_row_ids: set[str] = field(default_factory=set)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def add(self, issue: Issue) -> None:
        self.issues.append(issue)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "n_rows": self.n_rows,
            "ok": self.ok,
            "n_invalid_rows": len(self.invalid_row_ids),
            "issues": [asdict(i) for i in self.issues],
        }

    def raise_if_failed(self) -> None:
        if not self.ok:
            lines = [
                f"[{i.code}] {i.column}: {i.message} ({i.count} rows)"
                for i in self.issues
                if i.severity == "error"
            ]
            raise DataValidationError("Data validation failed:\n" + "\n".join(lines), self)


class DataValidationError(ValueError):
    def __init__(self, message: str, report: ValidationReport):
        super().__init__(message)
        self.report = report


def _ids(df: pd.DataFrame, mask: pd.Series, id_col: str) -> list[str]:
    if id_col in df.columns:
        return df.loc[mask, id_col].astype(str).head(MAX_EXAMPLES).tolist()
    return [str(i) for i in df.index[mask][:MAX_EXAMPLES]]


def validate_frame(
    df: pd.DataFrame,
    contract: DataContract,
    *,
    require_target: bool,
    now: datetime | None = None,
    future_tolerance: timedelta = timedelta(minutes=5),
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate and type-coerce ``df``.

    Returns the coerced frame (only contract columns, in contract order) and a report.
    Rows with row-level errors are listed in ``report.invalid_row_ids`` so batch scoring can
    skip them *explicitly* (and record why) rather than guessing values.
    """
    report = ValidationReport(contract=contract.name, n_rows=len(df))
    specs = contract.fields if require_target else contract.input_fields
    id_col = contract.id_column

    # 1. Required columns --------------------------------------------------------------
    missing_cols = [f.name for f in specs if f.required and f.name not in df.columns]
    if missing_cols:
        report.add(
            Issue(
                "missing_columns",
                "error",
                None,
                len(missing_cols),
                f"Missing required columns: {missing_cols}",
            )
        )
        return df, report  # nothing else is meaningful without the schema
    unexpected = [c for c in df.columns if c not in contract.column_names]
    if unexpected:
        report.add(
            Issue(
                "unexpected_columns",
                "warning",
                None,
                len(unexpected),
                f"Ignoring columns not in the contract: {unexpected[:10]}",
            )
        )

    out = pd.DataFrame(index=df.index)
    bad_rows = pd.Series(False, index=df.index)

    for spec in specs:
        if spec.name not in df.columns:
            continue
        raw = df[spec.name]
        # 2. Types -------------------------------------------------------------------
        if spec.dtype in ("float", "int"):
            col = pd.to_numeric(raw, errors="coerce")
            type_bad = col.isna() & raw.notna()
        elif spec.dtype == "datetime":
            col = pd.to_datetime(raw, errors="coerce", utc=True)
            type_bad = col.isna() & raw.notna()
        elif spec.dtype == "bool":
            col = raw.map(lambda v: _BOOLS.get(str(v).strip().lower()) if pd.notna(v) else None)
            type_bad = col.isna() & raw.notna()
        else:  # string / category
            col = raw.astype("string").str.strip()
            type_bad = pd.Series(False, index=df.index)
        if type_bad.any():
            report.add(
                Issue(
                    "invalid_type",
                    "error",
                    spec.name,
                    int(type_bad.sum()),
                    f"Values could not be parsed as {spec.dtype}",
                    _ids(df, type_bad, id_col),
                )
            )
            bad_rows |= type_bad

        # 3. Missing values ------------------------------------------------------------
        missing = col.isna() & ~type_bad
        if spec.dtype in ("string", "category"):
            missing |= col.fillna("").eq("")
        if missing.any() and not spec.nullable:
            report.add(
                Issue(
                    "missing_values",
                    "error",
                    spec.name,
                    int(missing.sum()),
                    "Required value is missing",
                    _ids(df, missing, id_col),
                )
            )
            bad_rows |= missing

        # 4. Non-finite numbers ----------------------------------------------------------
        if spec.dtype in ("float", "int"):
            nonfinite = col.notna() & ~np.isfinite(col.astype("float64"))
            if nonfinite.any():
                report.add(
                    Issue(
                        "non_finite",
                        "error",
                        spec.name,
                        int(nonfinite.sum()),
                        "Infinite values are not allowed",
                        _ids(df, nonfinite, id_col),
                    )
                )
                bad_rows |= nonfinite
            if spec.dtype == "int":
                frac = col.notna() & np.isfinite(col.astype("float64")) & (col % 1 != 0)
                if frac.any():
                    report.add(
                        Issue(
                            "invalid_type",
                            "error",
                            spec.name,
                            int(frac.sum()),
                            "Expected whole numbers",
                            _ids(df, frac, id_col),
                        )
                    )
                    bad_rows |= frac
            # 5. Ranges (amount, relative time, labels) ---------------------------------------
            if spec.min_value is not None:
                low = col.notna() & (col < spec.min_value)
                if low.any():
                    report.add(
                        Issue(
                            "out_of_range",
                            "error",
                            spec.name,
                            int(low.sum()),
                            f"Values below minimum {spec.min_value}",
                            _ids(df, low, id_col),
                        )
                    )
                    bad_rows |= low
            if spec.max_value is not None:
                high = col.notna() & np.isfinite(col.astype("float64")) & (col > spec.max_value)
                if high.any():
                    report.add(
                        Issue(
                            "out_of_range",
                            "error",
                            spec.name,
                            int(high.sum()),
                            f"Values above maximum {spec.max_value}",
                            _ids(df, high, id_col),
                        )
                    )
                    bad_rows |= high

        # 6. Categories ------------------------------------------------------------------
        if spec.dtype == "category" and spec.categories:
            invalid = col.notna() & ~col.isin(spec.categories) & ~missing
            if invalid.any():
                examples = sorted(col[invalid].astype(str).unique().tolist())[:MAX_EXAMPLES]
                report.add(
                    Issue(
                        "invalid_category",
                        "error",
                        spec.name,
                        int(invalid.sum()),
                        f"Unknown categories {examples}; allowed: {list(spec.categories)}",
                        _ids(df, invalid, id_col),
                    )
                )
                bad_rows |= invalid

        # 7. Timestamps: not in the future -------------------------------------------------
        if spec.dtype == "datetime":
            limit = pd.Timestamp(now or datetime.now(UTC)) + future_tolerance
            future = col.notna() & (col > limit)
            if future.any():
                report.add(
                    Issue(
                        "future_timestamp",
                        "error",
                        spec.name,
                        int(future.sum()),
                        "Timestamp is in the future",
                        _ids(df, future, id_col),
                    )
                )
                bad_rows |= future

        if spec.dtype == "int":
            col = col.astype("Int64")
        out[spec.name] = col

    # 8. Duplicate identifiers -------------------------------------------------------------
    if id_col in out.columns:
        dup = out[id_col].notna() & out[id_col].duplicated(keep=False)
        if dup.any():
            report.add(
                Issue(
                    "duplicate_id",
                    "error",
                    id_col,
                    int(dup.sum()),
                    "Transaction identifiers must be unique within a file",
                    _ids(out, dup, id_col),
                )
            )
            bad_rows |= dup

    if id_col in out.columns:
        report.invalid_row_ids = set(out.loc[bad_rows, id_col].astype(str))
    return out, report
