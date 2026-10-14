"""The data contract: one definition of a dataset's columns, shared by training and inference.

Why a contract?
  * Training validation, the API request schema and the frontend form are all *generated* from
    the same object, so they cannot silently drift apart.
  * The frontend asks only for the raw fields the model's pipeline actually consumes.
  * Missing required fields are rejected; we never quietly fill them with made-up defaults.

Two contracts exist:
  * ``creditcard`` - the public ULB/OpenML benchmark (real transactions, PCA-anonymized).
  * ``synthetic``  - a deterministic, clearly-labelled synthetic dataset for demos/installation checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

DType = Literal["float", "int", "category", "datetime", "string", "bool"]
Role = Literal["id", "timestamp", "entity", "feature", "target"]
Provenance = Literal["real_anonymized", "synthetic"]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: DType
    role: Role
    description: str
    required: bool = True
    nullable: bool = False
    min_value: float | None = None
    max_value: float | None = None
    categories: tuple[str, ...] | None = None
    unit: str | None = None


@dataclass(frozen=True)
class DataContract:
    name: str
    version: str
    provenance: Provenance
    description: str
    fields: tuple[FieldSpec, ...]
    id_column: str
    timestamp_column: str
    target_column: str
    target_description: str
    # Entity used for history features and group-aware splits (None if the data has no IDs).
    entity_column: str | None = None
    # Free-form notes shown in the UI and docs (limitations, units, ...).
    notes: tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------ helpers
    def field(self, name: str) -> FieldSpec:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(name)

    @property
    def input_fields(self) -> tuple[FieldSpec, ...]:
        """Fields an analyst/API client must send to score one transaction (never the target)."""
        return tuple(f for f in self.fields if f.role != "target")

    @property
    def column_names(self) -> list[str]:
        return [f.name for f in self.fields]

    @property
    def input_column_names(self) -> list[str]:
        return [f.name for f in self.input_fields]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fields"] = [asdict(f) for f in self.fields]
        return d

    def request_model(self) -> type[BaseModel]:
        """Build a strict Pydantic model for ONE transaction from the contract.

        * ``extra="forbid"``: unknown fields are rejected (catches client typos).
        * ``allow_inf_nan=False``: NaN/inf are rejected at the API boundary.
        * categorical fields become ``Literal[...]`` so invalid categories fail with 422.
        """
        fields: dict[str, Any] = {}
        for f in self.input_fields:
            py_type: Any
            constraints: dict[str, Any] = {"description": f.description}
            if f.dtype == "float":
                py_type = float
                constraints["allow_inf_nan"] = False
            elif f.dtype == "int":
                py_type = int
            elif f.dtype == "bool":
                py_type = bool
            elif f.dtype == "category":
                assert f.categories, f"{f.name} needs categories"
                py_type = Literal[f.categories]  # type: ignore[valid-type]
            elif f.dtype == "datetime":
                from datetime import datetime

                py_type = datetime
            else:
                py_type = str
                constraints["min_length"] = 1
                constraints["max_length"] = 64
            if f.min_value is not None and f.dtype in ("float", "int"):
                constraints["ge"] = f.min_value
            if f.max_value is not None and f.dtype in ("float", "int"):
                constraints["le"] = f.max_value
            if f.nullable:
                py_type = py_type | None
            default = ... if f.required else None
            fields[f.name] = (py_type, Field(default, **constraints))
        model: type[BaseModel] = create_model(  # type: ignore[call-overload]
            f"{self.name.capitalize()}Transaction",
            __config__=ConfigDict(extra="forbid", str_strip_whitespace=True),
            **fields,
        )
        return model


# ============================================================================ creditcard
_PCA_FIELDS = tuple(
    FieldSpec(
        name=f"V{i}",
        dtype="float",
        role="feature",
        description=(
            f"V{i}: principal component {i} released by the data owner. The original features "
            "and their meaning are confidential; do not attribute a business meaning to it."
        ),
    )
    for i in range(1, 29)
)

CREDITCARD = DataContract(
    name="creditcard",
    version="1.0",
    provenance="real_anonymized",
    description=(
        "Credit card transactions by European cardholders over two days in September 2013 "
        "(ULB Machine Learning Group / Worldline), obtained from OpenML dataset 1597."
    ),
    fields=(
        FieldSpec(
            "transaction_id",
            "string",
            "id",
            "Identifier assigned by TransactionGuard from the original row order "
            "(the source has no ID column).",
        ),
        FieldSpec(
            "Time",
            "float",
            "timestamp",
            "Seconds elapsed between this transaction and the first transaction in the "
            "dataset. The real clock time of the first transaction is not published.",
            min_value=0.0,
            max_value=7 * 24 * 3600.0,
            unit="seconds",
        ),
        FieldSpec(
            "Amount",
            "float",
            "feature",
            "Transaction amount (currency not stated by the source).",
            min_value=0.0,
            max_value=1_000_000.0,
            unit="currency units",
        ),
        *_PCA_FIELDS,
        FieldSpec(
            "Class",
            "int",
            "target",
            "1 = fraud, 0 = genuine (as labelled by the source).",
            min_value=0,
            max_value=1,
        ),
    ),
    id_column="transaction_id",
    timestamp_column="Time",
    target_column="Class",
    target_description="Class = 1 for transactions the data owner labelled as fraudulent.",
    entity_column=None,
    notes=(
        "No account or merchant identifiers exist, so account-history features are impossible.",
        "Time is relative; 'hour_of_cycle' is derived as (Time mod 86400)/3600 and is NOT the "
        "local clock hour because the start time is unknown.",
        "V1-V28 are PCA components; explanations can only name them, not interpret them.",
    ),
)

# ============================================================================ synthetic
CHANNELS = ("online", "pos", "atm", "transfer")
MERCHANT_CATEGORIES = (
    "grocery",
    "restaurants",
    "fuel",
    "fashion",
    "electronics",
    "travel",
    "entertainment",
    "utilities",
    "gaming",
    "crypto_exchange",
)
COUNTRIES = ("IN", "US", "GB", "DE", "SG", "AE", "NG", "BR")

SYNTHETIC = DataContract(
    name="synthetic",
    version="1.0",
    provenance="synthetic",
    description=(
        "SYNTHETIC transactions produced by TransactionGuard's deterministic generator "
        "(scripts/generate_synthetic.py). For demos and installation checks only; results on it "
        "are not benchmark evidence."
    ),
    fields=(
        FieldSpec("transaction_id", "string", "id", "Unique transaction identifier."),
        FieldSpec(
            "timestamp",
            "datetime",
            "timestamp",
            "Transaction time (UTC).",
        ),
        FieldSpec("account_id", "string", "entity", "Paying account identifier."),
        FieldSpec(
            "merchant_id",
            "string",
            "feature",
            "Merchant identifier (used only for 'first time at this merchant'; never one-hot encoded).",
        ),
        FieldSpec(
            "amount",
            "float",
            "feature",
            "Transaction amount.",
            min_value=0.01,
            max_value=50_000.0,
            unit="currency units",
        ),
        FieldSpec("channel", "category", "feature", "How the transaction was made.", categories=CHANNELS),
        FieldSpec(
            "merchant_category", "category", "feature", "Merchant category.", categories=MERCHANT_CATEGORIES
        ),
        FieldSpec(
            "country", "category", "feature", "Country where the transaction occurred.", categories=COUNTRIES
        ),
        FieldSpec(
            "is_fraud",
            "int",
            "target",
            "1 = injected fraud pattern, 0 = normal (synthetic label).",
            min_value=0,
            max_value=1,
        ),
    ),
    id_column="transaction_id",
    timestamp_column="timestamp",
    target_column="is_fraud",
    target_description="is_fraud = 1 for rows the generator created from a fraud scenario.",
    entity_column="account_id",
    notes=(
        "Synthetic data: patterns were designed by the generator, so models can learn them easily.",
        "Account-history features use only that account's earlier transactions.",
    ),
)

CONTRACTS: dict[str, DataContract] = {c.name: c for c in (CREDITCARD, SYNTHETIC)}


def get_contract(name: str) -> DataContract:
    try:
        return CONTRACTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset contract {name!r}; choose from {sorted(CONTRACTS)}") from exc
