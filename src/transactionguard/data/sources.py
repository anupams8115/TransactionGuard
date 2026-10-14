"""Dataset acquisition and loading.

The benchmark data is *downloaded* from OpenML at setup time and kept out of git: we document
provenance but never redistribute it. ``fetch_openml`` verifies the file's MD5 checksum against
OpenML's metadata, and we record a SHA-256 fingerprint of the processed table for MLflow.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from transactionguard.core.config import get_settings
from transactionguard.core.logging import get_logger
from transactionguard.data import synthetic
from transactionguard.data.contract import CREDITCARD, SYNTHETIC, DataContract, get_contract
from transactionguard.data.validation import validate_frame

log = get_logger(__name__)

OPENML_CREDITCARD_ID = 1597
PROVENANCE = {
    "creditcard": {
        "source": "OpenML dataset 1597 'creditcard' (https://www.openml.org/d/1597)",
        "original": "ULB Machine Learning Group / Worldline - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        "license": "Original release: Open Database License with Database Contents License (DbCL) v1.0; "
        "OpenML lists the licence as 'Public'. Not redistributed by this repository.",
        "access": "Public download from OpenML, no login required.",
        "citation": "Dal Pozzolo, Caelen, Johnson, Bontempi. Calibrating Probability with "
        "Undersampling for Unbalanced Classification. IEEE CIDM 2015.",
        "kind": "real transactions, anonymized via PCA",
    },
    "synthetic": {
        "source": "TransactionGuard deterministic generator (src/transactionguard/data/synthetic.py)",
        "license": "Generated data; same licence as this repository.",
        "access": "Generated locally.",
        "kind": "synthetic",
    },
}


def processed_path(name: str) -> Path:
    return get_settings().data_dir / "processed" / f"{name}.parquet"


def fingerprint(df: pd.DataFrame) -> str:
    """Order-sensitive SHA-256 of the table contents (logged to MLflow with every run)."""
    h = hashlib.sha256()
    h.update(json.dumps(list(map(str, df.columns))).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()


def _download_openml_arff(data_id: int, dest_dir: Path) -> Path:
    """Download the dataset's original ARFF file and verify OpenML's published MD5.

    We do not use ``sklearn.datasets.fetch_openml`` here because OpenML marks ``Time`` as the
    row-id attribute of dataset 1597 and ``fetch_openml`` therefore drops it - but we need
    ``Time`` for a chronological split.
    """
    import gzip

    import httpx

    meta = httpx.get(f"https://www.openml.org/api/v1/json/data/{data_id}", timeout=30)
    meta.raise_for_status()
    desc = meta.json()["data_set_description"]
    url, expected_md5 = desc["url"], desc["md5_checksum"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"openml_{data_id}.arff.gz"
    if not path.exists():
        log.info("downloading_openml", data_id=data_id)
        tmp = path.with_suffix(".part")
        with httpx.stream("GET", url, timeout=120, follow_redirects=True) as r, gzip.open(tmp, "wb") as fh:
            r.raise_for_status()
            for chunk in r.iter_bytes():
                fh.write(chunk)
        tmp.replace(path)
    md5 = hashlib.md5(usedforsecurity=False)
    with gzip.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            md5.update(chunk)
    if md5.hexdigest() != expected_md5:
        path.unlink()
        raise RuntimeError(
            f"MD5 mismatch for OpenML {data_id}: got {md5.hexdigest()}, "
            f"expected {expected_md5}. Deleted the file; please retry."
        )
    log.info("openml_checksum_verified", data_id=data_id, md5=expected_md5)
    return path


def _read_arff(path: Path) -> pd.DataFrame:
    import gzip

    names: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            low = line.strip().lower()
            if low.startswith("@attribute"):
                names.append(line.split()[1])
            elif low.startswith("@data"):
                return pd.read_csv(fh, header=None, names=names, quotechar="'")
    raise ValueError("ARFF file has no @data section")


def prepare_creditcard(force: bool = False) -> Path:
    out = processed_path("creditcard")
    if out.exists() and not force:
        log.info("dataset_already_prepared", path=str(out))
        return out
    arff = _download_openml_arff(OPENML_CREDITCARD_ID, get_settings().data_dir / "raw")
    df = _read_arff(arff)
    df["Class"] = pd.to_numeric(df["Class"], errors="raise").astype(int)
    for c in df.columns:
        if c != "Class":
            df[c] = pd.to_numeric(df[c], errors="raise").astype("float64")
    df.insert(0, "transaction_id", [f"cc-{i:06d}" for i in range(len(df))])
    df = df[CREDITCARD.column_names]
    _validate_and_write(df, CREDITCARD, out)
    return out


def prepare_synthetic(config: synthetic.SyntheticConfig | None = None, force: bool = False) -> Path:
    out = processed_path("synthetic")
    if out.exists() and not force:
        log.info("dataset_already_prepared", path=str(out))
        return out
    df = synthetic.generate(config)
    _validate_and_write(df, SYNTHETIC, out)
    return out


def _validate_and_write(df: pd.DataFrame, contract: DataContract, out: Path) -> None:
    coerced, report = validate_frame(df, contract, require_target=True)
    report.raise_if_failed()
    out.parent.mkdir(parents=True, exist_ok=True)
    coerced.to_parquet(out, index=False)
    summary = dataset_summary(coerced, contract)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info(
        "dataset_prepared",
        dataset=contract.name,
        rows=len(coerced),
        positives=summary["positives"],
        fingerprint=summary["fingerprint"][:12],
    )


def dataset_summary(df: pd.DataFrame, contract: DataContract) -> dict:
    y = df[contract.target_column].astype(int)
    ts = df[contract.timestamp_column]
    return {
        "dataset": contract.name,
        "provenance": PROVENANCE[contract.name],
        "rows": len(df),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "time_min": ts.min(),
        "time_max": ts.max(),
        "fingerprint": fingerprint(df),
    }


def load(name: str) -> pd.DataFrame:
    contract = get_contract(name)
    path = processed_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `tg data prepare {name}` first.")
    df = pd.read_parquet(path)
    coerced, report = validate_frame(df, contract, require_target=True)
    report.raise_if_failed()
    return coerced
