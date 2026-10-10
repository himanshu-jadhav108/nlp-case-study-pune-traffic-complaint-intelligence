"""Dataset loading, validation, and leakage-aware data splitting."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["Congestion", "Accident", "Signal", "Parking", "Construction", "Road Condition"]
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_traffic_complaints.csv"


class DatasetError(ValueError):
    """Raised when a user-provided complaint dataset is invalid."""


def validate_complaints(frame: pd.DataFrame, require_category: bool = True) -> pd.DataFrame:
    required = {"complaint_text"} | ({"category"} if require_category else set())
    missing = required - set(frame.columns)
    if missing:
        raise DatasetError(f"Missing required column(s): {', '.join(sorted(missing))}.")
    result = frame.copy()
    result["complaint_text"] = result["complaint_text"].fillna("").astype(str).str.strip()
    if not result["complaint_text"].str.len().gt(0).any():
        raise DatasetError("The file contains no non-blank complaints.")
    if require_category:
        result["category"] = result["category"].fillna("").astype(str).str.strip()
        bad = sorted(set(result.loc[~result.category.isin(LABELS), "category"]))
        if bad:
            raise DatasetError("Unsupported category label(s): " + ", ".join(repr(x) for x in bad))
        if result["category"].eq("").any():
            raise DatasetError("Category values cannot be blank.")
    return result


def load_sample_data() -> pd.DataFrame:
    return validate_complaints(pd.read_csv(DATA_PATH))


def split_data(frame: pd.DataFrame, seed: int = 42):
    """Return balanced train/validation/test splits ensuring all categories are represented."""
    frame = validate_complaints(frame)
    if len(frame) < 6 or frame.category.nunique() < 2:
        raise DatasetError("At least six rows and two categories are needed for evaluation.")
    idx = frame.index.to_numpy()
    stratify = frame.category if frame.category.value_counts().min() >= 3 else None
    trainval, test = train_test_split(idx, test_size=0.2, random_state=seed, stratify=stratify)
    remain = frame.loc[trainval]
    val_stratify = remain.category if remain.category.value_counts().min() >= 2 else None
    train, val = train_test_split(trainval, test_size=0.25, random_state=seed + 1, stratify=val_stratify)
    return frame.loc[train].copy(), frame.loc[val].copy(), frame.loc[test].copy()
