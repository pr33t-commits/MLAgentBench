"""Trusted, key-aligned WMAPE scoring. No aggregation before absolute errors."""

import numpy as np
import pandas as pd

KEYS = ["sku", "store", "week"]
TARGET = "sales_units"


def score_frames(actual, prediction):
    """Return WMAPE as a fraction; reject incomplete or invalid submissions."""
    columns = KEYS + [TARGET]
    for name, frame in [("actual", actual), ("prediction", prediction)]:
        if set(frame.columns) != set(columns) or len(frame.columns) != len(columns):
            raise ValueError(f"{name} must contain exactly {columns}")
        if frame.empty or frame[columns].isna().any().any():
            raise ValueError(f"{name} contains missing values or is empty")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{name} contains duplicate forecast keys")
        values = pd.to_numeric(frame[TARGET], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} requires finite, nonnegative sales_units")
    aligned = actual.merge(prediction, on=KEYS, how="outer", validate="one_to_one",
                           indicator=True, suffixes=("_actual", "_prediction"))
    if not aligned["_merge"].eq("both").all():
        raise ValueError("Prediction keys must exactly match the evaluation keys")
    y = aligned[TARGET + "_actual"].to_numpy(dtype=float)
    p = aligned[TARGET + "_prediction"].to_numpy(dtype=float)
    denominator = y.sum()
    if denominator == 0:
        raise ValueError("WMAPE is undefined when total actual demand is zero")
    return float(np.abs(y - p).sum() / denominator)


def score_files(actual_path, prediction_path):
    return score_frames(pd.read_csv(actual_path), pd.read_csv(prediction_path))
