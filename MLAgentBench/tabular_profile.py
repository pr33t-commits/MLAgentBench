"""Deterministic tabular profiling; no model calls or generated code."""
import json
from pathlib import Path

from .schema import EnvException

TABULAR_SUFFIXES = (".csv", ".tsv", ".tab", ".csv.gz", ".tsv.gz", ".tab.gz")


def profile_table(file_name, work_dir="."):
    import numpy as np
    import pandas as pd

    root = Path(work_dir).resolve()
    path = (root / file_name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise EnvException("Tabular input must be an existing file inside the workspace")
    sep = "\t" if path.name.lower().endswith((".tsv", ".tab", ".tsv.gz", ".tab.gz")) else ","
    try:
        frame = pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)
    except Exception as exc:
        raise EnvException(f"Cannot profile {file_name}: {type(exc).__name__}. No LLM fallback was attempted.") from exc
    result = {"file": file_name, "shape": list(frame.shape),
              "duplicate_rows": int(frame.duplicated().sum()), "columns": {}}
    for name in frame.columns:
        series = frame[name]
        info = {"dtype": str(series.dtype), "missing": int(series.isna().sum()),
                "unique_non_null": int(series.nunique())}
        if pd.api.types.is_numeric_dtype(series):
            finite = series[np.isfinite(series)]
            info.update(zero_count=int(series.eq(0).sum()),
                        zero_fraction_all_rows=float(series.eq(0).mean()) if len(series) else None,
                        nonfinite_non_null=int((series.notna() & ~np.isfinite(series)).sum()),
                        statistics=finite.describe(percentiles=[.25, .5, .75, .95]).to_dict())
        else:
            counts = series.value_counts().head(10)
            info["top_values"] = [{"value": str(value), "count": int(count)} for value, count in counts.items()]
        if name.lower() in ("date", "week", "datetime", "timestamp") or name.lower().endswith("_date"):
            dates = pd.to_datetime(series, errors="coerce", utc=True)
            info["date_coverage"] = {"start": str(dates.min()) if dates.notna().any() else None,
                                     "end": str(dates.max()) if dates.notna().any() else None,
                                     "invalid_non_null": int((series.notna() & dates.isna()).sum()),
                                     "unique_dates": int(dates.nunique())}
        result["columns"][str(name)] = info
    if {"sku", "store", "week"}.issubset(frame.columns):
        result["duplicate_sku_store_week_keys"] = int(frame.duplicated(["sku", "store", "week"]).sum())
        result["observed_sku_store_pairs"] = len(frame[["sku", "store"]].drop_duplicates())
    # Normalize pandas/numpy scalars and undefined statistics to JSON null.
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    return ("Computed profile of the complete file; no LLM calls. Numeric statistics exclude "
            "missing/infinite values. Zero fractions use all rows. Top categorical values are limited to 10. "
            "This standard profile does not answer every custom question or establish seasonality. "
            "For plots, seasonality analysis, or additional checks, use Create Script (AI) then Execute Script.\n"
            + json.dumps(clean(result), indent=2, allow_nan=False))
