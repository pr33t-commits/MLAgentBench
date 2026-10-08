"""Prepare the local demand panel without fitting or running a model."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
BENCHMARK = SCRIPTS.parent
KEYS = ["sku", "store", "week"]
HORIZON = 13


def read_table(source, filename, columns, keys, date_column=None, target=None):
    frame = pd.read_csv(source / filename)
    if list(frame.columns) != columns:
        raise ValueError(f"Unexpected columns in {filename}: {list(frame.columns)}")
    if frame.isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"Missing values or duplicate keys in {filename}")
    if date_column:
        frame[date_column] = pd.to_datetime(frame[date_column], format="%Y-%m-%d", errors="raise")
    if target:
        values = pd.to_numeric(frame[target], errors="raise")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid {target} in {filename}")
        frame[target] = values
    return frame


def prepare(source, output=BENCHMARK):
    source, output = Path(source).resolve(), Path(output).resolve()
    sales = read_table(source, "sales.csv", KEYS + ["sales_units"], KEYS, "week", "sales_units")
    inventory = read_table(source, "inventory.csv", ["sku", "store", "date", "stock_units"],
                           ["sku", "store", "date"], "date", "stock_units")
    items = read_table(source, "item_master.csv", ["sku", "class", "subcategory", "category"], ["sku"])
    locations = read_table(source, "location_master.csv", ["store", "city", "country", "region"], ["store"])
    dictionary = json.loads((source / "data_dictionary.json").read_text(encoding="utf-8"))
    if sales.empty or not sales.week.dt.dayofweek.eq(0).all():
        raise ValueError("Sales must contain Monday week starts")
    weeks = pd.date_range(sales.week.min(), sales.week.max(), freq="W-MON")
    if len(weeks) <= 2 * HORIZON:
        raise ValueError("Need more than 26 weeks for train/validation/test")
    skus, stores = sorted(sales.sku.unique()), sorted(sales.store.unique())
    # The task defines missing sales rows as zero, including each series' gaps.
    panel = sales.set_index(KEYS).reindex(
        pd.MultiIndex.from_product([skus, stores, weeks], names=KEYS), fill_value=0
    ).reset_index().sort_values(["week", "sku", "store"]).reset_index(drop=True)
    validation_start, test_start = weeks[-2 * HORIZON], weeks[-HORIZON]
    train = panel[panel.week < validation_start]
    validation = panel[(panel.week >= validation_start) & (panel.week < test_start)]
    test = panel[panel.week >= test_start]
    # Static master data is assumed known at both forecast origins. Missing master
    # rows remain missing; agents must left join rather than drop these series.
    historical_inventory = inventory[inventory.date < test_start]
    manifest = {
        "task": "demand-forecasting", "horizon_weeks": HORIZON,
        "grain": KEYS, "target": "sales_units", "metric": "WMAPE",
        "metric_direction": "minimize", "metric_scale": "fraction",
        "missing_sales": "zero", "panel_rows": len(panel), "source_sales_rows": len(sales),
        "zero_filled_rows": len(panel) - len(sales), "skus": len(skus), "stores": len(stores),
        "missing_item_master_skus": sorted(set(skus) - set(items.sku)),
        "missing_location_master_stores": sorted(set(stores) - set(locations.store)),
        "inventory_exclusive_cutoff": test_start.strftime("%Y-%m-%d"),
        "validation_inventory_exclusive_cutoff": validation_start.strftime("%Y-%m-%d"),
        "splits": {name: {"start": frame.week.min().strftime("%Y-%m-%d"),
                          "end": frame.week.max().strftime("%Y-%m-%d"), "rows": len(frame)}
                   for name, frame in [("train", train), ("validation", validation), ("test", test)]},
        "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                          for name in ["sales.csv", "inventory.csv", "item_master.csv",
                                       "location_master.csv", "data_dictionary.json"]},
    }
    env, scripts = output / "env", output / "scripts"
    env.mkdir(parents=True, exist_ok=True)
    scripts.mkdir(parents=True, exist_ok=True)
    for name, frame in [("train.csv", train), ("validation.csv", validation),
                        ("test.csv", test[KEYS]), ("inventory.csv", historical_inventory),
                        ("item_master.csv", items), ("location_master.csv", locations)]:
        frame.to_csv(env / name, index=False, date_format="%Y-%m-%d")
    sample = test[KEYS].copy()
    sample["sales_units"] = 0.0
    sample.to_csv(env / "sample_submission.csv", index=False, date_format="%Y-%m-%d")
    test.to_csv(scripts / "answer.csv", index=False, date_format="%Y-%m-%d")
    (env / "data_dictionary.json").write_text(json.dumps(dictionary, indent=2) + "\n", encoding="utf-8")
    (env / "task_metadata.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(SCRIPTS / "scoring.py", env / "metrics.py")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path(os.environ.get(
        "MLAGENTBENCH_DEMAND_DATASET", str(SCRIPTS.parents[4] / "dataset"))))
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset_dir), indent=2))
