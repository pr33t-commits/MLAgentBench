"""Label-free demand-forecasting submission contract."""
import csv
import math

KEYS = ("sku", "store", "week")
TARGET = "sales_units"


def read_keys(path, predictions=False):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        expected = set(KEYS) | ({TARGET} if predictions else set())
        if predictions and (set(columns) != expected or len(columns) != 4):
            raise ValueError("Prediction columns must be sku,store,week,sales_units")
        if not set(KEYS) <= set(columns):
            raise ValueError("Missing forecast keys")
        keys = set()
        for row in reader:
            if None in row:
                raise ValueError("Malformed CSV row")
            key = tuple(row[k] for k in KEYS)
            if any(v is None or not v.strip() for v in key) or key in keys:
                raise ValueError("Missing or duplicate forecast keys")
            keys.add(key)
            if predictions:
                try:
                    value = float(row[TARGET])
                except (ValueError, TypeError):
                    raise ValueError("Missing or nonnumeric prediction") from None
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Predictions must be finite and nonnegative")
        if not keys:
            raise ValueError("Empty submission or test keys")
        return keys


def validate_submission(test_path, prediction_path):
    expected = read_keys(test_path)
    actual = read_keys(prediction_path, predictions=True)
    if expected != actual:
        raise ValueError("Prediction keys must exactly match test keys")
    return {"valid": True, "rows": len(actual)}
