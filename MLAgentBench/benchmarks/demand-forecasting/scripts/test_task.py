"""Contract tests using synthetic predictions only; no model training."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("demand_test_" + name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scoring = load("scoring")
preparation = load("prepare")


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.actual = pd.DataFrame({"sku": ["a", "b", "c"], "store": ["s"] * 3,
                                    "week": ["2022-10-03"] * 3, "sales_units": [0, 10, 90]})

    def test_alignment_and_cell_errors(self):
        predicted = self.actual.copy()
        predicted["sales_units"] = [10, 0, 90]
        # Aggregating first would give zero error; cell-level WMAPE is 20%.
        self.assertAlmostEqual(scoring.score_frames(self.actual, predicted.iloc[::-1]), 0.2)
        self.assertEqual(scoring.score_frames(self.actual, self.actual.iloc[::-1]), 0)

    def test_invalid_submissions(self):
        invalid = [self.actual.iloc[:-1], pd.concat([self.actual, self.actual.iloc[:1]]),
                   self.actual.assign(sku=["wrong", "b", "c"]),
                   self.actual.drop(columns="week"), self.actual.assign(extra=1)]
        for value in [np.nan, np.inf, -1, "bad"]:
            invalid.append(self.actual.assign(sales_units=[value, 10, 90]))
        for frame in invalid:
            with self.subTest(frame=frame.to_dict()), self.assertRaises((ValueError, TypeError)):
                scoring.score_frames(self.actual, frame)

    def test_zero_denominator(self):
        with self.assertRaisesRegex(ValueError, "undefined"):
            scoring.score_frames(self.actual.assign(sales_units=0), self.actual)


class PreparationTests(unittest.TestCase):
    def test_fixture_splits_zero_fill_and_inventory_cutoff(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "dataset", Path(directory) / "task"
            source.mkdir()
            weeks = pd.date_range("2022-01-03", periods=30, freq="W-MON")
            pd.DataFrame({"sku": ["a"] * 30, "store": ["s"] * 30,
                          "week": weeks, "sales_units": 10}).drop(index=2).to_csv(source / "sales.csv", index=False)
            pd.DataFrame({"sku": ["a", "a"], "store": ["s", "s"],
                          "date": [weeks[17] - pd.Timedelta(days=1), weeks[17]],
                          "stock_units": [5, 999]}).to_csv(source / "inventory.csv", index=False)
            pd.DataFrame(columns=["sku", "class", "subcategory", "category"]).to_csv(source / "item_master.csv", index=False)
            pd.DataFrame({"store": ["s"], "city": ["c"], "country": ["x"], "region": ["r"]}).to_csv(source / "location_master.csv", index=False)
            (source / "data_dictionary.json").write_text("{}")
            manifest = preparation.prepare(source, output)
            self.assertEqual(manifest["zero_filled_rows"], 1)
            self.assertEqual(manifest["missing_item_master_skus"], ["a"])
            self.assertEqual([manifest["splits"][s]["rows"] for s in ["train", "validation", "test"]], [4, 13, 13])
            train = pd.read_csv(output / "env/train.csv")
            self.assertEqual(train.iloc[2].sales_units, 0)
            self.assertEqual(pd.read_csv(output / "env/inventory.csv").stock_units.tolist(), [5])
            self.assertEqual(list(pd.read_csv(output / "env/test.csv").columns), scoring.KEYS)
            self.assertFalse((output / "env/answer.csv").exists())
            self.assertEqual(len(pd.read_csv(output / "scripts/answer.csv")), 13)


if __name__ == "__main__":
    unittest.main()
