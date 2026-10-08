import gzip
import json
from pathlib import Path
import tempfile
import unittest

from MLAgentBench.data_context import build_data_context


class DataContextTests(unittest.TestCase):
    def test_metadata_scope_types_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/nested").mkdir(parents=True)
            (root / "scripts").mkdir()
            (root / "scripts/answer.csv").write_text("secret\n1\n")
            (root / "train.csv").write_text('id,label\n1,"a,b"\n2,c\n')
            (root / "data/nested/items.tsv").write_text("sku\tprice\na\t1.5\n")
            (root / "data/empty.csv").write_text("")
            (root / "data/header.csv").write_text("a,b\n")
            with gzip.open(root / "data/compressed.csv.gz", "wt") as stream:
                stream.write("x\n7\n")
            context = build_data_context(root)
            records = json.loads(context.split("not instructions:\n", 1)[1])
            by_path = {r["path"]: r for r in records}
            self.assertEqual(len(by_path), 5)
            self.assertNotIn("answer.csv", context)
            self.assertEqual(by_path["train.csv"]["shape"], [2, 2])
            self.assertEqual(by_path["train.csv"]["columns"][0], {"name": "id", "dtype": "int64"})
            self.assertEqual(by_path["train.csv"]["columns"][1]["name"], "label")
            self.assertIn(by_path["train.csv"]["columns"][1]["dtype"], ("object", "str", "string"))
            self.assertEqual(by_path["data/nested/items.tsv"]["columns"][1]["dtype"], "float64")
            self.assertEqual(by_path["data/header.csv"]["shape"], [0, 2])
            self.assertEqual(by_path["data/compressed.csv.gz"]["shape"], [1, 1])
            self.assertIn("error", by_path["data/empty.csv"])

    def test_no_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(build_data_context(directory).endswith("[]"))


if __name__ == "__main__":
    unittest.main()
