"""Offline action tests; no model requests or generated-code execution."""

import importlib
from pathlib import Path
import sys
import tempfile
import types
import subprocess
import json
import gzip
import unittest
from unittest.mock import Mock, patch


class CreateScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the real actions and file tools in an isolated package. Stub only
        # the API provider and readline (unavailable on native Windows).
        package = types.ModuleType("_create_script_test")
        package.__path__ = [str(Path(__file__).resolve().parents[1] / "MLAgentBench")]
        llm = types.ModuleType("_create_script_test.LLM")
        llm.complete_text = Mock()
        llm.complete_text_fast = Mock()
        cls.modules = patch.dict(sys.modules, {
            package.__name__: package, llm.__name__: llm,
            "readline": types.ModuleType("readline"),
        })
        cls.modules.start()
        cls.actions = importlib.import_module("_create_script_test.high_level_actions")
        cls.schema = importlib.import_module("_create_script_test.schema")

    @classmethod
    def tearDownClass(cls):
        cls.modules.stop()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.actions.complete_text.reset_mock()
        self.actions.complete_text.return_value = "```python\nraise RuntimeError('not executed')\n```"
        self.trace = self.schema.Trace([], [], {}, "test")

    def create(self, **overrides):
        args = dict(datasets={}, objective="Create a forecasting script", script_name="train.py",
                    work_dir=str(self.root), research_problem="Demand forecasting",
                    read_only_files=[], log_file="test.log", trace=self.trace)
        args.update(overrides)
        return self.actions.create_script(**args)

    def test_understand_tables_never_calls_llm(self):
        self.actions.complete_text_fast.reset_mock()
        for filename, separator in [("sales.csv", ","), ("sales.tsv.gz", "\t")]:
            text = separator.join(["sku", "store", "week", "sales_units"]) + "\n"
            text += separator.join(["a", "s", "2022-01-03", "0"]) + "\n"
            text += separator.join(["b", "s", "2022-01-10", "10"]) + "\n"
            path = self.root / filename
            if filename.endswith(".gz"):
                with gzip.open(path, "wt") as stream:
                    stream.write(text)
            else:
                path.write_text(text)
            observation = self.actions.understand_file(filename, "Sales statistics", work_dir=str(self.root))
            profile = json.loads(observation.split("\n", 1)[1])
            self.assertEqual(profile["shape"], [2, 4])
            self.assertEqual(profile["columns"]["sales_units"]["statistics"]["mean"], 5)
            self.assertEqual(profile["columns"]["sales_units"]["zero_fraction_all_rows"], .5)
            self.assertEqual(profile["columns"]["sku"]["unique_non_null"], 2)
            self.assertEqual(profile["columns"]["week"]["date_coverage"]["unique_dates"], 2)
        (self.root / "empty.csv").write_text("")
        for filename in ("empty.csv", "missing.csv", "../outside.csv"):
            with self.assertRaises(self.schema.EnvException):
                self.actions.understand_file(filename, "Inspect", work_dir=str(self.root))
        self.actions.complete_text_fast.assert_not_called()
        self.actions.complete_text.assert_not_called()

    def test_understand_text_keeps_llm_route(self):
        self.actions.complete_text_fast.reset_mock()
        self.actions.complete_text_fast.return_value = "Text summary"
        (self.root / "notes.txt").write_text("Short note")
        result = self.actions.understand_file("notes.txt", "Summarize", work_dir=str(self.root), log_file="test.log", trace=self.trace)
        self.assertEqual(result, "Text summary")
        self.actions.complete_text_fast.assert_called_once()

    def test_registered_action_creates_and_records_without_execution(self):
        action = next(a for a in self.actions.HIGH_LEVEL_ACTIONS if a.name == "Create Script (AI)")
        self.assertEqual(set(action.usage), {"datasets", "objective", "script_name"})
        self.assertIs(action.function, self.actions.create_script)
        self.assertIn("Created train.py", self.create(library_context="sklearn available, version TEST"))
        self.assertIn("raise RuntimeError", (self.root / "train.py").read_text())
        self.assertEqual(self.trace.low_level_steps[0].action.name, "Write File")
        args, kwargs = self.actions.complete_text.call_args
        self.assertIn("Demand forecasting", args[0])
        self.assertIn("sklearn available, version TEST", args[0])
        self.assertIn("Create a forecasting script", args[0])
        self.assertEqual(kwargs["model"], self.actions.EDIT_SCRIPT_MODEL)

    def test_existing_file_is_preserved(self):
        (self.root / "train.py").write_text("original")
        with self.assertRaises(self.schema.EnvException):
            self.create()
        self.assertEqual((self.root / "train.py").read_text(), "original")
        self.actions.complete_text.assert_not_called()

    def test_invalid_requests_do_not_call_model(self):
        for override in [dict(objective=" "), dict(script_name="../escape.py"),
                         dict(script_name=str(self.root / "absolute.py")),
                         dict(script_name="data.csv"), dict(script_name="missing/new.py"),
                         dict(read_only_files=["./train.py"])]:
            with self.subTest(override=override), self.assertRaises(self.schema.EnvException):
                self.create(**override)
        self.actions.complete_text.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_bad_generations_leave_no_file(self):
        for completion in ["no code", "```python\n```", "```python\ndef broken(\n```",
                           "```python\nx=1\n```\n```python\ny=2\n```",
                           "```python\nx=1"]:
            with self.subTest(completion=completion), self.assertRaises(self.schema.EnvException):
                self.actions.complete_text.return_value = completion
                self.create()
            self.assertFalse((self.root / "train.py").exists())

    def test_dataset_contract_errors_precede_model_call(self):
        (self.root / "data.bin").write_bytes(b"abc")
        for datasets in [None, [], "", '{"train": "sales.csv"}',
                         {"train": "missing.csv"}, {"train": "../outside.csv"},
                         {"train": "data.bin"}, {"train": 1}, {"": "sales.csv"},
                         {1: "sales.csv"}, {"train": ""}]:
            with self.subTest(datasets=datasets), self.assertRaises(self.schema.EnvException):
                self.create(datasets=datasets)
        self.actions.complete_text.assert_not_called()
        self.assertFalse((self.root / "train.py").exists())

    def test_injected_loaders_work_from_another_directory(self):
        (self.root / "input files").mkdir()
        (self.root / "scripts").mkdir()
        (self.root / "input files/sales.csv").write_text('sku,sales_units\na,7\nb,9\n')
        (self.root / "items.tsv").write_text('sku\tcategory\na\tcat_a\n')
        (self.root / "config.json").write_text('{"horizon": 13}')
        self.actions.complete_text.return_value = '''```python
"""Test script."""
from __future__ import annotations
assert DATASETS["sales"].sales_units.sum() == 16
assert DATASETS["items"].category.iloc[0] == "cat_a"
assert DATASETS["config"]["horizon"] == 13
print("loaded real inputs")
```'''
        self.create(script_name="scripts/load.py", objective='Check inputs', datasets={"sales": "sales.csv", "items": "items.tsv", "config": "config.json"})
        prompt = self.actions.complete_text.call_args.args[0]
        self.assertIn('"sales_units"', prompt)
        self.assertIn('"category"', prompt)
        self.assertIn("NO tools", prompt)
        self.assertIn("_DATASET_INPUTS =", prompt)
        self.assertIn("input files", prompt)
        self.assertIn("return ONLY the", prompt)
        # Execute only the fixed, test-authored assertions above, never real LLM output.
        with tempfile.TemporaryDirectory() as other:
            result = subprocess.run([sys.executable, str(self.root / "scripts/load.py")],
                                    cwd=other, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("loaded real inputs", result.stdout)

    def test_duplicate_entries_require_disambiguation(self):
        for name in ("first", "second"):
            (self.root / name).mkdir()
            (self.root / name / "sales.csv").write_text("sales_units\n5\n")
        with self.assertRaisesRegex(self.schema.EnvException, "exactly one"):
            self.create(objective='Read sales', datasets={"sales": "sales.csv"})
        self.actions.complete_text.assert_not_called()
        self.create(objective='Read sales', datasets={"sales": "first/sales.csv"})
        self.assertIn("first", (self.root / "train.py").read_text())

    def test_list_files_classification_marker(self):
        (self.root / "sales.csv").write_text("sales_units\n5\n")
        self.create(objective='Read sales', datasets={"sales": "sales.csv*"})
        self.assertIn("'sales': 'sales.csv'", (self.root / "train.py").read_text())

    def test_tool_example_parses_and_creates_with_structured_datasets(self):
        from test_action_input import Agent
        action = next(a for a in self.actions.HIGH_LEVEL_ACTIONS if a.name == "Create Script (AI)")
        raw = action.description.split("Action Input: ", 1)[1].splitlines()[0]
        parsed = Agent.parse_action_input(raw, action)
        self.assertIsInstance(parsed["datasets"], dict)
        self.assertNotIn("DATASETS:", parsed["objective"])
        for name in parsed["datasets"].values():
            (self.root / name).write_text("sales_units\n5\n")
        self.create(**parsed)
        content = (self.root / parsed["script_name"]).read_text()
        self.assertIn("'train': 'train.csv'", content)
        self.assertIn("'validation': 'validation.csv'", content)
        self.assertIn("'test': 'test.csv'", content)

    def test_legacy_objective_does_not_select_files(self):
        self.create(objective='DATASETS: {"train": "missing.csv"}')
        self.assertIn("_DATASET_INPUTS = {}", (self.root / "train.py").read_text())


if __name__ == "__main__":
    unittest.main()
