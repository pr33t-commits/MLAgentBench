"""Real subprocess and artifact checks; no model/API calls."""
import json
import importlib.util
import shutil
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

from MLAgentBench.completeness import execute_experiment, evaluate, judge_pending
from MLAgentBench.submission_validation import validate_submission


class CompletenessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.receipts = self.root / "receipts"
        self.receipts.mkdir()
        (self.receipts / "test_keys.csv").write_text("sku,store,week\na,s,2022-10-03\n")
        self.csv = "sku,store,week,sales_units\na,s,2022-10-03,2\n"
        self.script = self.work / "model.py"
        self.script.write_text("import os\nfrom pathlib import Path\n"
                               "(Path(os.environ['EXPERIMENT_OUTPUT_DIR']) / 'submission.csv').write_text("
                               + repr(self.csv) + ")\n")

    def run_model(self, role="baseline", parent="", timeout=5):
        return execute_experiment("model.py", role, parent, self.work,
                                  self.receipts, sys.executable, timeout=timeout)

    def test_baseline_experiment_requires_bound_semantic_judgment(self):
        baseline = self.run_model()
        self.assertTrue(baseline["accepted"])
        self.assertEqual(evaluate(self.receipts)["completeness_score"], 50)
        experiment = self.run_model("experiment", baseline["execution_id"])
        report = evaluate(self.receipts)
        self.assertIsNone(report["completeness_score"])
        self.assertEqual(report["score_bounds"], [50, 100])
        packet = report["judge_requests"][0]
        assessment = {"receipt_hashes": packet["receipt_hashes"], "verdict": "pass",
                      "reason": "Fixture approval", "evidence": ["model.py:1"], "judge": "test-reviewer"}
        judgments = {experiment["execution_id"]: assessment}
        self.assertEqual(evaluate(self.receipts, judgments)["completeness_score"], 100)
        assessment["receipt_hashes"] = {}
        self.assertEqual(evaluate(self.receipts, judgments)["experiment"], "unknown")
        self.assertEqual(baseline["artifact_sha256"], experiment["artifact_sha256"])

    def test_nonzero_exit_and_stale_workspace_submission_do_not_count(self):
        (self.work / "submission.csv").write_text(self.csv)
        self.script.write_text("raise RuntimeError('failure')")
        receipt = self.run_model()
        self.assertFalse(receipt["accepted"])
        self.assertNotEqual(receipt["exit_code"], 0)
        self.script.write_text("print('no predictions')")
        self.assertFalse(self.run_model()["accepted"])
        self.assertEqual(evaluate(self.receipts)["completeness_score"], 0)

    def test_tampering_invalidates_archived_submission(self):
        receipt = self.run_model()
        archive = self.receipts / receipt["execution_id"]
        (archive / "submission.csv").write_text(self.csv.replace(",2", ",3"))
        result = evaluate(self.receipts)
        self.assertEqual(result["completeness_score"], 0)
        self.assertIn("hash mismatch", result["invalid_receipts"][0]["error"])

    def test_parent_and_path_guards(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.run_model("experiment", "missing")
        with self.assertRaises(ValueError):
            self.run_model("experiment", "../../escape")
        with self.assertRaises(ValueError):
            execute_experiment("../escape.py", "baseline", "", self.work, self.receipts, sys.executable)

    def test_timeout_is_preserved_as_rejected_receipt(self):
        self.script.write_text("import time\ntime.sleep(20)")
        receipt = self.run_model(timeout=0.1)
        self.assertFalse(receipt["accepted"])
        self.assertIn("TimeoutExpired", receipt["error"])
        self.assertIsNotNone(receipt["finished_at"])

    def test_label_free_invalid_predictions(self):
        pred = self.work / "pred.csv"
        for csv in [self.csv.replace(",2", ",-1"), self.csv.replace(",2", ",nan"),
                    self.csv.replace("a,s", "b,s"), self.csv + "a,s,2022-10-03,3\n",
                    "sku,store,week,sales_units\n"]:
            pred.write_text(csv)
            with self.subTest(csv=csv), self.assertRaises(ValueError):
                validate_submission(self.receipts / "test_keys.csv", pred)

    def test_portable_scoring_helper_uses_same_contract(self):
        package = Path(__file__).resolve().parents[1] / "MLAgentBench"
        shutil.copyfile(package / "submission_validation.py", self.work / "submission_validation.py")
        shutil.copyfile(package / "benchmarks/demand-forecasting/scripts/scoring.py", self.work / "metrics.py")
        spec = importlib.util.spec_from_file_location("portable_metrics_fixture", self.work / "metrics.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        (self.work / "actual.csv").write_text(self.csv)
        (self.work / "pred.csv").write_text(self.csv)
        self.assertEqual(module.score_files(self.work / "actual.csv", self.work / "pred.csv"), 0)
        (self.work / "pred.csv").write_text(self.csv.replace(",2", ",-2"))
        with self.assertRaises(ValueError):
            module.score_files(self.work / "actual.csv", self.work / "pred.csv")

    def test_uninstrumented_history_is_unknown(self):
        self.assertEqual(evaluate(self.work)["score_bounds"], [0, 100])
        self.assertIsNone(evaluate(self.work)["completeness_score"])

    def test_judge_only_receives_valid_pairs_and_checks_citations(self):
        judge = Mock()
        self.assertEqual(judge_pending(self.receipts, judge, "fixture"), {})
        judge.assert_not_called()
        baseline = self.run_model()
        experiment = self.run_model("experiment", baseline["execution_id"])
        judge.return_value = {"verdict": "pass", "reason": "Different model",
                              "evidence": [experiment["execution_id"] + "/sources/model.py:1"]}
        judgments = judge_pending(self.receipts, judge, "fixture")
        self.assertEqual(evaluate(self.receipts, judgments)["completeness_score"], 100)
        self.assertIn("files", judge.call_args.args[0])
        judge.return_value["evidence"] = ["invented.py:1"]
        judgments = judge_pending(self.receipts, judge, "fixture")
        self.assertEqual(evaluate(self.receipts, judgments)["experiment"], "unknown")

    def test_judge_failure_does_not_erase_verified_baseline(self):
        baseline = self.run_model()
        self.run_model("experiment", baseline["execution_id"])
        judgments = judge_pending(self.receipts, Mock(side_effect=RuntimeError("offline")), "fixture")
        self.assertEqual(evaluate(self.receipts, judgments)["score_bounds"], [50, 100])


if __name__ == "__main__":
    unittest.main()
