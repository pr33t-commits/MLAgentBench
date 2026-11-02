"""Post-run scoring against synthetic hidden labels; no model/API calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


PACKAGE = Path(__file__).resolve().parents[1] / "MLAgentBench"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunnerMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Runner's model and Linux environment imports are irrelevant to scoring.
        modules = {}
        for name, attributes in {
            "MLAgentBench.LLM": [],
            "MLAgentBench.environment": ["Environment"],
            "MLAgentBench.agents.agent": ["Agent", "SimpleActionAgent", "ReasoningActionAgent"],
            "MLAgentBench.agents.agent_research": ["ResearchAgent"],
            "MLAgentBench.agents.agent_langchain": ["LangChainAgent"],
            "MLAgentBench.agents.agent_autogpt": ["AutoGPTAgent"],
        }.items():
            modules[name] = types.ModuleType(name)
            for attribute in attributes:
                setattr(modules[name], attribute, Mock())
        with patch.dict(sys.modules, modules):
            cls.runner = load_file("_runner_metrics_test", PACKAGE / "runner.py")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.receipts = self.root / "logs" / "completeness"
        self.receipts.mkdir(parents=True)
        self.work = self.root / "work"
        self.work.mkdir()
        self.env = types.SimpleNamespace(
            static_kwargs_for_tools={"receipt_dir": str(self.receipts)},
            benchmark_folder_name="demand-forecasting",
            args=types.SimpleNamespace(task="demand-forecasting"),
            log_dir=str(self.receipts.parent), work_dir=str(self.work))
        # Use the actual task evaluator, with only its hidden labels replaced.
        scorer_dir = self.root / "scorer"
        scorer_dir.mkdir()
        for name in ("eval.py", "scoring.py"):
            shutil.copyfile(PACKAGE / "benchmarks/demand-forecasting/scripts" / name,
                            scorer_dir / name)
        shutil.copyfile(PACKAGE / "submission_validation.py", scorer_dir / "submission_validation.py")
        self.header = "sku,store,week,sales_units\n"
        (scorer_dir / "answer.csv").write_text(
            self.header + "a,s,2022-10-03,2\nb,s,2022-10-03,6\n", encoding="utf-8")
        self.scorer = load_file("_runner_fixture_eval", scorer_dir / "eval.py")
        original_import = self.runner.importlib.import_module
        def import_module(name, *args, **kwargs):
            if name == "MLAgentBench.benchmarks.demand-forecasting.scripts.eval":
                return self.scorer
            return original_import(name, *args, **kwargs)
        patcher = patch.object(self.runner.importlib, "import_module", side_effect=import_module)
        self.importer = patcher.start()
        self.addCleanup(patcher.stop)

    def register(self, execution_id, values=(2, 6), role="experiment", accepted=True):
        folder = self.receipts / execution_id
        folder.mkdir()
        # Reversed rows also verify that the trusted evaluator aligns keys.
        prediction = folder / "submission.csv"
        prediction.write_text(self.header + f"b,s,2022-10-03,{values[1]}\na,s,2022-10-03,{values[0]}\n")
        receipt = {"execution_id": execution_id, "role": role,
                   "parent_submission_id": "" if role == "baseline" else "exec_base",
                   "script_name": execution_id + ".py", "accepted": accepted,
                   "artifact_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest()}
        (folder / "receipt.json").write_text(json.dumps(receipt))
        return prediction

    def test_scores_baseline_and_multiple_experiments_by_id(self):
        self.register("exec_base", values=(0, 0), role="baseline")
        chosen = self.register("exec_first", values=(4, 6))
        self.register("exec_second")
        shutil.copyfile(chosen, self.work / "submission.csv")
        report = self.runner.record_test_metrics(self.env)
        scores = report["experiments"]
        self.assertEqual({key: row["test_score"] for key, row in scores.items()},
                         {"exec_base": 1.0, "exec_first": .25, "exec_second": 0.0})
        self.assertEqual(scores["exec_first"]["parent_submission_id"], "exec_base")
        self.assertEqual(report["selected_submission"]["execution_ids"], ["exec_first"])
        self.assertEqual(report["selected_submission"]["test_score"], .25)
        self.assertEqual(report["metric"], "WMAPE")
        self.assertEqual(json.loads((self.receipts.parent / "test_metrics.json").read_text()), report)
        self.importer.assert_any_call("MLAgentBench.benchmarks.demand-forecasting.scripts.eval")

    def test_rejected_corrupt_and_modified_archives_do_not_hide_good_scores(self):
        self.register("exec_base", role="baseline")
        self.register("exec_rejected", accepted=False)
        changed = self.register("exec_changed")
        changed.write_text(self.header)
        self.register("exec_corrupt")
        (self.receipts / "exec_corrupt/receipt.json").write_text("invalid JSON")
        self.register("exec_invalid", values=(-1, 6))
        scores = self.runner.record_test_metrics(self.env)["experiments"]
        self.assertEqual(scores["exec_base"]["test_score"], 0)
        self.assertEqual(scores["exec_rejected"]["status"], "skipped")
        for key in ("exec_changed", "exec_corrupt", "exec_invalid"):
            self.assertEqual(scores[key]["status"], "error")
            self.assertIsNone(scores[key]["test_score"])
            self.assertTrue(scores[key]["error"])

    def test_evaluator_failure_and_nonfinite_score_are_recorded(self):
        self.register("exec_base", role="baseline")
        for value in (float("nan"), float("inf")):
            with patch.object(self.scorer, "get_score", return_value=value):
                row = self.runner.record_test_metrics(self.env)["experiments"]["exec_base"]
                self.assertEqual(row["status"], "error")
                self.assertIsNone(row["test_score"])
        self.importer.side_effect = ImportError("scorer unavailable")
        row = self.runner.record_test_metrics(self.env)["experiments"]["exec_base"]
        self.assertIn("scorer unavailable", row["error"])

    def test_identical_archives_keep_all_ids_and_unregistered_final_is_unmatched(self):
        chosen = self.register("exec_base", role="baseline")
        self.register("exec_same")
        shutil.copyfile(chosen, self.work / "submission.csv")
        selected = self.runner.record_test_metrics(self.env)["selected_submission"]
        self.assertEqual(selected["execution_ids"], ["exec_base", "exec_same"])
        (self.work / "submission.csv").write_text(self.header)
        selected = self.runner.record_test_metrics(self.env)["selected_submission"]
        self.assertEqual(selected["status"], "unmatched")
        self.assertIsNone(selected["test_score"])

    def test_empty_and_uninstrumented_runs(self):
        report = self.runner.record_test_metrics(self.env)
        self.assertEqual(report["experiments"], {})
        self.assertEqual(report["selected_submission"]["status"], "missing")
        self.env.static_kwargs_for_tools = {}
        self.assertIsNone(self.runner.record_test_metrics(self.env))

    def test_runner_scores_after_environment_exit_even_when_agent_fails(self):
        for failure in (None, RuntimeError("agent failed")):
            events = []
            env = Mock()
            env.__enter__ = Mock(side_effect=lambda: (events.append("enter"), env)[1])
            env.__exit__ = Mock(side_effect=lambda *args: events.append("exit"))
            env.get_task_description.return_value = ("task", "demand-forecasting")
            env.low_level_actions = env.high_level_actions = []
            env.save.side_effect = lambda step: events.append("save")
            agent = Mock()
            agent.run.side_effect = failure
            with patch.object(self.runner, "Environment", return_value=env), \
                 patch.object(self.runner, "record_test_metrics", side_effect=lambda e: events.append("score")):
                if failure:
                    with self.assertRaisesRegex(RuntimeError, "agent failed"):
                        self.runner.run(Mock(return_value=agent), Mock())
                else:
                    self.runner.run(Mock(return_value=agent), Mock())
            self.assertEqual(events, ["enter", "exit", "save", "score"])


if __name__ == "__main__":
    unittest.main()
