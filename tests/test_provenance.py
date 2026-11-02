"""Real staged executions demonstrate versioned lineage without an LLM."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from MLAgentBench.completeness import execute_experiment
from MLAgentBench.provenance import Provenance, execution_command, review_bundle, archive_preparation


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / "workspace"
        self.work.mkdir()
        (self.work / "raw.csv").write_text("value\n3\n")
        (self.work / "test.csv").write_text("sku,store,week\na,s,w\n")
        (self.work / "TASK.md").write_text("Predict using historical inputs only.")
        self.receipts = self.root / "env_log/completeness"
        self.receipts.mkdir(parents=True)
        shutil.copyfile(self.work / "test.csv", self.receipts / "test_keys.csv")
        self.tracker = Provenance(self.work, self.root / "env_log/provenance")
        self.index = 0

    def write(self, name, text):
        self.tracker.begin(self.index, "Write File", {"file_name": name, "content": text})
        (self.work / name).write_text(text)
        self.tracker.finish()
        self.index += 1

    def execute(self, name):
        log = self.tracker.begin(self.index, "Execute Script", {"script_name": name})
        command, env = execution_command(sys.executable, self.work / name, self.work, log, self.tracker.root)
        result = subprocess.run(command, cwd=self.work, env=env, capture_output=True, text=True)
        self.tracker.finish()
        self.index += 1
        self.assertEqual(result.returncode, 0, result.stderr)

    def submit(self, name):
        log = self.tracker.begin(self.index, "Execute Experiment", {"script_name": name})
        r = execute_experiment(name, "baseline", "", self.work, self.receipts, sys.executable,
                               provenance_log=log, provenance_dir=self.tracker.root)
        self.assertTrue(r["accepted"], r)
        self.tracker.finish(self.receipts, json.dumps(r))
        self.index += 1
        return r["execution_id"]

    def test_raw_to_features_to_model_to_submission_and_all_edits(self):
        self.write("features.py", "# first draft\n") # 0
        self.write("features.py", "from pathlib import Path\nPath('features.json').write_text(Path('raw.csv').read_text().splitlines()[1])") # 1
        self.execute("features.py") # 2
        self.write("fit.py", "from pathlib import Path\nPath('model.bin').write_bytes(Path('features.json').read_bytes())") # 3
        self.execute("fit.py") # 4
        self.write("predict.py", "import os\nfrom pathlib import Path\nv=Path('model.bin').read_text()\n"
                   "(Path(os.environ['EXPERIMENT_OUTPUT_DIR'])/'submission.csv').write_text('sku,store,week,sales_units\\na,s,w,'+v+'\\n')") # 5
        execution_id = self.submit("predict.py") # 6
        before = review_bundle(self.tracker.root, execution_id)
        self.assertEqual(before["step_indices"], list(range(7)))
        self.assertEqual(before["integrity_errors"], [])
        paths = {n["path"] for n in before["nodes"].values()}
        self.assertTrue({"raw.csv", "features.py", "features.json", "fit.py", "model.bin", "predict.py"} <= paths)
        self.assertEqual(len([n for n in before["nodes"].values() if n["path"] == "features.py"]), 2)
        receipt = json.loads((self.receipts / execution_id / "receipt.json").read_text())
        self.assertEqual(receipt["provenance"]["step_index"], 6)
        bundle_path = self.root / "review"
        result = subprocess.run([sys.executable, "-m", "MLAgentBench.provenance", str(self.tracker.root),
                                 "--submission", execution_id, "--output", str(bundle_path)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((bundle_path / "review.json").read_text())["step_indices"], list(range(7)))
        self.assertTrue((bundle_path / "executions" / execution_id / "stdout.txt").is_file())
        self.write("features.py", "# later unrelated edit")
        self.assertEqual(review_bundle(self.tracker.root, execution_id), before)

    def test_deleted_intermediate_read_is_archived_and_linked(self):
        self.write("stage.py", "from pathlib import Path\np=Path('scratch.txt')\np.write_text(Path('raw.csv').read_text())\n"
                   "Path('kept.txt').write_text(p.read_text())\np.unlink()")
        self.execute("stage.py")
        nodes = self.tracker.graph["nodes"].values()
        scratch = next(n for n in nodes if n["path"] == "scratch.txt")
        self.assertTrue((self.tracker.root / scratch["object"]).exists())
        self.assertEqual(scratch["step_index"], 1)

    def test_same_bytes_rewrite_and_final_copy_are_in_history(self):
        self.write("submission.csv", "sku,store,week,sales_units\na,s,w,3\n")
        self.write("submission.csv", "sku,store,week,sales_units\na,s,w,3\n")
        self.tracker.begin(2, "Final Answer", {"final_answer": "done"})
        self.tracker.finish()
        self.assertEqual(review_bundle(self.tracker.root, "final")["step_indices"], [0, 1, 2])

    def test_uninstrumented_execution_is_not_reported_as_full_lineage(self):
        self.tracker.begin(0, "Python REPL", {"command": "unknown computation"})
        (self.work / "submission.csv").write_text("sku,store,week,sales_units\na,s,w,3\n")
        self.tracker.finish()
        self.tracker.begin(1, "Final Answer", {})
        self.tracker.finish()
        report = review_bundle(self.tracker.root, "final")
        self.assertIn("no Python file-access instrumentation", report["steps"][0]["coverage_gaps"][0])
        self.assertEqual(report["correctness_verdict"], "not_reviewed")

    def test_integrity_checks_and_declared_raw_sources(self):
        self.write("submission.csv", "sku,store,week,sales_units\na,s,w,3\n")
        self.tracker.begin(1, "Final Answer", {})
        self.tracker.finish()
        report = review_bundle(self.tracker.root, "final")
        node = report["nodes"][report["submission"]["node"]]
        (self.tracker.root / node["object"]).write_text("tampered")
        self.assertIn("hash mismatch", review_bundle(self.tracker.root, "final")["integrity_errors"][0])

    def test_preparation_archive_links_inputs_and_code(self):
        import hashlib
        prepared = self.root / "prepared"
        prepared.mkdir()
        shutil.copyfile(self.work / "raw.csv", prepared / "train.csv")
        code = self.root / "prepare.py"
        code.write_text("# fixture preparation code")
        archive = self.root / "preparation_archive"
        archive_preparation(self.work, prepared, [code], archive,
                            {"raw.csv": hashlib.sha256((self.work / "raw.csv").read_bytes()).hexdigest()})
        tracker = Provenance(prepared, self.root / "prepared_provenance", preparation_dir=archive)
        node = tracker.graph["nodes"][tracker.graph["current"]["train.csv"]]
        parents = [tracker.graph["nodes"][key] for key in node["dependencies"]]
        self.assertEqual({n["kind"] for n in parents}, {"raw_data", "preparation_code"})
        for parent in parents:
            self.assertTrue((tracker.root / parent["object"]).is_file())

    def test_environment_wrapper_records_step_even_when_execution_raises(self):
        import ast
        from types import SimpleNamespace
        source = Path(__file__).resolve().parents[1] / "MLAgentBench/environment.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Environment")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        def failure(action):
            (self.work / "partial.txt").write_text("failed stage left this artifact")
            raise RuntimeError("interrupted")
        env = SimpleNamespace(_provenance=self.tracker, _trace=SimpleNamespace(steps=[]),
                              _static_kwargs_for_tools={}, _execute=failure)
        with self.assertRaises(RuntimeError):
            namespace["execute"](env, SimpleNamespace(name="Execute Script", args={}))
        self.assertIn("partial.txt", self.tracker.graph["current"])
        self.assertEqual(self.tracker.graph["steps"][0]["step_index"], 0)
        self.assertNotIn("provenance_log", env._static_kwargs_for_tools)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Legacy Execute Script uses POSIX pipe selectors")
    def test_actual_execute_script_action_captures_intermediate_reads(self):
        from MLAgentBench.low_level_actions import execute_script
        from types import SimpleNamespace
        self.write("stage.py", "from pathlib import Path\nPath('derived.csv').write_text(Path('raw.csv').read_text())")
        log = self.tracker.begin(1, "Execute Script", {"script_name": "stage.py"})
        observation = execute_script("stage.py", work_dir=str(self.work), python=sys.executable,
                                     device="", trace=SimpleNamespace(low_level_steps=[]),
                                     provenance_log=log, provenance_dir=str(self.tracker.root))
        self.assertEqual(json.loads(observation)["exit_code"], 0)
        self.tracker.finish(observation=observation)
        node = self.tracker.graph["nodes"][self.tracker.graph["current"]["derived.csv"]]
        paths = {self.tracker.graph["nodes"][key]["path"] for key in node["dependencies"]}
        self.assertTrue({"raw.csv", "stage.py"} <= paths)


if __name__ == "__main__":
    unittest.main()
