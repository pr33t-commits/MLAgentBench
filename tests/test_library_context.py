import json
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from MLAgentBench.library_context import build_library_context


class LibraryContextTests(unittest.TestCase):
    def test_selected_interpreter_and_import_failures(self):
        data = {"libraries": {"sklearn": {"available": True, "version": "test"},
                              "lightgbm": {"available": False, "error": "ImportError"}}}
        with patch("MLAgentBench.library_context.subprocess.run", return_value=SimpleNamespace(
                returncode=0, stdout="LIBRARY_CONTEXT=" + json.dumps(data))) as run:
            result = build_library_context("/sandbox/bin/python")
        self.assertEqual(run.call_args.args[0][0], "/sandbox/bin/python")
        self.assertIn('"available": false', result)
        self.assertIn("scikit-learn", result)

    def test_timeout_does_not_claim_availability(self):
        with patch("MLAgentBench.library_context.subprocess.run", side_effect=subprocess.TimeoutExpired("python", 60)):
            self.assertIn("could not be verified", build_library_context("python"))
