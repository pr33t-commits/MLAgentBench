import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from MLAgentBench.provenance import execution_command, resolve_execution_mode


class ExecutionModeTests(unittest.TestCase):
    def test_defaults_and_explicit_override(self):
        self.assertEqual(resolve_execution_mode("auto", True), "bubblewrap")
        self.assertEqual(resolve_execution_mode("auto", False), "unsandboxed")
        self.assertEqual(resolve_execution_mode("unsandboxed", True), "unsandboxed")
        self.assertEqual(resolve_execution_mode("bubblewrap", False), "bubblewrap")
        with self.assertRaises(ValueError):
            resolve_execution_mode("invalid")

    def test_unsandboxed_does_not_require_bwrap(self):
        with patch("MLAgentBench.provenance.shutil.which", side_effect=AssertionError):
            command, env = execution_command(sys.executable, "run.py", ".",
                                             execution_mode="unsandboxed")
        self.assertEqual(command, [sys.executable, "-u", "run.py"])
        self.assertEqual(env["MLAB_EXECUTION_MODE"], "unsandboxed")

    def test_bubblewrap_has_no_fallback(self):
        with patch("MLAgentBench.provenance.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "no unprotected fallback"):
                execution_command(sys.executable, "run.py", ".", execution_mode="bubblewrap")

    @unittest.skipUnless(os.name == "posix", "POSIX command construction")
    def test_explicit_bubblewrap_without_feature_engine(self):
        with patch("MLAgentBench.provenance.shutil.which", return_value="/usr/bin/bwrap"):
            command, env = execution_command(sys.executable, "run.py", ".", execution_mode="bubblewrap")
        self.assertEqual(command[0], "/usr/bin/bwrap")
        self.assertEqual(command[-3:], [sys.executable, "-u", "run.py"])
        self.assertIn(str(Path.cwd()), command)
        self.assertEqual(env["MLAB_EXECUTION_MODE"], "bubblewrap")
