import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest

from MLAgentBench.execution_runtime import run_script, SCRIPT_TIMEOUT


class ExecutionRuntimeTests(unittest.TestCase):
    def run_code(self, code, timeout=0.3, **kwargs):
        with tempfile.TemporaryDirectory() as work:
            return run_script([sys.executable, "-u", "-c", code], work, dict(os.environ),
                              timeout=timeout, **kwargs)

    def test_partial_output_without_newline_and_timeout(self):
        result = self.run_code("import time; print('partial', end='', flush=True); time.sleep(30)")
        self.assertEqual(result['stdout'], 'partial')
        self.assertEqual(result['timeout_reason'], 'script_timeout')
        self.assertEqual(result['timeout_stage'], 'outside engine.build()')
        self.assertEqual(SCRIPT_TIMEOUT, 600)

    def test_real_build_timeout_and_child_cleanup(self):
        code = """import os, sys, time, subprocess
import pandas as pd
from MLAgentBench.forecast_features import demand_engine
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
print('child=' + str(child.pid), flush=True)
def slow(window):
    time.sleep(30)
    return 1
history = pd.DataFrame({'sku': ['a', 'a'], 'week': ['2020-01-01', '2020-01-02'], 'sales': [1, 2]})
engine = demand_engine(history, {'sales': {'slow': slow}}, level=['sku'], time_column='week', frequency='1D')
engine.build(mode='recursive')
"""
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        with tempfile.TemporaryDirectory() as work:
            result = run_script([sys.executable, '-u', '-c', code], work, env, timeout=3)
        self.assertEqual(result['timeout_stage'], 'engine.build()', result)
        self.assertIn('engine.build()', result['feedback'])
        self.assertEqual(result['active_builds'][0]['input_rows'], 2)
        pid = int(result['stdout'].split('child=')[1].splitlines()[0])
        stat = Path('/proc') / str(pid) / 'stat'
        if stat.exists():
            self.assertEqual(stat.read_text().split()[2], 'Z')

    def test_success_and_stderr_preserved(self):
        result = self.run_code("import sys; print('done'); print('diagnostic', file=sys.stderr)")
        self.assertTrue(result['ok'])
        self.assertIn('diagnostic', result['stderr'])

    def test_session_budget_reported_separately(self):
        import time
        result = self.run_code('import time; time.sleep(30)', timeout=3,
                               session_deadline=time.time() + 0.2)
        self.assertEqual(result['timeout_reason'], 'session_timeout')
        self.assertEqual(result['session_remaining_after_seconds'], 0)

    def test_completed_build_is_not_active(self):
        start = '[MLAB_BUILD] ' + json.dumps(dict(event='start', call_id='one'))
        end = '[MLAB_BUILD] ' + json.dumps(dict(event='complete', call_id='one', elapsed_seconds=0.1))
        result = self.run_code('import sys,time; print(' + repr(start) + ', file=sys.stderr); print(' + repr(end) + ', file=sys.stderr); time.sleep(30)')
        self.assertEqual(result['timeout_stage'], 'outside engine.build()')
        self.assertEqual(len(result['build_timings']), 2)
