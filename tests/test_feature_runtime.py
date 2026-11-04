import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from MLAgentBench.provenance import Provenance, execution_command


class UnsandboxedRuntimeTests(unittest.TestCase):
    def test_ordinary_filesystem_access_and_feature_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / 'work'
            work.mkdir()
            (work / 'input.csv').write_text('series,time,value\na,2020-01-06,7\na,2020-01-13,\n')
            # A local module with the same name must not shadow the supplied one.
            (work / 'forecast_features.py').write_text("raise RuntimeError('shadowed')")
            script = work / 'run.py'
            script.write_text('''
from pathlib import Path
import pandas as pd
import forecast_features as ff
# A normal process may write outside cwd when its OS user has permission.
outside = Path.cwd().parent / 'outside_workspace.txt'
outside.write_text('ordinary Python execution')
engine = ff.demand_engine(pd.read_csv('input.csv'), {'value': {'lag': 'lag'}},
                          level=['series'], time_column='time', frequency='7D')
result = engine.build(mode='recursive')
assert result.frame['lag'].iloc[1] == 7
assert list(result.frame.columns) == ['series', 'time', 'value', 'lag']
result.save('features.csv')
''')
            tracker = Provenance(work, root / 'provenance')
            log = tracker.begin(0, 'Execute Script', {'script_name': 'run.py'})
            cmd, env = execution_command(sys.executable, script, work, log, tracker.root, protected_features=True)
            self.assertEqual(cmd[0], sys.executable)
            self.assertNotIn('bwrap', cmd)
            self.assertEqual(env['MLAB_EXECUTION_MODE'], 'unsandboxed')
            completed = subprocess.run(cmd, cwd=work, env=env, text=True, capture_output=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual((root / 'outside_workspace.txt').read_text(), 'ordinary Python execution')
            tracker.finish()
            graph = json.loads(tracker.file.read_text())
            output = graph['nodes'][graph['current']['features.csv']]
            dependencies = [graph['nodes'][n] for n in output['dependencies']]
            self.assertTrue(any(n['kind'] == 'trusted_utility' for n in dependencies))
            self.assertTrue(any(n['path'].startswith('.feature_audit/') for n in dependencies))
            self.assertTrue(any(n['path'] == 'input.csv' for n in dependencies))


if __name__ == '__main__':
    unittest.main()
