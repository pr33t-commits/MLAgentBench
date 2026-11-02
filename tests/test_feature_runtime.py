import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from MLAgentBench.provenance import Provenance, execution_command


@unittest.skipUnless(sys.platform == 'linux' and shutil.which('bwrap'), 'Linux bubblewrap required')
class ProtectedRuntimeTests(unittest.TestCase):
    def test_read_only_library_and_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / 'work'
            work.mkdir()
            (work / 'input.csv').write_text('series,time,value\na,2020-01-06,7\na,2020-01-13,\n')
            # A local module with the same name must not shadow the supplied one.
            (work / 'forecast_features.py').write_text("raise RuntimeError('shadowed')")
            script = work / 'run.py'
            script.write_text('''import errno
from pathlib import Path
import pandas as pd
import forecast_features as ff
try:
    with open(ff.__file__, 'a') as handle:
        handle.write('# altered')
except OSError as exc:
    assert exc.errno in (errno.EROFS, errno.EACCES), exc
else:
    raise AssertionError('utility was writable')
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
            completed = subprocess.run(cmd, cwd=work, env=env, text=True, capture_output=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            tracker.finish()
            graph = json.loads(tracker.file.read_text())
            output = graph['nodes'][graph['current']['features.csv']]
            dependencies = [graph['nodes'][n] for n in output['dependencies']]
            self.assertTrue(any(n['kind'] == 'trusted_utility' for n in dependencies))
            self.assertTrue(any(n['path'].startswith('.feature_audit/') for n in dependencies))
            self.assertTrue(any(n['path'] == 'input.csv' for n in dependencies))


if __name__ == '__main__':
    unittest.main()
