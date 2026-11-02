"""Equivalence against the retained scalar evaluator of the current API."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from MLAgentBench.forecast_features import demand_engine, iter_window_lineage


def custom(window):
    values = np.asarray(window.values)
    return {'first': values[0], 'last': values[-1]}


def reference(engine, data, mode, horizons):
    active = horizons if mode == 'direct' else [None]
    frame = data.iloc[np.repeat(np.arange(len(data)), len(active))].copy()
    if mode == 'direct':
        frame = frame.reset_index(drop=True)
        frame['horizon'] = np.tile(horizons, len(data))
    records, lineage = [], []
    policy = next(iter(engine.policies.values()))
    for row in data.to_dict('records'):
        series = tuple(row[k] for k in policy.keys)
        target = pd.Timestamp(row[policy.time_column])
        for horizon in active:
            record = {}
            for feature in engine.features:
                window = engine.window(feature, series, target, mode=mode, horizon=horizon)
                record.update(engine._calculate(feature, window))
                lineage.append(dict(feature=feature.name, series=list(series), target=str(target),
                                    horizon=horizon, anchor=feature.anchor or 'target',
                                    periods=[str(p) for p in window.periods],
                                    available=list(window.available_mask),
                                    observed=list(window.observed_mask), missing=list(window.missing_mask)))
            records.append(record)
    for feature in engine.features:
        names = [feature.name + '__' + name for name in feature.outputs] if feature.outputs else [feature.name]
        for name in names:
            frame[name] = np.array([row[name] for row in records], dtype=float)
    return frame, lineage


class CurrentFeatureTests(unittest.TestCase):
    def data(self, timezone=None):
        dates = pd.date_range('2024-03-07', periods=12, freq='D', tz=timezone)
        # Engine frequency is elapsed time, so include DST without changing the grid.
        dates = dates[0] + np.arange(12) * pd.Timedelta('1D')
        rng = np.random.default_rng(729)
        data = pd.DataFrame({'series': ['a'] * 12 + ['b'] * 12,
                             'time': list(dates) * 2, 'value': rng.normal(size=24)})
        data.loc[[0, 4, 18], 'value'] = np.nan
        data.loc[[3, 15], 'value'] = 0.0
        data['other'] = data.value * 2
        return data.drop(index=[2, 16]).sample(frac=1, random_state=4)

    def check(self, data, specs, mode='direct', horizons=None, options=None, level=None):
        horizons = [3, 1, 2] if horizons is None else horizons
        engine = demand_engine(data, specs, level=['series'] if level is None else level,
                               time_column='time', frequency='1D', source_options=options)
        expected, lineage = reference(engine, data, mode, horizons)
        result = engine.build(mode=mode, **({'horizons': horizons} if mode == 'direct' else {}))
        pd.testing.assert_frame_equal(result.frame, expected, check_exact=False, rtol=1e-10, atol=1e-10)
        serialized = json.loads(json.dumps(result.provenance))
        self.assertEqual(list(iter_window_lineage(serialized)), lineage)
        self.assertEqual(list(iter_window_lineage({'windows': lineage})), lineage)
        return result

    def test_all_builtins_modes_anchors_and_missing_policies(self):
        ops = ['lag', 'mean', 'sum', 'median', 'std', 'min', 'max', 'quantile',
               'nonzero_count', 'nonzero_fraction']
        for mode, anchors in [('recursive', [None, 'target']), ('direct', ['target', 'recent'])]:
            for missing in ['nan', 'zero']:
                for availability in ['observed', 'known_future']:
                    with self.subTest(mode=mode, missing=missing, availability=availability):
                        specs = {f'f_{op}_{anchor}_{minimum}': dict(operation=op, window=4,
                                 offset=3 if anchor == 'target' else 1, anchor=anchor,
                                 min_periods=minimum, quantile=.3)
                                 for op in ops for anchor in anchors for minimum in [None, 1, 3]}
                        self.check(self.data(), {'value': specs}, mode,
                                   options={'value': dict(missing=missing, availability=availability)})

    def test_callbacks_release_times_and_multiple_columns(self):
        data = self.data()
        data['release'] = data.time + pd.Timedelta('2D')
        data.loc[data.index[:3], 'release'] = pd.NaT
        specs = {'value': {'mean': dict(operation='mean', window=3, anchor='recent'),
                           'custom': dict(operation=custom, window=2, anchor='target', offset=3,
                                          outputs=['first', 'last'], min_periods=1)},
                 'other': {'released': dict(operation='lag', window=2, anchor='recent', min_periods=1)}}
        for availability in ['observed', 'known_future']:
            self.check(data, specs, options={'other': dict(available_at='release', missing='zero', availability=availability)})

    def test_timezone_and_global_series(self):
        self.check(self.data('America/New_York'), {'value': {'mean': dict(window=4, offset=3, anchor='target', min_periods=1)}})
        data = self.data().query('series == "a"').drop(columns='series')
        self.check(data, {'value': {'lag': dict(operation='lag', window=4, min_periods=1)}}, mode='recursive', level=[])

    def test_cache_reuses_statistics_across_offsets_and_horizons(self):
        data = self.data()
        specs = {'value': {'first': dict(operation='mean', window=4, anchor='recent'),
                           'second': dict(operation='mean', window=4, offset=3, anchor='target')}}
        import MLAgentBench.forecast_features as ff
        with patch.object(ff, '_rolling_builtin', wraps=ff._rolling_builtin) as rolling:
            self.check(data, specs)
        self.assertEqual(rolling.call_count, 2)  # One calculation for each of two series.

    def test_save_and_audit_json(self):
        import os
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'MLAB_FEATURE_AUDIT_DIR': directory}):
            result = self.check(self.data(), {'value': {'mean': dict(window=3)}}, mode='recursive')
            path = Path(directory) / 'features.csv'
            result.save(path)
            saved = json.loads(path.with_suffix('.csv.lineage.json').read_text())
            self.assertEqual(list(iter_window_lineage(saved)), list(iter_window_lineage(result.provenance)))
            self.assertEqual(len(list(Path(directory).glob('features_*.json'))), 1)

    def test_large_offset_small_variation_statistics(self):
        data = self.data().sort_values(['series', 'time'])
        data['value'] = 1e12 + np.arange(len(data)) * .001
        self.check(data, {'value': {'std': dict(operation='std', window=4, min_periods=1),
                                   'mean': dict(operation='mean', window=4, min_periods=1)}}, mode='recursive')

    def test_sparse_grid_falls_back_and_empty_input(self):
        data = pd.DataFrame({'series': ['a', 'a'], 'time': pd.to_datetime(['1900-01-01', '2180-01-01']), 'value': [2., 3.]})
        self.check(data, {'value': {'mean': dict(window=2)}}, mode='recursive')
        self.check(self.data().iloc[:0], {'value': {'mean': dict(window=2)}}, mode='recursive')


if __name__ == '__main__':
    unittest.main()
