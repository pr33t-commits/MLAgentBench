import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from MLAgentBench.forecast_features import demand_engine, Feature, FeatureEngine, SourcePolicy, iter_window_lineage


def custom(window):
    return {'range': max(window.values) - min(window.values), 'last': window.values[-1]}


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({'sku': ['a'] * 8 + ['b'] * 8,
            'time': list(pd.date_range('2024-01-01', periods=8, freq='D')) * 2,
            'sales': list(range(1, 9)) + list(range(11, 19)),
            'note': pd.Categorical(['kept'] * 16),
            'label': pd.array(list(range(16)), dtype='Int64')})

    def engine(self, transforms=None, data=None, **kwargs):
        return demand_engine(self.data if data is None else data,
            {'sales': {'previous': 'lag'}} if transforms is None else transforms,
            level=['sku'], time_column='time', frequency='1D', **kwargs)

    def test_recursive_retains_all_columns_dtypes_order_and_index(self):
        data = self.data.iloc[[7, 3, 0, 5, 2, 6, 1, 4]].copy()
        data.index = [8, 8, 9, 2, 3, 5, 7, 1]
        result = self.engine(data=data).build()
        pd.testing.assert_frame_equal(result.frame[data.columns], data)
        self.assertEqual(result.frame.previous.iloc[0], 7)
        self.assertTrue(np.isnan(result.frame.previous.iloc[2]))
        self.assertEqual(result.provenance['feature_columns'], ['previous'])
        self.assertNotIn('horizon', result.frame)

    def test_direct_mixed_anchors_and_retained_values(self):
        transforms = {'sales': {
            'target_mean': {'window': 2, 'offset': 3, 'anchor': 'target'},
            'recent_mean': {'window': 2, 'offset': 1, 'anchor': 'recent'}}}
        result = self.engine(transforms).build(mode='direct', horizons=[1, 2, 3])
        self.assertEqual(len(result.frame), 48)
        expected = self.data.iloc[np.repeat(np.arange(16), 3)].reset_index(drop=True)
        pd.testing.assert_frame_equal(result.frame[self.data.columns], expected)
        rows = result.frame[(result.frame.sku == 'a') & (result.frame.time == pd.Timestamp('2024-01-06'))]
        self.assertEqual(rows.horizon.tolist(), [1, 2, 3])
        self.assertEqual(rows.target_mean.tolist(), [2.5, 2.5, 2.5])
        self.assertEqual(rows.recent_mean.tolist(), [3.5, 2.5, 1.5])
        self.assertEqual(rows.sales.tolist(), [6, 6, 6])
        self.assertEqual(result.provenance['feature_columns'], ['target_mean', 'recent_mean'])

    def test_direct_nonconsecutive_horizons_order(self):
        result = self.engine({'sales': {'x': {'anchor': 'target', 'offset': 4}}}).build(
            mode='direct', horizons=[4, 1, 2])
        self.assertEqual(result.frame.horizon.iloc[:6].tolist(), [4, 1, 2, 4, 1, 2])

    def test_direct_rejects_invalid_offsets_before_calculation(self):
        for anchor in [None, 'target']:
            e = self.engine({'sales': {'x': {'anchor': anchor, 'offset': 2, 'min_periods': 1}}})
            with patch.object(e, '_calculate', side_effect=AssertionError('too late')):
                with self.assertRaises(ValueError):
                    e.build(mode='direct', horizons=[1, 3])
        e = self.engine({'sales': {'x': {'anchor': 'target', 'offset': 3}}})
        self.assertEqual(e.build(mode='direct', horizons=[3]).frame.x.iloc[5], 3)

    def test_horizons_required_validated_and_recursive_rejects_them(self):
        e = self.engine({'sales': {'x': {'anchor': 'target', 'offset': 5}}})
        for invalid in [None, [], [1, 1], [0], [-1], [True], [1.0], ['1'], '12', 3, {1, 2}]:
            with self.subTest(horizons=invalid), self.assertRaises(ValueError):
                e.build(mode='direct', horizons=invalid)
        with self.assertRaises(ValueError):
            e.build(horizons=[1])
        self.assertTrue(pd.api.types.is_integer_dtype(e.build(mode='direct', horizons=[np.int64(1)]).frame.horizon))

    def test_removed_api_rejected(self):
        with self.assertRaisesRegex(ValueError, 'lag parameter'):
            self.engine({'sales': {'x': {'lag': 1}}})
        for anchor in ['origin', 'auto', 'invalid']:
            with self.assertRaises(ValueError):
                self.engine({'sales': {'x': {'anchor': anchor}}})
        with self.assertRaises(ValueError):
            self.engine({'sales': {'x': {'anchor': 'recent'}}}).build()
        for kwargs in [{'predictions': {}}, {'origin': '2024-01-02'}, {'requests': self.data}]:
            with self.assertRaises(TypeError):
                self.engine().build(**kwargs)
        with self.assertRaises(TypeError):
            self.engine().build(self.data)
        with self.assertRaises(ValueError):
            self.engine(source_options={'sales': {'recursive_target': True}})

    def test_column_collisions_never_overwrite_input(self):
        for name in self.data.columns:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.engine({'sales': {name: 'mean'}})
        data = self.data.assign(horizon='keep me', origin='metadata')
        result = self.engine(data=data).build()
        pd.testing.assert_frame_equal(result.frame[data.columns], data)
        e = self.engine({'sales': {'x': {'anchor': 'target', 'offset': 3}}}, data=data)
        with self.assertRaisesRegex(ValueError, 'conflicts'):
            e.build(mode='direct', horizons=[1])
        with self.assertRaises(ValueError):
            self.engine({'sales': {'horizon': {'anchor': 'target'}}}).build(mode='direct', horizons=[1])

    def test_custom_multiple_columns_and_save_lineage(self):
        e = self.engine({'sales': {'spread': {'operation': custom, 'outputs': ['range', 'last'], 'window': 3}},
                         'label': {'label_mean': {'window': 2}}})
        result = e.build()
        self.assertEqual(result.frame.spread__range.iloc[3], 2)
        self.assertEqual(result.frame.spread__last.iloc[3], 3)
        self.assertEqual(result.frame.label_mean.iloc[3], 1.5)
        first_window = next(iter_window_lineage(result.provenance))
        self.assertEqual(first_window['target'], str(self.data.time.iloc[0]))
        self.assertNotIn('origin', first_window)
        self.assertIn('source_sha256', result.provenance['features'][0]['operation'])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'features.csv'
            result.save(path)
            meta = json.loads(path.with_suffix('.csv.lineage.json').read_text())
            self.assertEqual(meta['input_columns'], list(self.data.columns))
            self.assertEqual(meta['engine_version'], '6.0')

    def test_input_and_output_mutation_isolation(self):
        e = self.engine()
        self.data.sales = 999
        result = e.build()
        self.assertEqual(result.frame.previous.iloc[3], 3)
        result.frame.sales = -1
        self.assertEqual(e.build().frame.sales.iloc[0], 1)

    def test_offsets_windows_and_frequencies(self):
        for freq in ['1D', '7D']:
            data = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=80, freq=freq), 'value': range(80)})
            for width, offset in [(4, 53), (3, 9), (1, 7)]:
                e = demand_engine(data, {'value': {'x': {'window': width, 'offset': offset}}}, level=[], time_column='date', frequency=freq)
                result = e.build()
                end = 75 - offset
                self.assertEqual(result.frame.x.iloc[75], np.mean(range(end-width+1, end+1)))

    def test_missing_periods_and_partial_windows(self):
        data = self.data[self.data.time != pd.Timestamp('2024-01-03')]
        e = self.engine({'sales': {'full': {'window': 3}, 'partial': {'window': 3, 'min_periods': 2}}}, data=data)
        rows = e.build().frame
        row = rows[(rows.sku == 'a') & (rows.time == pd.Timestamp('2024-01-05'))].iloc[0]
        self.assertTrue(np.isnan(row.full))
        self.assertEqual(row.partial, 3)
        self.assertEqual(len(rows), len(data))
        zero = self.engine({'sales': {'full': {'window': 3}}}, data=data, source_options={'sales': {'missing': 'zero'}}).build().frame
        self.assertEqual(zero.loc[(zero.sku == 'a') & (zero.time == pd.Timestamp('2024-01-05')), 'full'].iloc[0], 2)

    def test_future_rows_use_only_supplied_history(self):
        data = self.data[self.data.sku == 'a'].copy()
        data.loc[data.time >= pd.Timestamp('2024-01-07'), 'sales'] = np.nan
        result = self.engine(data=data).build().frame
        self.assertEqual(result.previous.iloc[6], 6)
        self.assertTrue(np.isnan(result.previous.iloc[7]))
        # Caller updates a source value explicitly; a new engine uses it.
        data.loc[data.time == pd.Timestamp('2024-01-07'), 'sales'] = 100
        self.assertEqual(self.engine(data=data).build().frame.previous.iloc[7], 100)

    def test_direct_cutoff_blocks_more_recent_values(self):
        transforms = {'sales': {'target': {'offset': 3, 'anchor': 'target'},
                                'recent': {'offset': 1, 'anchor': 'recent'}}}
        original = self.engine(transforms).build(mode='direct', horizons=[3]).frame
        data = self.data.copy()
        data.loc[data.time >= pd.Timestamp('2024-01-06'), 'sales'] = 9999
        other = self.engine(transforms, data=data).build(mode='direct', horizons=[3]).frame
        self.assertEqual(original.target.iloc[7], other.target.iloc[7])
        self.assertEqual(original.recent.iloc[7], other.recent.iloc[7])

    def test_release_times_direct_recursive_and_known_future(self):
        data = self.data.assign(release=self.data.time + pd.Timedelta(days=1))
        for availability in ['observed', 'known_future']:
            options = {'sales': {'available_at': 'release', 'availability': availability}}
            self.assertTrue(np.isnan(self.engine(data=data, source_options=options).build().frame.previous.iloc[3]))
            e = self.engine({'sales': {'x': {'anchor': 'target', 'offset': 2}}}, data=data, source_options=options)
            self.assertTrue(np.isnan(e.build(mode='direct', horizons=[2]).frame.x.iloc[5]))
            with self.assertRaises(ValueError):
                e.build(mode='direct', horizons=[3])

    def test_empty_input_retains_schema(self):
        data = self.data.iloc[:0]
        recursive = self.engine(data=data).build().frame
        pd.testing.assert_frame_equal(recursive[data.columns], data)
        direct = self.engine({'sales': {'x': {'anchor': 'target', 'offset': 3}}}, data=data).build(mode='direct', horizons=[1, 3]).frame
        self.assertEqual(len(direct), 0)
        self.assertEqual(list(direct.columns), list(data.columns) + ['horizon', 'x'])

    def test_invalid_data_and_feature_configurations(self):
        for field in ['window', 'offset']:
            for value in [0, -1, True, 1.2, '1', None]:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    Feature('x', 'sales', **{field: value})
        with self.assertRaises(ValueError):
            Feature('x', 'sales', operation='wrong')
        for data in [pd.concat([self.data, self.data.iloc[:1]]), self.data.assign(sku=None), self.data.assign(sales=np.inf)]:
            with self.assertRaises(ValueError):
                self.engine(data=data)
        data = self.data.copy()
        data.loc[0, 'time'] += pd.Timedelta(hours=1)
        with self.assertRaisesRegex(ValueError, 'grid'):
            self.engine(data=data)
        with self.assertRaises(TypeError):
            self.engine(data='file.csv')

    def test_multiple_keys_and_global_aggregation(self):
        data = self.data.assign(channel='web')
        e = demand_engine(data, {'sales': {'x': 'lag'}}, level=['sku', 'channel'], time_column='time', frequency='1D')
        self.assertEqual(e.build().frame.x.iloc[9], 11)
        grouped = data.groupby('time', as_index=False).sales.sum()
        e = demand_engine(grouped, {'sales': {'x': 'lag'}}, level=[], time_column='time', frequency='1D')
        self.assertEqual(e.build().frame.x.iloc[1], 12)

    def test_coder_examples_execute_against_installed_engine(self):
        import sys
        import MLAgentBench.forecast_features as module
        examples = module.AGENT_GUIDE.split('Example: built-in feature specs with mixed direct anchors:', 1)[1]
        snippets = [part.split('```', 1)[0] for part in examples.split('```python\n')[1:]]
        namespace = {}
        with patch.dict(sys.modules, {'forecast_features': module}):
            exec(snippets[0], namespace)
            direct = namespace['result'].frame
            self.assertEqual(len(direct), 18)
            self.assertEqual(direct.target_mean.iloc[-3:].tolist(), [25, 25, 25])
            self.assertEqual(direct.recent_mean.iloc[-3:].tolist(), [35, 25, 15])
            exec(snippets[1], namespace)
            self.assertEqual(namespace['result'].frame.recent_mean.iloc[-1], 45)

    def test_timezone_aware_input_and_nullable_values(self):
        data = self.data.copy()
        data.time = data.time.dt.tz_localize('Asia/Kolkata')
        data.sales = data.sales.astype('Int64')
        data.loc[0, 'sales'] = pd.NA
        result = self.engine(data=data).build()
        pd.testing.assert_frame_equal(result.frame[data.columns], data)
        self.assertTrue(np.isnan(result.frame.previous.iloc[1]))


if __name__ == '__main__':
    unittest.main()
