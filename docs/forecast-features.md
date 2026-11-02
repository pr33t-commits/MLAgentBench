# Forecast feature engine (v5)

The engine uses only the DataFrame passed to `demand_engine` at instantiation.
Each input row's time is its target time. There is no request DataFrame, origin,
separate prediction buffer or internal model call. The engine keeps a copy.

## Main-agent planning guide

demand_engine is a feature-building utility for forecasting models.
It calculates shifted values, rolling statistics and custom summaries from the
DataFrame supplied at instantiation. Both modes use that data only. It does not
train models, generate predictions or automatically populate future values.

What it can calculate:
- Shifted values; rolling mean, sum, median, population standard deviation,
  min, max, quantiles, nonzero counts and nonzero fractions.
- Windows shifted into older history, including seasonal windows.
- Custom calculations with one or several outputs, across numeric columns at
  explicitly chosen grouping levels such as SKU, SKU/store or region.

What inputs to plan:
1. One DataFrame with grouping columns, a time column and numeric input values.
   Each group/time row is a target row. Aggregate explicitly when necessary.
   Include future target rows with unavailable values left missing when needed.
2. Named transformations per input column: operation, window and offset.
   Offset is the total backward shift; there is no lag parameter.
3. Explicit grouping columns, time column and fixed frequency, e.g. 1D or 7D.
4. For direct mode, an explicit list of positive integer horizons and an anchor
   on each feature: target or recent. No requests DataFrame or origin is needed.
5. Data availability/release policies, split boundaries and missing-value rules.

How modes and anchors work:
- Recursive: one output row per input row. Every feature anchors at that row's
  target time and looks backward by offset, then spans window periods backwards.
  The mode name does not imply a forecasting loop: only supplied data is used.
- Direct: each input target row is expanded across the requested horizons.
  A target-anchored feature looks backward from the target; its offset must be
  at least the maximum requested horizon. Equality is valid.
  A recent-anchored feature starts from target minus horizon periods, then
  applies offset and window. Different features may use different anchors.
  With offset=1, recent's latest window value is target minus horizon minus 1.
  Offset/window count time periods, not available rows; missing periods do not
  compress the grid. Historical offsets are positive integers, default 1.

Expected output and downstream planning:
All original columns are retained and features appended. Direct additionally
adds horizon. Input order is retained; each direct row uses the supplied horizon
order. The original time column remains the target time. Short history gives
NaN according to min_periods; rows are not automatically removed.
Use the recorded feature_columns list for model inputs. Retained sales/labels,
metadata and other original columns are not automatically safe model features.
Plan holdouts and any iterative forecasting outside the engine. Never populate
unknown forecast periods with validation/test actuals. Instantiate a new engine
after explicitly updating an input frame, since the engine keeps a copy.

When delegating to Create Script (AI), set use_demand_engine=true for feature
creation. Pass input aliases, schemas/date formats, grouping/aggregation,
frequency, transformations, offsets, per-feature direct anchors, direct horizons,
source policies, split roles, output paths and acceptance checks. The coder has
no prior conversation and receives a separate implementation guide.
Keep historical feature generation, fitting and submission in separate stages;
register the baseline first. Set the flag false for stages that only consume
features, fit models, inspect data or plot. You remain responsible for validation
coverage, fit boundaries and preprocessing; retained columns must be selected
deliberately. Inspect the trusted utility if needed via Inspect Script Lines at
@trusted/forecast_features.py, at most 100 lines per call.

## Coder usage guide

demand_engine is a feature-building utility for forecasting models.
It calculates shifted values, rolling statistics and custom summaries from the
DataFrame supplied at instantiation. Both modes use that data only. It does not
train models, generate predictions or automatically populate future values.

What it can calculate:
- Shifted values; rolling mean, sum, median, population standard deviation,
  min, max, quantiles, nonzero counts and nonzero fractions.
- Windows shifted into older history, including seasonal windows.
- Custom calculations with one or several outputs, across numeric columns at
  explicitly chosen grouping levels such as SKU, SKU/store or region.

What inputs to plan:
1. One DataFrame with grouping columns, a time column and numeric input values.
   Each group/time row is a target row. Aggregate explicitly when necessary.
   Include future target rows with unavailable values left missing when needed.
2. Named transformations per input column: operation, window and offset.
   Offset is the total backward shift; there is no lag parameter.
3. Explicit grouping columns, time column and fixed frequency, e.g. 1D or 7D.
4. For direct mode, an explicit list of positive integer horizons and an anchor
   on each feature: target or recent. No requests DataFrame or origin is needed.
5. Data availability/release policies, split boundaries and missing-value rules.

How modes and anchors work:
- Recursive: one output row per input row. Every feature anchors at that row's
  target time and looks backward by offset, then spans window periods backwards.
  The mode name does not imply a forecasting loop: only supplied data is used.
- Direct: each input target row is expanded across the requested horizons.
  A target-anchored feature looks backward from the target; its offset must be
  at least the maximum requested horizon. Equality is valid.
  A recent-anchored feature starts from target minus horizon periods, then
  applies offset and window. Different features may use different anchors.
  With offset=1, recent's latest window value is target minus horizon minus 1.
  Offset/window count time periods, not available rows; missing periods do not
  compress the grid. Historical offsets are positive integers, default 1.

Expected output and downstream planning:
All original columns are retained and features appended. Direct additionally
adds horizon. Input order is retained; each direct row uses the supplied horizon
order. The original time column remains the target time. Short history gives
NaN according to min_periods; rows are not automatically removed.
Use the recorded feature_columns list for model inputs. Retained sales/labels,
metadata and other original columns are not automatically safe model features.
Plan holdouts and any iterative forecasting outside the engine. Never populate
unknown forecast periods with validation/test actuals. Instantiate a new engine
after explicitly updating an input frame, since the engine keeps a copy.

When delegating to Create Script (AI), set use_demand_engine=true for feature
creation. Pass input aliases, schemas/date formats, grouping/aggregation,
frequency, transformations, offsets, per-feature direct anchors, direct horizons,
source policies, split roles, output paths and acceptance checks. The coder has
no prior conversation and receives a separate implementation guide.
Keep historical feature generation, fitting and submission in separate stages;
register the baseline first. Set the flag false for stages that only consume
features, fit models, inspect data or plot. You remain responsible for validation
coverage, fit boundaries and preprocessing; retained columns must be selected
deliberately. Inspect the trusted utility if needed via Inspect Script Lines at
@trusted/forecast_features.py, at most 100 lines per call.
"""

AGENT_GUIDE = """Forecast feature utilities: implementation contract for the coding LLM.
Implement the supplied objective using its columns, transformations and split
policies. The examples below are not schema defaults.

Imports and calls:
    from forecast_features import demand_engine
    engine = demand_engine(DATASETS['history'], transformations, level=[...],
                           time_column='...', frequency='...', source_options={...})
    result = engine.build(mode='recursive')
    # Or, with explicit anchors on EVERY feature:
    result = engine.build(mode='direct', horizons=[1, 2, 3])
    frame = result.frame
    result.save('features.csv')
Do not pass requests, origin, predictions or lag. There is no separate forecast
input table. demand_engine retains a copy of the original data; build appends
features for those rows. It does not fit or call models. Recursive is the default
build mode; call modes explicitly in generated scripts.

Construct inputs correctly:
- Use loaded DATASETS entries, not filenames or replacement loaders. Parse dates
  using verified formats. Retain one row per group/time with non-null keys and
  numeric transformed values. Include target rows whose features are wanted.
- level is a list/tuple of grouping columns; [] means one global series. Supply
  time_column and fixed frequency such as '1D' or '7D', not calendar months.
  Group times must share a consistent grid and timezone. Aggregate explicitly.
- transformations maps input columns to named feature dictionaries containing
  operation, window, offset, anchor, min_periods, outputs and/or quantile.
  Offset is the total positive backward shift, default 1. There is no lag field;
  operation='lag' still means returning the last value of the selected window.
- Recursive features use target as the default/only anchor. Direct features
  must explicitly set anchor='target' or anchor='recent' individually. No 'origin'
  or 'auto' anchor is accepted. A table can mix both direct anchors.
- Direct requires a nonempty ordered iterable of distinct positive integer
  horizons. Do not supply booleans, floats or a pre-expanded input table.
  Input column 'horizon' conflicts with the generated direct column; rename it
  explicitly. Feature output names must not overwrite ANY retained input column.

Window arithmetic:
    target = input_row[time_column]
    # Recursive, and direct features with anchor='target':
    anchor_time = target
    # Direct features with anchor='recent':
    anchor_time = target - horizon * frequency
    window_end = anchor_time - offset * frequency
    window_start = window_end - (window - 1) * frequency
For direct target features require offset >= max(horizons). Equality is valid.
For recent features offset is applied AFTER the horizon shift; offset=1 places
the window end at target-(horizon+1) periods. Window length does not change the
offset constraint. Invalid configurations raise errors; do not silently alter
the objective, drop features/horizons, or change min_periods to bypass it.

Example: built-in feature specs with mixed direct anchors:
```python
import pandas as pd
from forecast_features import demand_engine
# Standalone demonstration; generated scripts use declared DATASETS entries.
history = pd.DataFrame({
    'sku': ['A'] * 6,
    'time': pd.date_range('2024-01-01', periods=6, freq='D'),
    'sales': [10, 20, 30, 40, 50, 60],
    'region': ['north'] * 6,
})
engine = demand_engine(history, {'sales': {
    'target_mean': {'operation': 'mean', 'window': 2,
                    'offset': 3, 'anchor': 'target'},
    'recent_mean': {'operation': 'mean', 'window': 2,
                    'offset': 1, 'anchor': 'recent'},
}}, level=['sku'], time_column='time', frequency='1D')
result = engine.build(mode='direct', horizons=[1, 2, 3])
```
There are 18 rows. For target Jan 6, retained sales=60 and region='north':
    horizon  target_mean  recent_mean
       1         25           35
       2         25           25
       3         25           15
The target window is Jan 2-3 for all horizons; recent windows are Jan 3-4,
Jan 2-3 and Jan 1-2. Earlier targets remain present with short-history NaNs.

Recursive example:
```python
engine = demand_engine(history, {'sales': {
    'recent_mean': {'operation': 'mean', 'window': 2, 'offset': 1},
}}, level=['sku'], time_column='time', frequency='1D')
result = engine.build(mode='recursive')
```
There are six rows with all original columns and no generated horizon. Jan 6's
recent_mean is 45, computed from Jan 4-5. No prediction loop is performed.
For future targets, explicitly include rows with missing unavailable values.
Only earlier supplied values are used. If a surrounding prediction stage updates
the input data, instantiate a new engine; no predictions= or recursive_target
argument exists. Do not use future held-out actuals to simulate predictions.

Consume outputs correctly:
Use result.frame. Original columns, values, dtypes and row order are retained;
direct repeats each input row in the requested horizon order and adds horizon.
The time column is already the target time: do not add horizon to it. Labels
already present in input remain present, but must be excluded from model inputs.
Use result.provenance['feature_columns'] as the generated feature list; do not
infer features by merely excluding keys. Save that ordered list and split roles.
Expected row count is len(data) for recursive, len(data)*len(horizons) for direct.
Validate shapes, key uniqueness, short-history coverage and missing-value policy
before fitting. Report any downstream validation filtering explicitly.
result.save(path) saves the full output CSV and its lineage JSON sidecar.

Custom transforms:
```python
def value_range(window):
    return {'range': max(window.values) - min(window.values),
            'last': window.values[-1]}
spread_spec = {'operation': value_range, 'window': 2, 'offset': 1,
               'anchor': 'recent', 'outputs': ('range', 'last')}
```
Define callbacks in Python files so their identity can be recorded. They receive
immutable values/periods (oldest first) and available/observed/missing masks.
Return numeric scalars or mappings with declared outputs; dictionary keys become
name__key feature columns. The engine does not distinguish predictions from
actuals in supplied numeric values. Custom code must respect the objective.

Other settings:
- Built-ins: lag, mean, sum, median, population std, min, max, quantile,
  nonzero_count and nonzero_fraction. min_periods defaults to window; insufficient
  finite values yield NaN. Lower it only when justified. quantile defaults to 0.5.
- source_options supports missing ('nan' or justified 'zero'), availability
  ('observed' or 'known_future'), and available_at (release timestamp column).
  For recursive, observed periods/releases must precede target. For direct,
  they must precede target-(horizon-1)*frequency. There is no origin input.
  Direct target offset constraints still apply to known-future features.
- Missing grid periods never compress windows. Missing output rows are not
  synthesized: supply the desired group/time rows yourself, with unknown values
  missing. No automatic label filtering or fitting takes place.
Keep historical feature generation, fitting and prediction as separate stages.

## Migration and runtime

- Replace `build(requests, ...)` with `build(mode=..., horizons=...)` for direct,
  or `build(mode='recursive')`. Target rows come from the instantiation DataFrame.
- Remove the lag parameter: new offset = old lag + old offset when preserving
  a target-relative window. The `lag` operation remains available.
- Replace origin-based planning with per-target rows. `recent` is exactly target
  minus horizon periods; its offset is applied afterwards, not inclusive of that shift.
- All original columns are retained. Select generated features using provenance,
  especially when the input includes the label. Do not modify archived run scripts
  and assume they will work with v5; regenerate or migrate them explicitly.
- Horizon-name and feature-name collisions raise errors instead of overwriting data.
- Forecast input updates require a new engine. The utility cannot infer whether
  a supplied value is an observation or an externally produced prediction.

The installed utility remains read-only inside protected executions. Builds emit
timing markers consumed by timeout diagnostics and preserve source/configuration,
window and output hashes in the feature audit. Custom callbacks are not a sandbox.
The main agent receives the planning guide; Create Script (AI) receives the usage
guide only with use_demand_engine=true. Existing staged model/submission workflows
and provenance capture remain in place.
