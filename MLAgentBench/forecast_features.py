"""Target-row feature windows calculated only from supplied data.

Custom callbacks receive immutable window values, never the underlying frames.
This module controls its own windows, not arbitrary file access by user code.
"""
from dataclasses import dataclass, asdict
from collections.abc import Mapping
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
import uuid
import sys
import time
from functools import wraps

import numpy as np
import pandas as pd

def _time_build(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        call_id = uuid.uuid4().hex
        started = time.monotonic()
        caller = sys._getframe(1)
        context = {"call_id": call_id, "file": caller.f_code.co_filename,
                   "line": caller.f_lineno, "input_rows": self.input_row_count,
                   "started_monotonic": started}
        del caller
        def emit(event, **extra):
            print("[MLAB_BUILD] " + json.dumps(dict(context, event=event, **extra)),
                  file=sys.stderr, flush=True)
        emit("start")
        status = "error"
        try:
            result = method(self, *args, **kwargs)
            status = "complete"
            return result
        finally:
            emit(status, elapsed_seconds=time.monotonic() - started)
    return wrapped


VERSION = "6.0"

MAIN_AGENT_GUIDE = """demand_engine is a feature-building utility for forecasting models.
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

Format: input column -> feature name -> operation and window settings.
```python
transform_dict = {
    'column_name': {  # Existing numeric value column, not a grouping/time column.
        'feature_name': {  # Nonempty output name; must not collide with other columns.
            # Calculation applied to the selected historical window. Default: 'mean'.
            # Accepted strings: 'lag', 'mean', 'sum', 'median', 'std', 'min', 'max',
            # 'quantile', 'nonzero_count', 'nonzero_fraction'. 'lag' returns the
            # last window value; 'std' uses population standard deviation (ddof=0).
            # Also accepts a callable function(window), defined in a Python file
            # with inspectable source, returning a scalar or a dict (see outputs).
            'operation': 'mean',
            # Number of frequency periods in the window: positive int; default 1.
            'window': 4,
            # Total backward shift from the anchor: positive int; default 1.
            # Direct mode with anchor='target' requires offset >= max(horizons).
            'offset': 1,
            # Window reference: 'target' or 'recent'. Omitted/None defaults to
            # 'target' in recursive mode, where 'recent' is not allowed. Direct
            # mode requires an explicit anchor on every feature: 'target' means
            # the row's time; 'recent' means row time minus horizon * frequency.
            'anchor': 'target',
            # Minimum finite values needed: int from 1 to window, or None
            # (default: require the full window). Insufficient values yield NaN.
            'min_periods': 4,
            # Optional list/tuple of unique string keys for a custom callback's
            # returned dict. Default []/() means one scalar output. Nonempty
            # keys must exactly match the returned dict; columns are named
            # feature_name__key. Use empty outputs for built-in operations.
            'outputs': [],
            # Quantile probability in [0, 1]; default 0.5 (median).
            # Used only when operation='quantile'; ignored for other operations.
            'quantile': 0.5,
        },
    },
}
```
Replace the column and feature names with your actual input/output names. Pass
transform_dict as the transformations argument to demand_engine. This example
uses a recursive target anchor. Boolean values are not accepted for window,
offset or min_periods. Fields may be omitted to use their defaults, except that
direct mode requires an explicit anchor. No other feature fields are accepted.

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
"""


@dataclass(frozen=True)
class SourcePolicy:
    name: str
    keys: tuple
    time_column: str
    value_column: str
    frequency: str = "7D"
    availability: str = "observed"
    available_at: str | None = None
    missing: str = "nan"

    def __post_init__(self):
        if self.availability not in {"observed", "known_future"} or self.missing not in {"nan", "zero"}:
            raise ValueError("Invalid source policy")
        if pd.Timedelta(self.frequency) <= pd.Timedelta(0):
            raise ValueError("frequency must be a positive fixed duration")


@dataclass(frozen=True)
class Window:
    values: tuple
    periods: tuple
    available_mask: tuple
    observed_mask: tuple
    missing_mask: tuple


@dataclass(frozen=True)
class Feature:
    name: str
    source: str
    window: int = 1
    offset: int = 1
    anchor: str | None = None
    operation: object = "mean"
    min_periods: int | None = None
    outputs: tuple = ()
    quantile: float = .5

    def __post_init__(self):
        for name in ("window", "offset"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(name + " must be a positive integer")
        if self.anchor not in {None, "target", "recent"}:
            raise ValueError("anchor must be 'target' or 'recent'")
        if self.min_periods is not None and (not isinstance(self.min_periods, int)
                or isinstance(self.min_periods, bool) or not 1 <= self.min_periods <= self.window):
            raise ValueError("min_periods must be an integer within the window")
        if isinstance(self.operation, str) and self.operation not in {
                "lag", "mean", "sum", "median", "std", "min", "max", "quantile",
                "nonzero_count", "nonzero_fraction"}:
            raise ValueError("Unknown operation: " + self.operation)
        if len(set(self.outputs)) != len(self.outputs) or any(not isinstance(k, str) for k in self.outputs):
            raise ValueError("Output keys must be unique strings")
        if not 0 <= self.quantile <= 1:
            raise ValueError("quantile must be in [0,1]")


@dataclass
class FeatureResult:
    frame: pd.DataFrame
    provenance: dict

    def save(self, path):
        path = Path(path)
        self.frame.to_csv(path, index=False)
        path.with_suffix(path.suffix + ".lineage.json").write_text(
            json.dumps(self.provenance, indent=2), encoding="utf-8")


def _operation_identity(operation):
    if isinstance(operation, str):
        return operation
    if not callable(operation):
        raise ValueError("operation must be a built-in name or callable")
    try:
        source = inspect.getsource(operation)
    except (OSError, TypeError):
        raise ValueError("Custom operations must have inspectable source; define them in a Python file") from None
    return {"name": getattr(operation, "__qualname__", type(operation).__name__),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "source_file": inspect.getsourcefile(operation),
            "note": "Hash covers callable source, not arbitrary globals or captured state."}


def _validate_horizon(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError("horizon must be a positive integer")
    return int(value)


def _rolling_builtin(values, feature):
    """Batch the same finite-value NumPy reductions used by _calculate.

    Packing finite values preserves the scalar evaluator's reduction order even
    for large baselines with tiny variance. Chunking bounds temporary matrices.
    """
    width = feature.window
    minimum = feature.min_periods if feature.min_periods is not None else width
    padded = np.pad(values.to_numpy(), (width - 1, 0), constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    result = np.full(len(values), np.nan)
    chunk_size = max(1, 1_000_000 // width)
    for start in range(0, len(values), chunk_size):
        batch = windows[start:start + chunk_size]
        finite = np.isfinite(batch)
        counts = finite.sum(axis=1)
        for count in np.unique(counts[counts >= minimum]):
            positions = np.flatnonzero(counts == count)
            rows = batch[positions]
            packed = rows[np.isfinite(rows)].reshape(len(rows), int(count))
            op = feature.operation
            if op == "lag":
                calculated = rows[:, -1]
            elif op == "std":
                calculated = packed.std(axis=1, ddof=0)
            elif op == "median":
                calculated = np.median(packed, axis=1)
            elif op == "quantile":
                calculated = np.quantile(packed, feature.quantile, axis=1)
            elif op == "nonzero_count":
                calculated = np.count_nonzero(packed, axis=1)
            elif op == "nonzero_fraction":
                calculated = np.count_nonzero(packed, axis=1) / count
            else:
                calculated = getattr(packed, op)(axis=1)
            result[start + positions] = calculated
    return pd.Series(result, index=values.index)


class FeatureEngine:
    def __init__(self, sources, features=()):
        """Retain the first source's full frame as target rows; copy all sources."""
        if not sources:
            raise ValueError("At least one source is required")
        self.__features = tuple(features)
        self.__sources, self.__policies, self.__source_hashes = {}, {}, {}
        first_policy, first_frame = next(iter(sources.values()))
        self.__data = first_frame.copy(deep=True)
        self.__keys = tuple(first_policy.keys)
        self.__time_column = first_policy.time_column
        self.__frequency = pd.Timedelta(first_policy.frequency)
        for name, (policy, frame) in sources.items():
            if name != policy.name:
                raise ValueError("Source name mismatch")
            if tuple(policy.keys) != self.__keys or policy.time_column != self.__time_column:
                raise ValueError("Sources must share grouping keys and time column")
            if pd.Timedelta(policy.frequency) != self.__frequency:
                raise ValueError("Sources must share frequency")
            data = frame.copy(deep=True)
            required = list(policy.keys) + [policy.time_column, policy.value_column]
            if policy.available_at:
                required.append(policy.available_at)
            if not data.columns.is_unique or not set(required) <= set(data.columns):
                raise ValueError("Missing or duplicate source columns: " + name)
            data[policy.time_column] = pd.to_datetime(data[policy.time_column], errors="raise")
            if policy.available_at:
                data[policy.available_at] = pd.to_datetime(data[policy.available_at], errors="raise")
            key_columns = list(policy.keys) + [policy.time_column]
            if data[key_columns].isna().any().any():
                raise ValueError("Missing source keys or periods")
            if data.duplicated(key_columns).any():
                raise ValueError("Duplicate source series/time keys")
            grid = data[policy.time_column].min()
            if not data.empty and ((data[policy.time_column] - grid) % self.__frequency != pd.Timedelta(0)).any():
                raise ValueError("Source timestamps do not lie on the declared time grid")
            if name == next(iter(sources)):
                self.__targets = data[key_columns].copy()
                self.__grid = grid
            elif not data.empty and not pd.isna(self.__grid) and (grid - self.__grid) % self.__frequency != pd.Timedelta(0):
                raise ValueError("Sources must share the same time grid")
            data[policy.value_column] = pd.to_numeric(data[policy.value_column], errors="raise")
            if np.isinf(data[policy.value_column].to_numpy(dtype=float, na_value=np.nan)).any():
                raise ValueError("Infinite source values")
            rows = {}
            for row in data.to_dict("records"):
                value = row[policy.value_column]
                rows[(tuple(row[k] for k in policy.keys), row[policy.time_column])] = (
                    math.nan if pd.isna(value) else float(value),
                    row[policy.available_at] if policy.available_at else row[policy.time_column])
            self.__sources[name] = rows
            self.__policies[name] = policy
            self.__source_hashes[name] = hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()

    @property
    def input_row_count(self):
        return len(self.__data)

    @property
    def features(self):
        return self.__features

    @property
    def policies(self):
        return MappingProxyType(self.__policies)

    def window(self, feature, series, target, *, mode="recursive", horizon=None):
        if mode not in {"direct", "recursive"}:
            raise ValueError("mode must be direct or recursive")
        if feature.source not in self.__policies:
            raise ValueError("Unknown feature source: " + feature.source)
        policy = self.__policies[feature.source]
        if len(series) != len(policy.keys):
            raise ValueError("Series key length mismatch")
        target = pd.Timestamp(target)
        if pd.isna(target):
            raise ValueError("Missing target time")
        delta = self.__frequency
        if not pd.isna(self.__grid) and (target - self.__grid) % delta != pd.Timedelta(0):
            raise ValueError("Target must align with source time grid")
        if mode == "direct":
            horizon = _validate_horizon(horizon)
            if feature.anchor is None:
                raise ValueError(f"Direct feature {feature.name!r} requires anchor='target' or 'recent'")
            if feature.anchor == "target" and feature.offset < horizon:
                raise ValueError(f"Feature {feature.name!r}: offset must be >= horizon {horizon}")
            cutoff = target - (horizon - 1) * delta
            anchor = target - horizon * delta if feature.anchor == "recent" else target
        else:
            if horizon is not None:
                raise ValueError("Recursive mode does not take horizon")
            if feature.anchor not in {None, "target"}:
                raise ValueError("Recursive features must anchor at target")
            cutoff, anchor = target, target
        end = anchor - feature.offset * delta
        periods = tuple(end - i * delta for i in reversed(range(feature.window)))
        values, available, observed, missing = [], [], [], []
        rows = self.__sources[feature.source]
        for period in periods:
            entry = rows.get((tuple(series), period))
            eligible = period < cutoff
            if policy.availability == "known_future":
                eligible = entry is not None
            if entry is not None and policy.available_at:
                eligible = eligible and not pd.isna(entry[1]) and entry[1] < cutoff
            value, obs = math.nan, False
            if eligible:
                if entry is not None:
                    value, obs = entry[0], not math.isnan(entry[0])
                elif policy.missing == "zero":
                    value = 0.0
            values.append(value)
            available.append(eligible)
            observed.append(obs)
            missing.append(eligible and (entry is None or math.isnan(value)))
        return Window(tuple(values), periods, tuple(available), tuple(observed), tuple(missing))

    def _calculate(self, feature, window):
        values = np.array(window.values, dtype=float)
        values = values[np.isfinite(values)]
        names = [feature.name + "__" + key for key in feature.outputs] if feature.outputs else [feature.name]
        if len(values) < (feature.min_periods if feature.min_periods is not None else feature.window):
            return dict.fromkeys(names, math.nan)
        op = feature.operation
        if callable(op):
            result = op(window)
        else:
            operations = {"lag": lambda: window.values[-1], "mean": lambda: values.mean(),
                          "sum": lambda: values.sum(), "median": lambda: np.median(values),
                          "std": lambda: values.std(ddof=0), "min": lambda: values.min(),
                          "max": lambda: values.max(), "quantile": lambda: np.quantile(values, feature.quantile),
                          "nonzero_count": lambda: np.count_nonzero(values),
                          "nonzero_fraction": lambda: np.count_nonzero(values) / len(values)}
            result = operations[op]()
        if feature.outputs:
            if not isinstance(result, dict) or set(result) != set(feature.outputs):
                raise ValueError("Custom operation output keys do not match declared outputs")
            result = {feature.name + "__" + key: value for key, value in result.items()}
        else:
            result = {feature.name: result}
        for key, value in result.items():
            if not np.isscalar(value) or isinstance(value, (str, bytes)):
                raise ValueError("Feature must return numeric scalars")
            result[key] = float(value)
            if math.isinf(result[key]):
                raise ValueError("Infinite feature value")
        return result

    def _builtin_columns(self, active_horizons):
        """Compute rolling statistics once per series/spec, then gather horizons.

        Release-time policies and callbacks retain the general window evaluator.
        Dense grids are bounded so sparse, very long histories also fall back.
        """
        columns = {}
        targets = self.__targets
        if targets.empty:
            return columns
        groups = {}
        for position, row in enumerate(targets.to_dict("records")):
            series = tuple(row[k] for k in self.__keys)
            groups.setdefault(series, []).append((position, row[self.__time_column]))
        for source, policy in self.__policies.items():
            features = [f for f in self.features if f.source == source
                        and isinstance(f.operation, str) and not f.outputs]
            if not features or policy.available_at:
                continue
            padding = max(f.offset + f.window - 1 +
                          (max(active_horizons) if active_horizons[0] is not None
                           and f.anchor == "recent" else 0) for f in features)
            # Avoid densifying arbitrarily sparse histories or huge windows.
            if any((max(t for _, t in rows) - min(t for _, t in rows)) /
                   self.__frequency + padding + 1 > 100_000 for rows in groups.values()):
                continue
            output = {f.name: np.empty((len(targets), len(active_horizons))) for f in features}
            source_groups = {}
            for (series, period), (value, _) in self.__sources[source].items():
                source_groups.setdefault(series, {})[period] = value
            for series, rows in groups.items():
                positions, times = zip(*rows)
                times = pd.DatetimeIndex(times)
                start = times.min() - padding * self.__frequency
                size = int((times.max() - start) / self.__frequency) + 1
                # Fixed elapsed periods, including across daylight-saving changes.
                grid = pd.DatetimeIndex(start + np.arange(size) * self.__frequency)
                entries = source_groups.get(series, {})
                fill = 0.0 if policy.missing == "zero" and policy.availability == "observed" else math.nan
                values = pd.Series([entries.get(t, fill) for t in grid], index=grid, dtype=float)
                cache = {}
                for feature in features:
                    minimum = feature.min_periods if feature.min_periods is not None else feature.window
                    key = (feature.operation, feature.window, minimum,
                           feature.quantile if feature.operation == "quantile" else None)
                    if key not in cache:
                        cache[key] = _rolling_builtin(values, feature)
                    for index, horizon in enumerate(active_horizons):
                        shift = feature.offset + (horizon if horizon is not None and feature.anchor == "recent" else 0)
                        ends = times - shift * self.__frequency
                        output[feature.name][np.asarray(positions), index] = cache[key].reindex(ends).to_numpy()
            for name, values in output.items():
                if np.isinf(values).any():
                    raise ValueError("Infinite feature value")
                columns[name] = values.ravel()
        return columns

    def _compact_windows(self):
        """Store source presence/release facts once, not once per feature window."""
        series_ids, series_list = {}, []
        def series_id(series):
            if series not in series_ids:
                series_ids[series] = len(series_list)
                series_list.append(list(series))
            return series_ids[series]
        targets = {"series": [], "times": []}
        timezone = None
        for row in self.__targets.to_dict("records"):
            targets["series"].append(series_id(tuple(row[k] for k in self.__keys)))
            target = row[self.__time_column]
            targets["times"].append(str(target))
            timezone = str(target.tz) if target.tz is not None else None
        sources = {}
        for name, rows in self.__sources.items():
            source = {"series": [], "times": [], "observed": []}
            if self.__policies[name].available_at:
                source["released_at"] = []
            for (series, period), (value, release) in rows.items():
                source["series"].append(series_id(series))
                source["times"].append(str(period))
                source["observed"].append(not math.isnan(value))
                if "released_at" in source:
                    source["released_at"].append(None if pd.isna(release) else str(release))
            sources[name] = source
        return {"encoding": "source-target-v1", "frequency": str(self.__frequency),
                "target_timezone": timezone, "series": series_list,
                "targets": targets, "sources": sources}

    @_time_build
    def build(self, *, mode="recursive", horizons=None):
        """Append features to the retained input rows; direct expands by horizons."""
        if mode not in {"direct", "recursive"}:
            raise ValueError("mode must be direct or recursive")
        if not self.features:
            raise ValueError("At least one feature is required")
        if mode == "direct":
            if horizons is None or isinstance(horizons, (str, bytes, Mapping, set, frozenset)):
                raise ValueError("Direct mode requires a nonempty ordered horizons iterable")
            try:
                horizons = [_validate_horizon(h) for h in horizons]
            except TypeError:
                raise ValueError("Direct mode requires a nonempty ordered horizons iterable") from None
            if not horizons or len(set(horizons)) != len(horizons):
                raise ValueError("horizons must be nonempty and unique")
            if "horizon" in self.__data.columns:
                raise ValueError("Input column 'horizon' conflicts with direct output; rename it explicitly")
            for feature in self.features:
                if feature.anchor is None:
                    raise ValueError(f"Direct feature {feature.name!r} requires anchor='target' or 'recent'")
                if feature.anchor == "target" and feature.offset < max(horizons):
                    raise ValueError(f"Feature {feature.name!r}: offset must be >= maximum horizon {max(horizons)}")
        else:
            if horizons is not None:
                raise ValueError("horizons is only supported in direct mode")
            if any(f.anchor not in {None, "target"} for f in self.features):
                raise ValueError("Recursive features must anchor at target")
        config, output_names = [], []
        for feature in self.features:
            if feature.source not in self.__sources:
                raise ValueError("Unknown feature source: " + feature.source)
            spec = asdict(feature)
            spec["anchor"] = feature.anchor or "target"
            spec["operation"] = _operation_identity(feature.operation)
            config.append(spec)
            output_names.extend([feature.name + "__" + k for k in feature.outputs] if feature.outputs else [feature.name])
        reserved = set(self.__data.columns) | ({"horizon"} if mode == "direct" else set())
        if len(set(output_names)) != len(output_names) or set(output_names) & reserved:
            raise ValueError("Feature names collide with input/output columns")
        active_horizons = horizons if mode == "direct" else [None]
        positions = np.repeat(np.arange(len(self.__data)), len(active_horizons))
        frame = self.__data.iloc[positions].copy(deep=True)
        if mode == "direct":
            frame = frame.reset_index(drop=True)
            frame["horizon"] = np.tile(np.array(horizons, dtype=np.int64), len(self.__data))
        columns = self._builtin_columns(active_horizons)
        remaining = [feature for feature in self.features if feature.name not in columns]
        records = []
        for row in (self.__targets.to_dict("records") if remaining else []):
            series, target = tuple(row[k] for k in self.__keys), row[self.__time_column]
            for horizon in active_horizons:
                record = {}
                for feature in remaining:
                    window = self.window(feature, series, target, mode=mode, horizon=horizon)
                    record.update(self._calculate(feature, window))
                records.append(record)
        for name in output_names:
            frame[name] = columns[name] if name in columns else np.array([r[name] for r in records], dtype=float)
        provenance = {"engine_version": VERSION,
                      "engine_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "mode": mode, "horizons": horizons,
                      "time_column": self.__time_column, "feature_columns": output_names,
                      "input_columns": list(self.__data.columns),
                      "input_rows": len(self.__data), "output_rows": len(frame),
                      "input_table_sha256": hashlib.sha256(self.__data.to_csv(index=False).encode()).hexdigest(),
                      "policies": [asdict(p) for p in self.__policies.values()],
                      "source_table_sha256": dict(self.__source_hashes),
                      "features": config, "windows": self._compact_windows(),
                      "feature_csv_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()}
        audit = os.environ.get("MLAB_FEATURE_AUDIT_DIR")
        if audit:
            directory = Path(audit)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / ("features_" + uuid.uuid4().hex + ".json")).write_text(json.dumps(provenance, default=str), encoding="utf-8")
        return FeatureResult(frame, provenance)


def iter_window_lineage(provenance):
    """Yield the original detailed window records from compact or legacy lineage.

    Numeric source values are unnecessary for masks; source hashes still identify
    the exact input bytes. Reconstruction reuses the engine's window evaluator.
    Iterate lazily: materializing every record can be very large.
    """
    windows = provenance["windows"]
    if isinstance(windows, list):
        yield from windows
        return
    if windows.get("encoding") != "source-target-v1":
        raise ValueError("Unknown window lineage encoding")
    timezone = windows["target_timezone"]
    def timestamp(value):
        t = pd.Timestamp(value)
        return t.tz_convert(timezone) if timezone is not None and not pd.isna(t) else t
    sources = {}
    series_list = windows["series"]
    for spec in provenance["policies"]:
        policy = SourcePolicy(**dict(spec, keys=tuple(spec["keys"])))
        saved = windows["sources"][policy.name]
        frame = pd.DataFrame([series_list[i] for i in saved["series"]], columns=list(policy.keys))
        frame[policy.time_column] = pd.DatetimeIndex([timestamp(t) for t in saved["times"]])
        frame[policy.value_column] = [1.0 if observed else math.nan for observed in saved["observed"]]
        if policy.available_at:
            frame[policy.available_at] = pd.DatetimeIndex([timestamp(t) for t in saved["released_at"]])
        sources[policy.name] = (policy, frame)
    engine = FeatureEngine(sources)
    features = [Feature(**dict(spec, operation="lag")) for spec in provenance["features"]]
    mode = provenance["mode"]
    horizons = provenance["horizons"] if mode == "direct" else [None]
    for series_id, time_string in zip(windows["targets"]["series"], windows["targets"]["times"]):
        series, target = tuple(series_list[series_id]), timestamp(time_string)
        for horizon in horizons:
            for feature in features:
                window = engine.window(feature, series, target, mode=mode, horizon=horizon)
                yield {"feature": feature.name, "series": list(series),
                       "target": str(target), "horizon": horizon,
                       "anchor": feature.anchor or "target",
                       "periods": [str(p) for p in window.periods],
                       "available": list(window.available_mask),
                       "observed": list(window.observed_mask), "missing": list(window.missing_mask)}


def demand_engine(data, transformations, *, level, time_column, frequency, source_options=None):
    """Retain input rows and configure features. Call build() with no request frame."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not data.columns.is_unique:
        raise ValueError("DataFrame columns must be unique")
    if not isinstance(level, (list, tuple)) or any(not isinstance(k, str) for k in level):
        raise ValueError("level must be a list or tuple of grouping column names")
    keys = tuple(level)
    if len(set(keys)) != len(keys):
        raise ValueError("level columns must be unique")
    if not isinstance(time_column, str) or time_column in keys:
        raise ValueError("time_column must name a column distinct from level")
    if not set(keys + (time_column,)) <= set(data.columns):
        raise ValueError("Missing level or time columns")
    if not isinstance(frequency, str):
        raise ValueError("frequency must be an explicit fixed-duration string")
    if not isinstance(transformations, Mapping) or not transformations:
        raise ValueError("transformations must be a nonempty column-to-transforms mapping")
    source_options = {} if source_options is None else source_options
    if not isinstance(source_options, Mapping) or set(source_options) - set(transformations):
        raise ValueError("source_options must map transformed columns to policies")
    sources, features, output_names = {}, [], set(data.columns)
    allowed_features = {"operation", "window", "offset", "anchor", "min_periods", "outputs", "quantile"}
    allowed_policies = {"availability", "available_at", "missing"}
    for column, transforms in transformations.items():
        if not isinstance(column, str) or column not in data.columns:
            raise ValueError("Unknown input column: " + str(column))
        if column in keys or column == time_column:
            raise ValueError("Transform inputs must be value columns, not level/time columns")
        if not isinstance(transforms, Mapping) or not transforms:
            raise ValueError("Each input column needs a nonempty named transform mapping")
        options = source_options.get(column, {})
        if not isinstance(options, Mapping) or set(options) - allowed_policies:
            raise ValueError("Unknown source policy options for " + column)
        policy = SourcePolicy(column, keys, time_column, column, frequency=frequency, **options)
        sources[column] = (policy, data)
        for name, spec in transforms.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Transform output names must be nonempty strings")
            if isinstance(spec, str) or callable(spec):
                spec = {"operation": spec}
            if isinstance(spec, Mapping) and "lag" in spec:
                raise ValueError("lag parameter was removed; use offset as the total backward shift")
            if not isinstance(spec, Mapping) or set(spec) - allowed_features:
                raise ValueError("Invalid transform specification for " + name)
            spec = dict(spec)
            if "outputs" in spec:
                if not isinstance(spec["outputs"], (list, tuple)):
                    raise ValueError("outputs must be a list or tuple")
                spec["outputs"] = tuple(spec["outputs"])
            feature = Feature(name, column, **spec)
            _operation_identity(feature.operation)
            names = {name + "__" + k for k in feature.outputs} if feature.outputs else {name}
            if output_names & names:
                raise ValueError("Feature names collide with input/output columns")
            output_names.update(names)
            features.append(feature)
    return FeatureEngine(sources, features)
