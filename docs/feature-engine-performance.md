# Feature engine execution and lineage (version 6)

The public `demand_engine(...)` and `build(...)` arguments and output columns are
unchanged. Built-in operations are batched per series and cached by operation,
window, minimum valid count and quantile. Offsets and horizons gather from those
cached results. The engine still uses fixed elapsed-time periods, not row counts
or calendar-day shifts, and retains input ordering and direct horizon ordering.

Batched NumPy reductions pack finite values in their original order, matching
the scalar evaluator, including population standard deviation. No missing values
are imputed except absent periods explicitly covered by the existing observed
source `missing='zero'` policy. Custom callbacks, release-time policies and very
sparse grids use the existing scalar window evaluator. Large sparse inputs are
not expanded into unbounded dense time grids.

## Compact provenance

`result.provenance['windows']` now contains a versioned dictionary with encoding
`source-target-v1`, rather than a list with one dictionary per output feature.
It records each source's period presence, observed-value mask and optional
release times, together with target rows, series keys and timezone. Feature
configuration, source policies and source/input/output hashes remain in the
surrounding provenance. Actual source values are identified by their existing
hashes; the detailed window records never contained those values.

Use the reconstruction helper when individual window records are needed:

```python
from forecast_features import iter_window_lineage

for record in iter_window_lineage(result.provenance):
    # Same feature, series, target, horizon, anchor, periods and masks as v5.
    inspect(record)
```

The helper also accepts legacy provenance containing a windows list. It reuses
the scalar window evaluator and yields records lazily in the original order.
Do not convert the iterator to a list for large jobs unless all records are
needed: the original expanded representation can occupy many gigabytes.
`result.save(...)` and the protected execution audit write the compact JSON.

## Regression checks

Run `python -B -m unittest discover -s tests -p test_forecast_features_current.py`.
The suite compares values and reconstructed lineage with the scalar evaluator
for all built-in operations, both modes/anchors, missing values and grid periods,
known-future/observed policies, release times, global series, custom multi-output
callbacks, DST transitions, sparse histories, empty inputs, and numerically
sensitive variance. It also exercises JSON save/audit and cache reuse.

The older `test_forecast_features.py` and `test_feature_runtime.py` target the
previous requests/origin API and require a separate migration; they are not
current-API regression coverage.
