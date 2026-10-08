# Weekly demand forecasting

Build your own solution and produce `submission.csv`. No model or training
implementation is supplied. The target is observed sales units, used as the
demand proxy; there are no labels for unmet demand during stockouts.

## Data and forecast origins

| File | Contents |
| --- | --- |
| train.csv | Labeled weekly history before validation |
| validation.csv | Next 13 weeks, labeled for local model selection |
| test.csv | Final 13 weeks, keys only |
| inventory.csv | Daily end-of-day stock, strictly before test origin |
| item_master.csv | SKU to class/subcategory/category (some SKUs lack metadata) |
| location_master.csv | Store to city/country/region |
| data_dictionary.json | Original column definitions |
| task_metadata.json | Exact dates, row counts, assumptions, source hashes |
| sample_submission.csv | Schema/key template; zeros are placeholders |

All sales splits form a complete grid of source sales SKUs x source sales stores
x Monday weeks. Missing source sales records mean **zero demand**, including
leading, trailing, and internal gaps. Do not fill missing inventory or missing
master attributes with invented values. Left join metadata to retain all series.
Static master attributes are assumed available at each forecast origin.

Forecast all 13 weeks simultaneously at each origin. For validation, fit only
on train.csv and restrict inventory to dates strictly before validation starts.
validation.csv labels are for scoring and model selection, not within-horizon
features. For final test predictions, training may include validation.csv and
all provided inventory. No observed outcomes inside the test horizon are known.
Calendar features and recursively predicted lags are allowed. Random row splits
and future observed lags are inappropriate for this task.

## Submission and metric

Write `submission.csv` with exactly `sku,store,week,sales_units`, one row per
test key. Preserve ISO dates (`YYYY-MM-DD`). Row order does not matter; duplicate,
missing, extra, null, infinite, and negative predictions are rejected. Nonnegative
decimal predictions are allowed. No hierarchy-level substitution is accepted.

WMAPE = sum(|actual - predicted|) / sum(actual), evaluated directly over all
SKU-store-week cells, including zero-demand cells. Lower is better. This is a
fraction, not a percentage value and not an unweighted average of series WMAPEs.
An all-zero actual evaluation set is undefined and rejected by the scorer.

To check predictions for the validation keys:

```bash
python evaluate.py validation_predictions.csv
```

You may create and edit your own solution code. Keep supplied data, metadata,
task instructions, and scoring helpers unchanged. Stay within the workspace;
do not access source data or hidden evaluation files outside it. Leave your code
and final submission in the workspace before submitting a Final Answer.
