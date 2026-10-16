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

Before final submission, explore and compare different demand forecasting
modeling families or philosophies. Use validation performance to select the
final approach.

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


Completeness contract:
Produce and preserve BOTH a baseline test submission and a test submission from
at least one changed modeling approach. First use Execute Experiment with
role="baseline" and parent_submission_id="". Then use it with role="experiment"
and parent_submission_id set to the accepted baseline execution_id. Repair and
rerun rejected executions. Merely writing a script or describing an experiment
does not count. Comment, filename, and random-seed-only changes do not qualify.

Scripts must save their test predictions to:
    Path(os.environ["EXPERIMENT_OUTPUT_DIR"]) / "submission.csv"
Import pathlib.Path and os as needed. Each execution receives a fresh output
directory. Continue reading the supplied data from the workspace. The tool
archives submissions, source/configuration snapshots, and execution receipts.
Validation comparisons remain required; do not use hidden test performance.
The experiment need not outperform the baseline. Select either accepted approach
using validation performance, copy its predictions to workspace submission.csv,
and leave solution code and a Final Answer as usual. Final model selection does
not discard either registered submission.
