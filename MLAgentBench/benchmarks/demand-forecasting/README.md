# Demand forecasting task

Task ID: `demand-forecasting`. This task defines an environment for an agent to
solve; it does not supply or train a forecasting model. Preparation requires
Python, pandas, and NumPy. The full agent runner uses MLAgentBench's dependencies.

## Framework integration

- `prepare_task.py:get_task_info` resolves the task in `benchmarks/tasks.json`
  and reads `scripts/research_problem.txt`.
- `prepare_task.py:prepare_task` runs `scripts/prepare.py` once, then writes
  `scripts/prepared`. The source defaults to the sibling `DS_Agent/dataset`.
- `environment.py:_initialize_task_env` copies **only** `env/` into the agent's
  workspace and applies the patterns in `scripts/read_only_files.txt`.
- `environment.py:save` omits read-only data from snapshots and preserves the
  agent's code and `submission.csv`.
- `MLAgentBench.eval` imports `scripts/eval.py:get_score(submission_folder)` to
  score saved submissions against `scripts/answer.csv`. The evaluator loads its
  trusted scorer from `scripts/`, independent of agent-written files.
- `plot.py` recognizes this task as lower-is-better. No changes to the core
  environment or preparation framework are necessary.

The framework's read-only list controls file tools and snapshots; it is not OS
isolation for arbitrary Python execution. For benchmark integrity, expose only
the prepared workspace to the agent, keep the original dataset and `scripts/`
outside its sandbox, and evaluate externally. In particular, mounting the entire
DS_Agent checkout into an unrestricted agent gives it access to hidden labels.

## Task definition

The source contains 50,330 sales records, 50 SKUs and 10 stores over 156 weeks.
As requested, missing sales records are zero demand across the full SKU x store
x week grid, yielding 78,000 rows. SKU_020, SKU_029 and SKU_036 have no item master
row; they remain in all splits. Sales units are the available proxy for demand.

| Split | Monday week starts | Rows |
| --- | --- | ---: |
| Train | 2020-01-06 through 2022-06-27 | 65,000 |
| Validation | 2022-07-04 through 2022-09-26 | 6,500 |
| Hidden test | 2022-10-03 through 2022-12-26 | 6,500 |

Each holdout is a fixed-origin 13-week forecast. Inventory is cut strictly before
2022-10-03. Validation must additionally filter inventory before 2022-07-04.
Validation labels may be used for final fitting after model selection.

The metric is `sum(abs(actual - forecast)) / sum(actual)` at SKU x store x week,
reported as a fraction. Absolute errors are taken before any summation; zero
actuals contribute to the numerator. This is neither hierarchy-aggregated error
nor mean per-series WMAPE. An all-zero actual set raises an undefined-metric error.
Submissions must exactly match the test keys and have finite nonnegative values.

## Prepare and use

Run from the MLAgentBench repository root:

```bash
python -m MLAgentBench.prepare_task demand-forecasting
```

If the dataset is elsewhere, set `MLAGENTBENCH_DEMAND_DATASET` to its directory
before running that command. Generated data and the prepared marker are ignored
by Git. Preparation writes source hashes, cutoffs and counts to
`env/task_metadata.json`. The framework does not recheck hashes after its marker
exists. To intentionally refresh data, run the preparation script directly:

```bash
python MLAgentBench/benchmarks/demand-forecasting/scripts/prepare.py --dataset-dir /path/to/dataset
```

For your own agent, use `Environment(args)` with `args.task = 'demand-forecasting'`
and implement your agent against the provided research problem and workspace.
The agent should create its own solution code and `submission.csv`.
The built-in trivial `Agent` assumes a pre-existing `train.py`, so it does not
apply to this task. A ResearchAgent can instead start from the task instructions.

Example runner invocation on Linux/WSL with configured model credentials:

```bash
python -m MLAgentBench.runner --task demand-forecasting --agent-type ResearchAgent --work-dir workspace --log-dir runs/demand-forecasting --python python --llm-name gpt-4.1-mini --fast-llm-name gpt-4.1-mini --edit-script-llm-name gpt-4.1-mini
```

The existing Environment uses `readline` and `signal.SIGALRM`, so the full runner
requires Linux/WSL; task preparation and the standalone evaluator also work on
Windows. Environment initialization replaces its named task workspace; choose
a dedicated workspace. No agent run or model training was performed in defining
this task.

Evaluate an agent workspace or a saved submission snapshot externally:

```bash
python MLAgentBench/benchmarks/demand-forecasting/scripts/eval.py /path/to/submission_folder
python -m MLAgentBench.eval --task demand-forecasting --log-folder runs/demand-forecasting --output-file demand-results.json
```

Run the contract tests (synthetic data/predictions, no models):

```bash
python MLAgentBench/benchmarks/demand-forecasting/scripts/test_task.py
```
