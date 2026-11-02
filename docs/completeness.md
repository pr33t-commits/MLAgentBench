# Demand-forecasting completeness

Completeness requires an accepted baseline test submission and an accepted test
submission from a changed modeling approach. Either can be selected as the final
workspace `submission.csv`; improvement is not required. Split validity and model
quality remain separate dimensions.

## Agent workflow

Use `Execute Experiment` with all three arguments:

```json
{"script_name":"baseline.py","role":"baseline","parent_submission_id":""}
```

The script reads inputs from the workspace and writes predictions with:

```python
import os
from pathlib import Path
predictions.to_csv(Path(os.environ["EXPERIMENT_OUTPUT_DIR"]) / "submission.csv", index=False)
```

After an accepted baseline, run a changed model:

```json
{"script_name":"candidate.py","role":"experiment","parent_submission_id":"exec_ID_FROM_BASELINE"}
```

Every execution gets a fresh output directory. A failed process, missing output,
or malformed submission is rejected. Files left at workspace `submission.csv`
are never registered by accident. Copy the chosen accepted output from
`experiment_outputs/<execution_id>/submission.csv` to workspace `submission.csv`
for the existing final performance evaluator. Finish with Final Answer as usual.

The environment archives receipts, Python/JSON/YAML/TOML source/configuration
files, stdout/stderr, and validated predictions under
`env_log/completeness/exec_<id>/`. Sources are captured before execution; installed
packages, binary models and arbitrary external dependencies are not snapshotted.
Registration checks columns, exact test keys, uniqueness and finite nonnegative
values using an initial test-key copy. It never reads hidden labels. Regular
Execute Script also reports structured exit status, but does not register a
submission: use Execute Experiment for completeness milestones.

Execute Experiment returns stdout and stderr, including ordinary full tracebacks
and printed validation metrics. Each stream is limited to its last 16,000 bytes
in the immediate response, with explicit truncation and byte counts. Full logs
remain archived: use Read Experiment Log with execution_id, stream (stdout or
stderr), and offset=0, then follow next_offset until null. List Experiments
recovers exact IDs and script names when earlier observations leave context.
Source hashes remain in the receipt archive but are omitted from tool responses
to keep debugging feedback readable. These tools are enabled only for the
demand-forecasting environment, alongside Execute Experiment.

For multi-stage work, write explicit stage input/output contracts and preserve
validated intermediate artifacts. Execute Script runs intermediate stages;
Execute Experiment registers a submission-producing stage. Creating or editing
a script does not establish that it executes successfully. Read actual errors
before changing code, and keep accepted IDs and artifact paths in the research
plan rather than recreating completed work. Create Script includes small raw
input samples to ground dtype/date-format decisions; samples are not a full
dataset profile.

## Offline verification

The runner automatically scores every accepted registered baseline and experiment
after the agent stops, including when its run ends with an exception or timeout.
It calls the task's trusted `scripts/eval.py:get_score` on each archived
submission and writes `env_log/test_metrics.json`. For demand forecasting this
is key-aligned WMAPE over all SKU/store/week cells, expressed as a fraction;
lower is better. Hidden test scores are not returned as agent observations.

The report's `experiments` object is keyed by the exact `execution_id` returned
to the agent by Execute Experiment. Each entry includes its role,
`parent_submission_id`, script, artifact hash/path, `test_score`, and scoring
status. Rejected executions are skipped. Missing, changed or invalid artifacts
and evaluator failures receive a null score and an explicit error; other
executions are still evaluated. No model is rerun to obtain predictions.

`selected_submission` matches workspace `submission.csv` to scored archives by
SHA-256. It lists all matching execution IDs when multiple archives contain
identical bytes. An unmatched or missing final file has no inferred ID or score.
All registered scores remain available even if only one solution is selected.
Tasks without experiment registration retain their existing evaluation workflow.

From the MLAgentBench repository:

```bash
python -m MLAgentBench.completeness runs/my-run/env_log/completeness --output completeness.json
```

The baseline is worth 50 points; a verified experimental submission adds 50.
Artifacts and archived source hashes are rechecked. A valid pair with unresolved
semantic distinctness yields `completeness_score: null`, bounds `[50, 100]`, and
`fully_complete: null`. Missing instrumentation is unknown, not zero. Rejected
attempts earn no points. A judge rejection leaves baseline credit intact.

## Narrow semantic judge

Free-form code cannot prove distinctness from hashes. The verifier emits judge
requests only for accepted experiment/baseline pairs. The judge assesses whether
executed sources introduce a modeling change, not whether predictions improve.
Seed-only, cosmetic and filename changes do not qualify. Identical predictions
alone do not imply failure. An LLM assessment remains a judgment, not a proof.

Use a provider-specific adapter command (model choice and credentials belong in
that adapter). It receives one JSON packet on stdin containing instructions,
source/configuration files and execution logs, and must write only JSON on stdout:

```json
{"verdict":"pass","reason":"Specific modeling change and its execution evidence",
 "evidence":["exec_EXPERIMENT/sources/candidate.py:12"]}
```

Allowed verdicts are `pass`, `fail`, `unknown`. Cite supplied files with one-based
line numbers. Instructions explicitly treat source/log text as untrusted data.
Missing evidence, invalid citations, adapter errors, or evidence over 200 KB leave
the result unknown. The adapter must implement the supplied rubric and use an
appropriate model; a constant-response adapter is not a valid evaluation.

```bash
python -m MLAgentBench.completeness runs/my-run/env_log/completeness --output completeness.json --judge-output judgments.json --judge-command python my_judge_adapter.py
```

Judge responses are bound to hashes of both receipts and can be replayed without
another model call:

```bash
python -m MLAgentBench.completeness runs/my-run/env_log/completeness --judgments judgments.json --output completeness.json
```

For embedding, `judge_pending(receipt_dir, callback, identity)` accepts a Python
callback using the same packet/result contract. No model is called by default.
Calibrate the adapter against reviewed trajectories before using its scores for
training. External judgment files are trusted evaluator inputs, never agent files.

## Migration and limits

Existing snapshots can be manually reviewed, but are not automatically relabeled
as registered submissions. Re-run task preparation to refresh the portable
`metrics.py` and `submission_validation.py` helpers in previously prepared data.
Fresh runs need no training-data regeneration beyond that normal preparation.
On resume into a new log directory, previous receipts are not automatically
imported: register a baseline again before new experiments.

This runner is not an OS security sandbox. Receipt archives outside the workspace
and hashes protect against accidental overwrites, not hostile Python with host
filesystem access. Use external isolation for adversarial evaluations. Fresh
outputs do not rule out intentional copying; semantic review addresses that
ambiguity. No claim of cryptographic provenance or deterministic model distinctness
is made. Source inspection also cannot establish all runtime behavior.
