# Submission lineage and code-review bundles

New demand-forecasting runs automatically record provenance. No extra agent tool
arguments or model-specific interfaces are required. Each accepted Execute
Experiment receipt links to `env_log/provenance/graph.json`, its submission node,
and its zero-based trajectory step index. Final Answer also registers the current
workspace `submission.csv` under submission ID `final` when that file exists.

## Evidence recorded

- Initial workspace files, including raw/prepared inputs and task instructions.
- Content-addressed immutable copies of file versions (including binary models).
- Each step's action, arguments, trace reference, inputs, created/edited versions,
  deletions, same-content write attempts, and coverage gaps.
- File access attempts during Execute Script and Execute Experiment, including
  imported workspace source, model/configuration loads, and intermediate reads.
- Intermediate files captured when read, even if deleted before the action ends.
- Dependencies from each action's output versions to its observed input versions.
- Previous-version links retaining all earlier creation/edit steps for relevant
  code and artifacts. Later edits cannot change an earlier submission's lineage.

The review graph distinguishes execution dependencies from `previous_version`
history. Historical code is not necessarily the code that executed. Dependencies
are conservative at action/execution level: a file opened by the program might
not influence every output. No row-level or column-level causality is claimed.

## Export a review bundle

From the repository root, after a recorded run:

```bash
python -m MLAgentBench.provenance runs/RUN/env_log/provenance --submission exec_ID --output reviews/exec_ID
```

Use `--submission final` to inspect the final workspace submission. Choose a new
output directory; existing reviews are not overwritten. Execution IDs are in the
completeness receipts or List Experiments. The output contains:

- `review.json`: relevant version graph, chronological steps, all `step_indices`,
  initial/raw roots, preparation records, integrity checks and coverage limits.
- `files/<version_id>/<original_filename>`: exact archived bytes for review.
- `access/`: runtime file-access evidence for linked steps.
- `executions/<execution_id>/`: available receipts and stdout/stderr.
- `REVIEW.md`: a per-submission correctness-review rubric for a human or LLM.

`steps[*].trace_reference` points to the corresponding **zero-based** entry in
`env_log/trace.json`. Snapshot names use the same index. The report does not use
potentially duplicated human-readable step labels in the agent main log.
Each node records its creation/change step and hash. Follow dependencies back
through intermediate artifacts, and previous_version links for all earlier edits.

No LLM is called and no correctness pass is inferred. The bundle starts with
`correctness_verdict: not_reviewed`; it supplies auditable evidence for the first
per-submission review stage. Targeted replay and controlled data isolation are
separate future work.

## Link prepared files back to original raw data

Future task preparation writes `scripts/preparation_provenance/manifest.json`
and a content-addressed archive of the original raw inputs, preparation/scoring
code, and prepared outputs. New runs import the matching preparation evidence
into their own archive. Preparation dependencies are conservative at preparation
execution level, outside the agent trajectory; they have a preparation ID rather
than an invented agent step index.

For an already-prepared task, the usual `prepared` marker bypasses preparation.
Run the existing preparation script explicitly if you want to regenerate inputs
with the new preparation archive, using the same source dataset:

```bash
python MLAgentBench/benchmarks/demand-forecasting/scripts/prepare.py --dataset-dir /path/to/source/dataset
```

This updates the prepared environment, so do it between runs. The update does
not automatically regenerate existing datasets or alter old trajectories.
Without that archive, raw source hashes from task_metadata.json are included as
**declared** upstream roots; missing raw bytes and preparation code remain gaps.

## Limits and storage

Python audit hooks are cooperative observation, not a security sandbox. Native
I/O can bypass them; child process activity is flagged as a coverage gap, not
silently considered traced. Python REPL and uninstrumented executions retain
step-level file changes but cannot claim observed read dependencies. External
access paths are recorded without copying host files or credentials. Runtime
packages outside the workspace are not bundled. Files with identical contents
share storage, but large evolving model/data artifacts require disk space.

Only action-boundary state and observed runtime reads are retained. A file
created and deleted entirely in unobserved native code may be unavailable. An
audit event proves an access attempt, not that the operation succeeded or its
value influenced predictions. A run starting from pre-existing derived artifacts
has unknown pre-run creation history unless preparation evidence supplies it.

Legacy trajectories without these records cannot gain verified runtime lineage
retroactively. Resume into a new log directory records the resumed workspace as
an initial state and marks earlier history unavailable. Historical snapshot
reconstruction can be developed separately, with inferred edges explicitly
distinguished from this runtime evidence.
