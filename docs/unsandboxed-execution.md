# Unsandboxed execution branch

`codex/unsandboxed-execution` runs generated scripts as ordinary Python
subprocesses. It never invokes Bubblewrap, creates namespaces or installs a
replacement sandbox. This is the default on this branch; no extra flag is needed.

Execute Script, Execute Experiment and the demand-forecasting Python REPL use
the same command builder. The cooperative provenance bootstrap still loads the
supplied feature engine and records file accesses. Logging is not access control:
scripts have the runner user's normal filesystem, environment and network access.
The legacy internal `protected_features` argument enables feature loading and
auditing only on this branch.

Script and experiment timeouts, stdout/stderr logs, experiment IDs, receipts,
artifact validation and post-run test scoring continue to operate. Existing
file-tool checks are not a sandbox for generated Python. No OS restriction
prevents Python from modifying supplied files, helper modules or other runs.

Each run records `env_log/execution_mode.json` and prints its execution mode.
Use the normal runner command or `bash run_trajectories.sh 5` after checking out
this branch on the execution machine. The normal main checkout can remain on
main by cloning this branch into a separate directory:

```bash
git clone --branch codex/unsandboxed-execution --single-branch \
  https://github.com/pr33t-commits/MLAgentBench.git MLAgentBench-unsandboxed
```

Configure the Python environment, model credentials and dataset location as for
main. Generate the task inputs in this new checkout before running trajectories.
