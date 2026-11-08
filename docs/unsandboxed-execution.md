# Script execution modes

Both modes are available on main; a separate branch is no longer required.

```bash
python -m MLAgentBench.runner --task demand-forecasting --execution-mode bubblewrap
python -m MLAgentBench.runner --task demand-forecasting --execution-mode unsandboxed
bash run_trajectories.sh 5 --execution-mode unsandboxed
```

Add your usual model, Python, log-directory and workspace arguments. Use distinct
log directories and workspaces for parallel runs.

`--execution-mode auto` is the default and preserves previous behavior:
Bubblewrap for non-interactive demand forecasting, unsandboxed for other tasks.
Explicit `bubblewrap` requires Linux and bwrap and never falls back silently.
Explicit `unsandboxed` runs Python with the runner user's permissions.

The mode applies to Execute Script, Execute Experiment, and Python REPL.
Bubblewrap uses a read-only host filesystem with writable workspace and provenance
paths and a temporary /tmp. It does not disable networking or hide host files.
For tasks without forecasting features, selecting Bubblewrap moves REPL execution
into a fresh Python subprocess; in-process REPL state is not preserved.

Feature-engine loading, provenance, timeouts, receipts and scoring remain enabled
where supported in either mode. Unsandboxed auditing is not access control.
Each run records the requested and effective modes in env_log/execution_mode.json
and prints the effective mode. Pass the desired mode explicitly when resuming.
