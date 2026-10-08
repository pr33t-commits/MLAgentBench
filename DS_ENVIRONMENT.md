# Agent execution libraries

Install the common data-science stack into the exact interpreter passed to the
runner's `--python` option, not merely the interpreter running the main agent:

```bash
/path/to/sandbox/bin/python -m pip install -r requirements-ds.txt
/path/to/sandbox/bin/python -m pip check
```

The requirements cover NumPy, pandas, SciPy, scikit-learn (`import sklearn`),
LightGBM, XGBoost, statsmodels, matplotlib, seaborn, joblib, tqdm and PyArrow.
On Linux, LightGBM may also require the OS OpenMP runtime (`libgomp1`).
These are unpinned additions; resolve them in your target Python environment
and retain its dependency lock/freeze for reproducible runs.

Each Environment startup probes these libraries plus torch and tensorflow in
the execution interpreter. The main agent and Create/Edit Script coding prompts
receive actual import availability and versions. Broken imports are marked
unavailable. Probe failure/timeouts report unverified availability. The probe
runs once per environment; restart the runner after installing libraries.

For the separate DS_Agent Docker executor, `sandbox/Dockerfile` now installs
the same common stack and libgomp1. Rebuild from DS_Agent with:

```bash
docker build -t ds-understanding-python:local sandbox
```

Use the image tag configured for your Docker executor if different. That image
is separate from MLAgentBench's `--python` interpreter; rebuilding it does not
install packages into an existing MLAgentBench virtual environment.
