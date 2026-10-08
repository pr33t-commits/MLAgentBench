"""Probe the execution interpreter, not the agent host's installed packages."""
import json
import subprocess

MODULES = ["numpy", "pandas", "scipy", "sklearn", "lightgbm", "xgboost",
           "statsmodels", "matplotlib", "seaborn", "joblib", "tqdm", "pyarrow",
           "torch", "tensorflow"]


def build_library_context(python):
    code = '''import contextlib, importlib, io, json, sys
result = {"python": sys.executable, "version": sys.version.split()[0], "libraries": {}}
for name in json.loads(sys.argv[1]):
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            module = importlib.import_module(name)
        result["libraries"][name] = {"available": True, "version": str(getattr(module, "__version__", "unknown"))}
    except Exception as exc:
        result["libraries"][name] = {"available": False, "error": type(exc).__name__ + ": " + str(exc)[:300]}
print("LIBRARY_CONTEXT=" + json.dumps(result))
'''
    try:
        process = subprocess.run([python, "-c", code, json.dumps(MODULES)],
                                 capture_output=True, text=True, timeout=60)
        lines = [line for line in process.stdout.splitlines() if line.startswith("LIBRARY_CONTEXT=")]
        if process.returncode or not lines:
            raise ValueError("Interpreter probe did not complete successfully")
        result = json.loads(lines[-1].split("=", 1)[1])
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return f"Execution library availability could not be verified ({type(exc).__name__}). Do not assume optional libraries are installed."
    return ("Execution environment libraries (import-tested once at startup using --python):\n"
            + json.dumps(result, indent=2)
            + "\nUse imports marked available; sklearn is the import name for scikit-learn. "
            "Do not pip-install packages from generated scripts. Versions determine supported APIs. "
            "An available import does not guarantee GPU support. Unlisted libraries are unverified.")
