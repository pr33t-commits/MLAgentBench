"""Bounded subprocess execution with preserved output and feature-stage diagnostics."""
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

SCRIPT_TIMEOUT = 600
MARKER = "[MLAB_BUILD] "


def run_script(command, cwd, env, timeout=SCRIPT_TIMEOUT, session_deadline=None):
    started = time.monotonic()
    remaining = lambda: None if session_deadline is None else max(0, session_deadline - time.time())
    before = remaining()
    limit = min(timeout, SCRIPT_TIMEOUT, before if before is not None else SCRIPT_TIMEOUT)
    reason = None
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=(os.name != "nt"))
        try:
            process.wait(timeout=max(0, limit))
        except subprocess.TimeoutExpired:
            reason = "session_timeout" if before is not None and before <= min(timeout, SCRIPT_TIMEOUT) else "script_timeout"
        except Exception as exc:
            if type(exc).__name__ != "TimeoutException":
                raise
            reason = "session_timeout"
        finally:
            if process.poll() is None or reason:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait()
        stdout.seek(0)
        stderr.seek(0)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
    active = {}
    builds = []
    for line in err.splitlines():
        if not line.startswith(MARKER):
            continue
        try:
            event = json.loads(line[len(MARKER):])
            builds.append(event)
            if event["event"] == "start":
                active[event["call_id"]] = event
            else:
                active.pop(event["call_id"], None)
        except (ValueError, KeyError, TypeError):
            continue
    for event in active.values():
        if "started_monotonic" in event:
            event["elapsed_seconds"] = time.monotonic() - event["started_monotonic"]
    stage = "engine.build()" if active else "outside engine.build()"
    feedback = None
    if reason:
        feedback = (f"{reason}: execution stopped after {time.monotonic() - started:.2f}s "
                    f"(script limit {min(timeout, SCRIPT_TIMEOUT)}s), during {stage}. "
                    "The script process group was terminated; partial stdout/stderr are preserved.")
        if active:
            feedback += " An engine.build() call was still active; inspect its request size and timing before retrying."
    return dict(exit_code=process.returncode, ok=reason is None and process.returncode == 0,
                timed_out=reason is not None, timeout_reason=reason, timeout_stage=stage if reason else None,
                feedback=feedback, stdout=out, stderr=err, build_timings=builds,
                active_builds=list(active.values()), elapsed_seconds=time.monotonic() - started,
                session_remaining_before_seconds=before, session_remaining_after_seconds=remaining())
