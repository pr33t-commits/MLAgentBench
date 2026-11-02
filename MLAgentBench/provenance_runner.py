"""Standalone Python execution bootstrap with cooperative file-access auditing.

Not a sandbox: native I/O and child processes are not fully observed. The parent
records these limitations explicitly. No generated program imports are required.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time


def main():
    root = Path(os.environ["MLAB_PROVENANCE_WORKSPACE"]).resolve()
    log = Path(os.environ["MLAB_PROVENANCE_LOG"])
    objects = Path(os.environ["MLAB_PROVENANCE_OBJECTS"])
    objects.mkdir(parents=True, exist_ok=True)
    busy = False
    seen = set()

    def emit(record):
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(record, timestamp=time.time())) + "\n")

    def hook(event, args):
        nonlocal busy
        if busy:
            return
        busy = True
        try:
            if event in {"subprocess.Popen", "os.system", "os.fork", "os.exec"}:
                emit({"kind": "coverage_gap", "reason": "child_process", "event": event})
            elif event in {"open", "exec"}:
                name = args[0] if event == "open" else getattr(args[0], "co_filename", None)
                if not isinstance(name, (str, bytes)):
                    return
                path = Path(os.fsdecode(name)).resolve()
                if not path.is_relative_to(root):
                    # Package/runtime reads are recorded as external paths, not
                    # copied: credentials and other host files must not be archived.
                    key = ("external", str(path))
                    if key not in seen:
                        emit({"kind": "external_access", "path": str(path)})
                        seen.add(key)
                    return
                rel = path.relative_to(root).as_posix()
                if "__pycache__" in path.parts:
                    return
                mode = args[1] if event == "open" else "r"
                flags = args[2] if event == "open" else 0
                writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                    isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)))
                if writing:
                    emit({"kind": "write_attempt", "path": rel})
                reading = event == "exec" or not writing or (isinstance(mode, str) and "+" in mode) or (
                    isinstance(flags, int) and bool(flags & os.O_RDWR))
                if reading and path.is_file():
                    # Freeze the bytes at the access attempt, including files
                    # created and subsequently removed within one action.
                    h = hashlib.sha256()
                    temporary = objects / ("capture-" + str(os.getpid()))
                    with path.open("rb") as source, temporary.open("wb") as dest:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            h.update(block)
                            dest.write(block)
                    digest = h.hexdigest()
                    target = objects / digest
                    if target.exists():
                        temporary.unlink()
                    else:
                        temporary.replace(target)
                    key = (rel, digest)
                    if key not in seen:
                        emit({"kind": "read_attempt", "path": rel, "sha256": digest})
                        seen.add(key)
        except Exception as exc:
            emit({"kind": "coverage_gap", "reason": type(exc).__name__})
        finally:
            busy = False

    emit({"kind": "audit_started"})
    sys.addaudithook(hook)
    if os.environ.get("MLAB_FEATURE_ENGINE") == "1":
        import importlib.util
        utility = Path(__file__).with_name("forecast_features.py")
        spec = importlib.util.spec_from_file_location("forecast_features", utility)
        module = importlib.util.module_from_spec(spec)
        sys.modules["forecast_features"] = module
        spec.loader.exec_module(module)
        emit({"kind": "trusted_feature_engine", "sha256": hashlib.sha256(utility.read_bytes()).hexdigest()})
    script = str(Path(sys.argv[1]).resolve())
    sys.argv = [script] + sys.argv[2:]
    sys.path.insert(0, str(Path(script).parent))
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        busy = True
        emit({"kind": "audit_finished"})


if __name__ == "__main__":
    main()
