""" This file contains the low level actions that are provided by the environment, mostly file system operations and code execution. """

import os
import json
import uuid
from .execution_runtime import run_script, SCRIPT_TIMEOUT
from .provenance import execution_command
from .completeness import (execute_experiment as run_experiment,
                           read_experiment_log, list_experiments)
import subprocess
import selectors
import shutil
import glob
import sys
import inspect
from functools import wraps
import time
from io import StringIO
from .schema import Step, ActionInfo, Action, EnvException
import readline # This is needed to make sure that the input() function works properly


def normalize_args_kwargs(f, *args, **kwargs):
    """ This function takes a function and its arguments and returns a dictionary of the arguments, with the keys being the argument names."""
    sig = inspect.signature(f)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()  # This line is optional, it fills in any omitted arguments that have default values
    return bound.arguments

def append_to_low_level_steps(trace, name, args, observation):
    """ This function appends a low level step to the trace. """
    trace.low_level_steps.append(Step(action=Action(name, args),observation=observation,timestamp=time.time()))


def record_low_level_step(func):
    """ This decorator records a low level step in the trace."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        new_kwargs = normalize_args_kwargs(func, *args, **kwargs)
        if "trace" not in new_kwargs["kwargs"]:
            print("Warning: trace not found in kwargs; not recording low level step.")
            print(func)
            return func(*args, **kwargs)
        else:
            trace = new_kwargs["kwargs"]["trace"]
            for a in LOW_LEVEL_ACTIONS:
                if a.function.__name__ == func.__name__:
                    name = a.name
                    input_args = a.usage.keys()
                    break
            new_kwargs = {k: v for k, v in new_kwargs.items() if k in input_args}
            try:
                observation = func(*args, **kwargs)
                append_to_low_level_steps(trace, name, new_kwargs, observation)
                return observation
            except EnvironmentError as e:
                append_to_low_level_steps(trace, name, new_kwargs, e)
                raise EnvException(e)
    return wrapper


def check_file_read_only(arg_names, **kwargs):
    """ This decorator checks if the file is read-only. """
    
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            new_kwargs = normalize_args_kwargs(func, *args, **kwargs)
            for arg_name in arg_names:
                if new_kwargs[arg_name] in new_kwargs["kwargs"]["read_only_files"]:
                    raise EnvException(f"cannot write file {new_kwargs[arg_name]} because it is a read-only file.")
            return func(*args, **kwargs)
        return wrapper
    return inner


def check_file_in_work_dir(arg_names, **kwargs):
    """ This decorator checks if the file is in the work directory. """
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            new_kwargs = normalize_args_kwargs(func, *args, **kwargs)
            work_dir = new_kwargs["work_dir"]
            for arg_name in arg_names:
                file_name = new_kwargs[arg_name]
                if not os.path.abspath(os.path.join(work_dir, file_name)).startswith(os.path.abspath(work_dir)):
                    raise EnvException(f"cannot access file {file_name} because it is not in the work directory.")
            return func(*args, **kwargs)
        return wrapper
    return inner


@check_file_in_work_dir(["dir_path"])
@record_low_level_step
def list_files( dir_path, work_dir = ".", **kwargs):
    try:
        observation = subprocess.check_output(["ls", "-F", os.path.join(work_dir,dir_path)]).decode("utf-8")
        return observation
    except:
        raise EnvException(f"Cannot list file in the {dir_path} directory")

    



@check_file_in_work_dir(["file_name"])
@record_low_level_step
def read_file(file_name, work_dir = '.', **kwargs):
    try:
        observation = open(os.path.join(work_dir,file_name)).read()
        return observation
    except:
        raise EnvException(f"cannot read file {file_name}")


@check_file_in_work_dir(["file_name"])
@check_file_read_only(["file_name"])
@record_low_level_step
def write_file(file_name, content, work_dir = ".", **kwargs):
    try:
        with open(os.path.join(work_dir,file_name), "w") as f:
            f.write(content)
        observation = f"File {file_name} written successfully."
        return observation
    except:
        raise EnvException(f"cannot write file {file_name}")


@check_file_in_work_dir(["file_name"])
@check_file_read_only(["file_name"])
@record_low_level_step
def append_file(file_name, content, work_dir = ".", **kwargs):
    try:
        with open(os.path.join(work_dir,file_name), "a") as f:
            f.write(content)
        observation = f"File {file_name} appended successfully."
        return observation
    except:
        raise EnvException(f"cannot append file {file_name}")


@check_file_in_work_dir(["source", "destination"])
@check_file_read_only(["destination"])
@record_low_level_step
def copy_file( source, destination, work_dir = ".", **kwargs):
    
    try:
        shutil.copyfile(os.path.join(work_dir,source), os.path.join(work_dir,destination))
        observation = f"File {source} copied to {destination}"
        return observation
    except:
        raise EnvException(f"File {source} copy to {destination} failed. Check whether the source and destinations are valid.")


@check_file_in_work_dir(["script_name"])
@record_low_level_step
def undo_edit_script( script_name, work_dir = ".", **kwargs):
    
    backup_files = glob.glob(os.path.join(work_dir,"backup", f"{script_name}_*"))
    if len(backup_files) == 0:
        raise EnvException("There is no change to undo.")
    try:
        backup_files.sort()
        backup_file = backup_files[-1]
        shutil.copyfile(backup_file, os.path.join(work_dir,script_name))
        # delete the backup file
        os.remove(backup_file)

        new_content = open(os.path.join(work_dir,script_name)).read()
        observation = f"Content of {script_name} after undo the most recent edit:\n" + new_content
        return observation
    except:
        raise EnvException(f"Cannot undo the edit of file name {script_name}. Check the file name again."
        )


@check_file_in_work_dir(["script_name"])
@record_low_level_step
def execute_script(script_name, work_dir = ".", **kwargs):
    if not os.path.exists(os.path.join(work_dir,script_name)):
        raise EnvException(f"The file {script_name} does not exist.")
    try:
        script_path = script_name
        device = kwargs["device"]
        python = kwargs["python"]
        cmd, process_env = execution_command(python, os.path.abspath(os.path.join(work_dir, script_path)),
                                             work_dir, kwargs.get("provenance_log"), kwargs.get("provenance_dir"),
                                             protected_features=kwargs.get("protected_features", False))
        process_env["CUDA_VISIBLE_DEVICES"] = str(device)
        execution_id = "exec_" + uuid.uuid4().hex
        started_at = time.time()
        result = run_script(cmd, work_dir, process_env,
                            session_deadline=kwargs.get("session_deadline"))
        result.update(execution_id=execution_id, started_at=started_at, finished_at=time.time())
        result["output"] = result["stdout"] if result["ok"] else result["stderr"]
        if result["feedback"]:
            result["output"] = result["feedback"] + "\n" + result["stdout"] + result["stderr"]
        if kwargs.get("log_file"):
            with open(kwargs["log_file"], "w", encoding="utf-8") as log:
                json.dump(result, log, indent=2)
        return json.dumps(result)
    except Exception as e:
        raise EnvException(f"Something went wrong in executing {script_name}: {e}. Please check if it is ready to be executed.")


@record_low_level_step
def execute_experiment(script_name, role, parent_submission_id, work_dir=".", **kwargs):
    """Execute and archive a demand-forecasting submission."""
    root = kwargs.get("receipt_dir")
    if not root:
        raise EnvException("Execute Experiment is only enabled for demand-forecasting")
    try:
        receipt = run_experiment(script_name, role, parent_submission_id, work_dir,
                                 root, kwargs["python"],
                                 timeout=SCRIPT_TIMEOUT, session_deadline=kwargs.get("session_deadline"),
                                 device=kwargs.get("device", ""),
                                 provenance_log=kwargs.get("provenance_log"),
                                 provenance_dir=kwargs.get("provenance_dir"),
                                 protected_features=kwargs.get("protected_features", False))
        return json.dumps({k: v for k, v in receipt.items() if k != "source_hashes"})
    except (ValueError, OSError, KeyError) as exc:
        raise EnvException(str(exc)) from exc


@record_low_level_step
def inspect_experiment_log(execution_id, stream, offset, **kwargs):
    try:
        return json.dumps(read_experiment_log(kwargs["receipt_dir"], execution_id, stream, offset))
    except (ValueError, OSError, KeyError) as exc:
        raise EnvException(str(exc)) from exc


@record_low_level_step
def list_experiment_records(**kwargs):
    try:
        return json.dumps(list_experiments(kwargs["receipt_dir"]))
    except (ValueError, OSError, KeyError) as exc:
        raise EnvException(str(exc)) from exc


@record_low_level_step
def python_repl(command, work_dir = ".", **kwargs):
    """Run command and returns anything printed."""
    if kwargs.get("protected_features"):
        from pathlib import Path
        script = Path(work_dir).resolve() / (".repl_" + uuid.uuid4().hex + ".py")
        script.write_text(command, encoding="utf-8")
        try:
            cmd, env = execution_command(kwargs.get("python", sys.executable), script, work_dir,
                                         kwargs.get("provenance_log"), kwargs.get("provenance_dir"),
                                         protected_features=True)
            result = subprocess.run(cmd, cwd=work_dir, env=env, capture_output=True, text=True,
                                    timeout=kwargs.get("timeout", 600))
            return result.stdout + result.stderr
        finally:
            script.unlink(missing_ok=True)
    try:
        cwd = os.getcwd()
        import codeop
        compiler = codeop.CommandCompiler()
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            command = compiler(command)
            os.chdir(work_dir)
            exec(command, globals())
            sys.stdout = old_stdout
            output = mystdout.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            output = str(e)
        os.chdir(cwd)
        return output
    except Exception as e:
        raise EnvException(f"Something went wrong in executing {command}: {e}")


@record_low_level_step
def request_help(request, work_dir = ".", **kwargs):
    return input(f"Research Assistant is requesting help: {request}\n")


### describe the low level actions
LOW_LEVEL_ACTIONS = [
    ActionInfo(
        name="List Files",
        description="Use this to navigate the file system.",
        usage={   
            "dir_path": "a valid relative path to a directory, such as \".\" or \"folder1/folder2\""
        },
        return_value="The observation will be a list of files and folders in dir_path or current directory is dir_path is empty, or an error message if dir_path is invalid.",
        function=list_files,
        is_primitive=True
    ),
    ActionInfo(
        name="Read File",
        description="Use this to read an existing file.",
        usage={
            "file_name": "a valid file name with relative path to current directory if needed"
        },
        return_value="The observation will be the contents of the file read.",
        function=read_file,
        is_primitive=True
    ),
    ActionInfo(
        name="Write File",
        description="Use this to write a file. If the file already exists, it will be overwritten.",
        usage={
            "file_name": "a valid file name with relative path to current directory if needed",
            "content": "the content to be written to the file"
        },
        return_value="A success message if the file is written successfully, or an error message if the file cannot be written.",
        function=write_file,
        is_primitive=True
    ),
    ActionInfo(
        name="Append File",
        description="Use this to append a file to a new location with a new name.",
        usage={
            "file_name": "a valid file name with relative path to current directory if needed",
            "content": "the content to be appended to the file"
        },
        return_value="A success message if the file is appended successfully, or an error message if the file cannot be appended.",
        function=append_file,
        is_primitive=True
    ),
    ActionInfo(
        name="Copy File",
        description="Use this to copy a file to a new location with a new name.",
        usage={
            "source": "a valid file name with relative path to current directory if needed",
            "destination": "a valid file name with relative path to current directory if needed"
        },
        return_value="A success message if the file is copied successfully, or an error message if the file cannot be copied.",
        function=copy_file,
        is_primitive=True
    ),
    ActionInfo(
        name="Undo Edit Script",
        description="Use this to undo the last edit of the python script.",
        usage={
            "script_name": "a valid python script name with relative path to current directory if needed"
        },
        return_value="The observation will be the content of the script before the last edit. If the script does not exist, the observation will be an error message.",
        function=undo_edit_script,
        is_primitive=True
    ),
    ActionInfo(
        name="Execute Script",
        description="Use this to execute the python script. The script must already exist. Execution has a 600-second limit, or the remaining session budget if shorter. Timeout feedback includes partial output and whether engine.build() was active.",
        usage={
            "script_name": "a valid python script name with relative path to current directory if needed"
        },
        return_value="The observation will be output of the script or errors.",
        function=execute_script,
        is_primitive=True
    ),
    ActionInfo(
        name="Execute Experiment",
        description="Run a baseline or experiment and register its test submission. Execution has a 600-second limit, or the remaining session budget if shorter; receipts include engine.build() timing and timeout diagnostics. Read inputs from the workspace. The script MUST write submission.csv under os.environ['EXPERIMENT_OUTPUT_DIR']. Outputs are fresh for each execution. An experiment requires an accepted baseline execution ID. Returns acceptance status, exit code, stdout and stderr (tails when large). For rejected runs, inspect stderr before editing; use Read Experiment Log for truncated output. Use List Experiments to recover baseline IDs. No hidden test score is exposed.",
        usage={"script_name": "existing Python script relative to workspace",
               "role": "baseline or experiment",
               "parent_submission_id": "accepted baseline execution_id for experiment; empty string for baseline"},
        return_value="Execution receipt including accepted status, errors, source hashes and artifact hash.",
        function=execute_experiment,
        is_primitive=True
    ),
    ActionInfo(
        name="Read Experiment Log",
        description="Read archived stdout or stderr for an execution. Start with offset 0; use next_offset until null to retrieve the entire log. Read the actual traceback before proposing fixes. Does not expose hidden labels or allow arbitrary paths.",
        usage={"execution_id": "exact ID from Execute Experiment or List Experiments",
               "stream": "stdout or stderr", "offset": "nonnegative byte offset, initially 0"},
        return_value="Log text, total_bytes and next_offset (null at end).",
        function=inspect_experiment_log,
        is_primitive=True
    ),
    ActionInfo(
        name="List Experiments",
        description="Recover prior experiment IDs, script names and acceptance status for this run. Use an accepted baseline ID as parent_submission_id; never guess an ID or recreate a baseline just because its receipt left context.",
        usage={},
        return_value="Execution records in chronological order.",
        function=list_experiment_records,
        is_primitive=True
    ),
    ActionInfo(
        name="Python REPL",
        description="A python REPL. Use this to execute single line python commands.",
        usage={
            "command": "a valid python command"
        },
        return_value="The observation will be output of the command or errors.",
        function=python_repl,
        is_primitive=True 
    ),
    ActionInfo(
        name="Request Help",
        description="Use this to request help from human. Use this only when the provided tools and files are not enough for accomplishing necessary steps, such as requesting API reference or installing a library. So you should check through the provided tools and files first.",
        usage={
            "request": "a detailed description on what to do"
        },
        return_value="The observation will be the response from human.",
        function=request_help,
        is_primitive=True
    ),
    ActionInfo(
        name="Final Answer",
        description="Use this to provide the final answer to the current task.",
        usage={
            "final_answer": "a detailed description on the final answer"
        },
        return_value="The observation will be empty.",
        function=(lambda **kwargs: ""),
        is_primitive=True
    ),
]
