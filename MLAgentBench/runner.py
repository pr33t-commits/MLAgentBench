""" 
This file is the entry point for MLAgentBench.
"""

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import signal
import sys
from MLAgentBench import LLM
from MLAgentBench.environment import Environment
from MLAgentBench.agents.agent import Agent, SimpleActionAgent, ReasoningActionAgent
from MLAgentBench.agents.agent_research import ResearchAgent
try:
    from MLAgentBench.agents.agent_langchain import LangChainAgent
except Exception:
    print("Failed to import LangChainAgent; its optional dependencies are not installed.")
try:
    from MLAgentBench.agents.agent_autogpt  import AutoGPTAgent
except:
    print("Failed to import AutoGPTAgent; Make sure you have installed the autogpt dependencies if you want to use it.")


def record_test_metrics(env):
    """Score frozen registered predictions after the agent has stopped.

    Execution IDs are the IDs returned to the agent by Execute Experiment.
    Never infer IDs or scores from the agent's free-form final answer.
    """
    receipt_dir = env.static_kwargs_for_tools.get("receipt_dir")
    if not receipt_dir:
        return None  # This task does not support registered experiments.

    benchmark = env.benchmark_folder_name
    benchmark_dir = Path(__file__).resolve().parent / "benchmarks" / benchmark
    scorer_name = f"MLAgentBench.benchmarks.{benchmark}.scripts.eval"
    report = {"schema_version": 1, "task": env.args.task,
              "benchmark": benchmark, "scorer": scorer_name + ".get_score",
              "experiments": {}, "selected_submission": {}}
    metadata_path = benchmark_dir / "env" / "task_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for field in ("metric", "metric_direction", "metric_scale"):
            report[field] = metadata.get(field)

    try:
        scorer = importlib.import_module(scorer_name).get_score
        scorer_error = None
    except Exception as exc:
        scorer = None
        scorer_error = f"{type(exc).__name__}: {exc}"

    def sha256(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    for receipt_path in sorted(Path(receipt_dir).glob("exec_*/receipt.json")):
        execution_id = receipt_path.parent.name
        record = {"execution_id": execution_id, "test_score": None,
                  "status": "error"}
        report["experiments"][execution_id] = record
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("execution_id") != execution_id:
                raise ValueError("Receipt execution_id does not match archive directory")
            for field in ("role", "parent_submission_id", "script_name", "accepted"):
                record[field] = receipt.get(field)
            if not receipt.get("accepted"):
                record.update(status="skipped", error=receipt.get("error", "Submission was not accepted"))
                continue
            prediction = receipt_path.parent / "submission.csv"
            artifact_hash = sha256(prediction)
            if artifact_hash != receipt.get("artifact_sha256"):
                raise ValueError("Archived submission hash does not match receipt")
            record.update(artifact_sha256=artifact_hash, submission_path=str(prediction.resolve()))
            if scorer_error:
                raise RuntimeError(scorer_error)
            score = float(scorer(str(receipt_path.parent.resolve())))
            if not math.isfinite(score):
                raise ValueError("Evaluator returned a nonfinite test score")
            record.update(status="scored", test_score=score)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"

    selected = report["selected_submission"]
    selected.update(execution_ids=[], test_score=None, status="missing")
    final_path = Path(env.work_dir) / "submission.csv"
    if final_path.exists():
        try:
            final_hash = sha256(final_path)
            matches = [key for key, value in report["experiments"].items()
                       if value.get("artifact_sha256") == final_hash and value["status"] == "scored"]
            selected.update(artifact_sha256=final_hash, execution_ids=matches,
                            status="matched" if matches else "unmatched")
            if matches:
                selected["test_score"] = report["experiments"][matches[0]]["test_score"]
            else:
                selected["error"] = "Selected submission does not match a successfully scored archive"
        except Exception as exc:
            selected.update(status="error", error=f"{type(exc).__name__}: {exc}")

    output = Path(env.log_dir) / "test_metrics.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Test metrics saved to {output}")
    return report


def run(agent_cls, args):
    env = None
    try:
        with Environment(args) as env:

            print("=====================================")
            research_problem, benchmark_folder_name = env.get_task_description()
            print("Benchmark folder name: ", benchmark_folder_name)
            print("Research problem: ", research_problem)
            print("Lower level actions enabled: ", [action.name for action in env.low_level_actions])
            print("High level actions enabled: ", [action.name for action in env.high_level_actions])
            print("Script execution:", env.static_kwargs_for_tools.get("execution_mode", "auto"))
            print("Read only files: ", env.read_only_files, file=sys.stderr)
            print("=====================================")

            agent = agent_cls(args, env)
            print(f"Agent's initial prompt :- {agent.initial_prompt}")
            final_message = agent.run(env)
            print("=====================================")
            print("Final message: ", final_message)
    finally:
        if env is not None:
            # The agent budget must not interrupt post-run scoring or saving.
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
            try:
                env.save("final")
            finally:
                try:
                    record_test_metrics(env)
                except Exception as exc:
                    print(f"Failed to record test metrics: {exc}", file=sys.stderr)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="debug", help="task name")
    parser.add_argument("--log-dir", type=str, default="./logs", help="log dir")
    parser.add_argument("--work-dir", type=str, default="./workspace", help="work dir")
    parser.add_argument("--max-steps", type=int, default=50, help="number of steps")
    parser.add_argument("--max-time", type=int, default=5* 60 * 60, help="max time")
    parser.add_argument("--device", type=int, default=0, help="device id")
    parser.add_argument("--python", type=str, default="python", help="python command")
    parser.add_argument("--execution-mode", choices=("auto", "bubblewrap", "unsandboxed"),
                        default="auto", help="Script isolation: auto preserves task defaults")
    parser.add_argument("--interactive", action="store_true", help="interactive mode")
    parser.add_argument("--resume", type=str, default=None, help="resume from a previous run")
    parser.add_argument("--resume-step", type=int, default=0, help="the step to resume from")

    # general agent configs
    parser.add_argument("--agent-type", type=str, default="ResearchAgent", help="agent type")
    parser.add_argument("--llm-name", type=str, default="claude-v1", help="llm name")
    parser.add_argument("--enable-thinking", type=str.lower, choices=("true", "false"),
                        default="false", help="enable thinking in Hugging Face chat templates")
    parser.add_argument("--fast-llm-name", type=str, default="claude-v1", help="llm name")
    parser.add_argument("--edit-script-llm-name", type=str, default="claude-v1", help="llm name")
    parser.add_argument("--edit-script-llm-max-tokens", type=int, default=20000, help="llm max tokens")
    parser.add_argument("--agent-max-steps", type=int, default=50, help="max iterations for agent")

    # research agent configs
    parser.add_argument("--actions-remove-from-prompt", type=str, nargs='+', default=[], help="actions to remove in addition to the default ones: Read File, Write File, Append File, Retrieval from Research Log, Append Summary to Research Log, Python REPL, Edit Script Segment (AI)")
    parser.add_argument("--actions-add-to-prompt", type=str, nargs='+', default=[], help="actions to add")
    parser.add_argument("--retrieval", action="store_true", help="enable retrieval")
    parser.add_argument("--valid-format-entires", type=str, nargs='+', default=None, help="valid format entries")
    parser.add_argument("--max-steps-in-context", type=int, default=3, help="max steps in context")
    parser.add_argument("--max-observation-steps-in-context", type=int, default=3, help="max observation steps in context")
    parser.add_argument("--max-retries", type=int, default=5, help="max retries")

    # langchain configs
    parser.add_argument("--langchain-agent", type=str, default="zero-shot-react-description", help="langchain agent")


    args = parser.parse_args()
    print(args, file=sys.stderr)
    if not args.retrieval or args.agent_type != "ResearchAgent":
        # should not use these actions when there is no retrieval
        args.actions_remove_from_prompt.extend(["Retrieval from Research Log", "Append Summary to Research Log", "Reflection"])
    LLM.FAST_MODEL = args.fast_llm_name
    LLM.HF_ENABLE_THINKING = args.enable_thinking == "true"
    run(getattr(sys.modules[__name__], args.agent_type), args)
