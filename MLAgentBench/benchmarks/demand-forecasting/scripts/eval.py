"""MLAgentBench evaluator; hidden labels stay outside the agent workspace."""

import argparse
import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
# Load only the benchmark's trusted scorer, never code from a submission.
spec = importlib.util.spec_from_file_location("demand_forecasting_scoring", SCRIPTS / "scoring.py")
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)

def get_score(submission_folder=None):
    folder = Path(submission_folder) if submission_folder is not None else SCRIPTS.parent / "env"
    return scoring.score_files(SCRIPTS / "answer.csv", folder / "submission.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hidden test WMAPE (lower is better)")
    parser.add_argument("submission_folder", type=Path)
    args = parser.parse_args()
    print(get_score(args.submission_folder))
