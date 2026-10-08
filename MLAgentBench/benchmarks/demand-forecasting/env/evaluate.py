"""Score an agent-generated validation prediction file. No training is performed."""

import argparse
from pathlib import Path
from metrics import score_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, help="CSV with sku,store,week,sales_units")
    args = parser.parse_args()
    value = score_files(Path(__file__).resolve().parent / "validation.csv", args.predictions)
    print(f"Validation WMAPE: {value:.8f} ({100 * value:.4f}%)")
