"""Score predictions against an agent-selected labeled holdout. No training is performed."""

import argparse
from pathlib import Path
from metrics import score_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("actuals", type=Path, help="Labeled holdout CSV with sku,store,week,sales_units")
    parser.add_argument("predictions", type=Path, help="CSV with sku,store,week,sales_units")
    args = parser.parse_args()
    value = score_files(args.actuals, args.predictions)
    print(f"Holdout WMAPE: {value:.8f} ({100 * value:.4f}%)")
