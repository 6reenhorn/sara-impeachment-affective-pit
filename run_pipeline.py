"""One-command pipeline runner for Stages 1b through 5.

Usage:
  python run_pipeline.py                  # Runs merge -> clean -> lexicons -> ML (auto) -> comparative analysis
  python run_pipeline.py --prepare-sheets # Also generates/regenerates the 5-member annotation sheets
  python run_pipeline.py --mode gold      # Requires data/annotated/gold_labels.csv
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(script: str, extra_args: list[str] | None = None) -> None:
    cmd = [sys.executable, "-u", script] + (extra_args or [])
    print(f"\n>>> Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise SystemExit(f"Step failed ({res.returncode}): {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-sheets", action="store_true", help="(Re)generate 5-member annotation sheets")
    parser.add_argument("--mode", choices=["auto", "gold", "provisional"], default="auto")
    args = parser.parse_args()

    reddit_json_dir = Path("data/raw/reddit_json")
    if reddit_json_dir.exists() and list(reddit_json_dir.rglob("*.json")):
        run_step("src/01_mining/parse_reddit_json.py")

    run_step("src/01_mining/merge_raw_datasets.py")
    run_step("src/02_preprocessing/clean_comments.py")
    run_step("src/03_lexicon/run_lexicons.py")

    if args.prepare_sheets or not Path("data/annotated/sheets/member_1.csv").exists():
        run_step("src/04_modeling/prepare_annotation_sheets.py")

    run_step("src/04_modeling/train_ml_models.py", ["--mode", args.mode])
    run_step("src/05_analysis/comparative_analysis.py")
    print("\nPipeline complete! All outputs written to deliverables/ and data/.")


if __name__ == "__main__":
    main()
