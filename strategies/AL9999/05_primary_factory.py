"""
05_primary_factory.py - Step 5: Primary Model Factory runner.

输入:
  - output/bars/*.parquet 中的 dollar bars
  - output/features/trend_labels.parquet

输出:
  - output/primary_search/cusum_calibration.parquet
  - output/primary_search/scoring_lightweight.csv
  - output/primary_search/scoring_deep.csv
  - output/primary_search/scoring_final.csv
  - output/primary_search/top_candidates.parquet
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.AL9999.config import BARS_DIR, FEATURES_DIR, OUTPUT_DIR, PRIMARY_FACTORY_CONFIG
from strategies.AL9999.primary_factory import run_primary_factory


def _resolve_default_bars_path() -> str:
    candidates = sorted(glob.glob(os.path.join(BARS_DIR, "*.parquet")))
    if not candidates:
        raise FileNotFoundError(f"No parquet bars found under {BARS_DIR}")
    return candidates[0]


def main():
    """
    Run the AL9999 primary factory pipeline.
    """
    parser = argparse.ArgumentParser(description="Run AL9999 primary signal factory.")
    parser.add_argument(
        "--bars-path",
        default=None,
        help="Dollar bars parquet path. Defaults to the first parquet file in output/bars.",
    )
    parser.add_argument(
        "--trend-labels-path",
        default=os.path.join(FEATURES_DIR, "trend_labels.parquet"),
        help="Trend labels parquet path.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Root output directory.",
    )
    args = parser.parse_args()

    bars_path = args.bars_path or _resolve_default_bars_path()

    print("[Primary Factory] 开始运行...")
    print(f"  bars_path: {bars_path}")
    print(f"  trend_labels_path: {args.trend_labels_path}")
    print(f"  output_dir: {args.output_dir}")

    _, top_candidates, _ = run_primary_factory(
        bars_path=bars_path,
        trend_labels_path=args.trend_labels_path,
        output_dir=args.output_dir,
        config=PRIMARY_FACTORY_CONFIG,
    )

    print("\n[Primary Factory] Top 候选:")
    print(top_candidates[["rank", "combo_id", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
