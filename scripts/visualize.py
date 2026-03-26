#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ldlqe.visualization import plot_labeler_accuracy, plot_model_comparison, plot_quality_scatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize labeling and downstream model results.")
    parser.add_argument("--pseudo-dir", type=Path, default=Path("outputs/pseudo_labels"))
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    parser.add_argument("--figure-dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--sample-labeler", type=str, default="ensemble_vote")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_df = pd.read_csv(args.results_dir / "validation_results.csv")
    labeler_df = pd.read_csv(args.pseudo_dir / "labeler_summary.csv")
    quality_df = pd.read_csv(args.pseudo_dir / f"{args.sample_labeler}.csv")

    plot_model_comparison(results_df, args.figure_dir)
    plot_labeler_accuracy(labeler_df, args.figure_dir)
    plot_quality_scatter(quality_df, args.figure_dir)
    print(f"Saved figures to {args.figure_dir}")


if __name__ == "__main__":
    main()
