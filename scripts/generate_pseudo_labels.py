#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ldlqe.io_utils import ensure_dir
from ldlqe.labelers import build_labelers
from ldlqe.quality import add_quality_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pseudo-labels with multiple labelers.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pseudo_labels"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    seed_df = pd.read_csv(args.data_dir / "seed.csv")
    unlabeled_df = pd.read_csv(args.data_dir / "unlabeled_pool.csv")

    summary_rows = []
    for labeler in build_labelers():
        fitted = labeler.fit(seed_df)
        pred_df = fitted.predict(unlabeled_df).predictions
        merged = pd.concat([unlabeled_df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
        merged = add_quality_features(merged)
        merged.to_csv(args.output_dir / f"{labeler.name}.csv", index=False)

        summary_rows.append(
            {
                "labeler_name": labeler.name,
                "pseudo_label_accuracy": float(accuracy_score(merged["label_name"], merged["pseudo_label"])),
                "pseudo_label_macro_f1": float(f1_score(merged["label_name"], merged["pseudo_label"], average="macro")),
                "rows": int(len(merged)),
            }
        )

    pd.DataFrame(summary_rows).sort_values("pseudo_label_macro_f1", ascending=False).to_csv(
        args.output_dir / "labeler_summary.csv",
        index=False,
    )
    print(f"Saved pseudo-label outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
