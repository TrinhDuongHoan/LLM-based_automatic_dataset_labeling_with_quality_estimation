#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from ldlqe.io_utils import ensure_dir, write_json
from ldlqe.labelers import build_labelers
from ldlqe.quality import fit_quality_estimator
from ldlqe.training import build_downstream_models, train_and_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train downstream classifiers on pseudo-labeled data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--pseudo-dir", type=Path, default=Path("outputs/pseudo_labels"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    parser.add_argument("--quality-threshold", type=float, default=0.55)
    parser.add_argument("--max-pseudo-rows", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.model_dir)
    ensure_dir(args.results_dir)

    seed_df = pd.read_csv(args.data_dir / "seed.csv")
    val_df = pd.read_csv(args.data_dir / "val.csv")
    test_df = pd.read_csv(args.data_dir / "test.csv")

    experiment_rows = []
    quality_rows = []

    for labeler in build_labelers():
        pseudo_path = args.pseudo_dir / f"{labeler.name}.csv"
        pseudo_df = pd.read_csv(pseudo_path)

        quality_train = pseudo_df.sample(frac=0.8, random_state=42)
        quality_val = pseudo_df.drop(quality_train.index)
        quality_estimator, quality_metrics = fit_quality_estimator(quality_train, quality_val)
        pseudo_df["estimated_quality"] = quality_estimator.predict_proba(
            pseudo_df[["confidence", "margin", "entropy"]]
        )[:, 1]
        filtered_df = pseudo_df[pseudo_df["estimated_quality"] >= args.quality_threshold].copy()
        if filtered_df.empty:
            filtered_df = pseudo_df.copy()
        if len(filtered_df) > args.max_pseudo_rows:
            filtered_df = filtered_df.sample(n=args.max_pseudo_rows, random_state=42)

        train_df = pd.concat(
            [
                seed_df.assign(train_label=seed_df["label_name"]),
                filtered_df.assign(train_label=filtered_df["pseudo_label"]),
            ],
            ignore_index=True,
        )

        quality_record = {"labeler_name": labeler.name, **quality_metrics, "kept_rows": int(len(filtered_df))}
        quality_rows.append(quality_record)
        joblib.dump(quality_estimator, args.model_dir / f"{labeler.name}_quality_estimator.joblib")
        write_json(args.results_dir / f"{labeler.name}_quality_metrics.json", quality_record)

        for model_name, estimator in build_downstream_models().items():
            model, metrics = train_and_score(
                train_df=train_df,
                eval_df=val_df,
                labeler_name=labeler.name,
                model_name=model_name,
                estimator=estimator,
            )
            experiment_record = metrics.__dict__
            experiment_record["experiment_name"] = f"{labeler.name} + {model_name}"
            experiment_rows.append(experiment_record)
            joblib.dump(model, args.model_dir / f"{labeler.name}__{model_name}.joblib")

    pd.DataFrame(experiment_rows).sort_values("macro_f1", ascending=False).to_csv(
        args.results_dir / "validation_results.csv",
        index=False,
    )
    pd.DataFrame(quality_rows).sort_values("quality_auc", ascending=False).to_csv(
        args.results_dir / "quality_estimation_results.csv",
        index=False,
    )
    test_df.to_csv(args.results_dir / "heldout_test_reference.csv", index=False)
    print(f"Saved models to {args.model_dir} and results to {args.results_dir}")


if __name__ == "__main__":
    main()
