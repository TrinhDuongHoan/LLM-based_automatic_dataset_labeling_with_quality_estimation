#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from ldlqe.evaluation import save_classification_outputs
from ldlqe.io_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the best trained model on the held-out test set.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_df = pd.read_csv(args.results_dir / "validation_results.csv")
    best_row = results_df.sort_values("macro_f1", ascending=False).iloc[0]

    labeler_name = best_row["labeler_name"]
    model_name = best_row["model_name"]
    experiment_name = f"{labeler_name}__{model_name}"
    model = joblib.load(args.model_dir / f"{experiment_name}.joblib")

    test_df = pd.read_csv(args.data_dir / "test.csv")
    y_pred = model.predict(test_df["text"])
    labels = sorted(test_df["label_name"].unique())
    save_classification_outputs(args.results_dir, test_df["label_name"], y_pred, labels, prefix="best_test")

    payload = {
        "best_labeler": labeler_name,
        "best_model": model_name,
        "validation_macro_f1": float(best_row["macro_f1"]),
        "validation_accuracy": float(best_row["accuracy"]),
    }
    write_json(args.results_dir / "best_model_summary.json", payload)
    print(f"Evaluated best model: {experiment_name}")


if __name__ == "__main__":
    main()
