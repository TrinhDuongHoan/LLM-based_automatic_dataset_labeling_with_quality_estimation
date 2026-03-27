from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_eval_artifacts(out_dir, prefix, predictions_df, metrics, report_df, cm_df):
    out_dir = ensure_dir(out_dir)
    predictions_df.to_csv(out_dir / f"{prefix}_predictions.csv", index=False)
    with open(out_dir / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    report_df.to_csv(out_dir / f"{prefix}_report.csv", index=True)
    cm_df.to_csv(out_dir / f"{prefix}_confusion.csv", index=True)


def append_leaderboard(out_dir, row: dict):
    out_dir = ensure_dir(out_dir)
    path = out_dir / "leaderboard.csv"
    df = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)