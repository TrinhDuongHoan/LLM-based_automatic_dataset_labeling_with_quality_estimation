from __future__ import annotations

from collections import Counter
import pandas as pd
from .metrics import evaluate_classification


def majority_vote(values):
    c = Counter([v for v in values if pd.notna(v)])
    if not c:
        return None
    return c.most_common(1)[0][0]


def ensemble_predictions(val_df, prediction_files, task):
    merged = val_df[["sentence", task]].copy()
    pred_cols = []

    for i, path in enumerate(prediction_files):
        df = pd.read_csv(path)
        candidates = [f"pred_{task}", task]
        pred_col = next((c for c in candidates if c in df.columns), None)
        if pred_col is None:
            raise ValueError(f"No prediction column for task={task} in {path}")
        col_name = f"model_{i}_{task}"
        merged = merged.merge(df[["sentence", pred_col]].rename(columns={pred_col: col_name}), on="sentence", how="left")
        pred_cols.append(col_name)

    merged[f"pred_{task}"] = merged[pred_cols].apply(lambda r: majority_vote(r.tolist()), axis=1)
    return merged