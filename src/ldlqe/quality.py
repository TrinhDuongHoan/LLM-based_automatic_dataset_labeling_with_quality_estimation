from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


PROBA_COLUMNS = [
    "proba_business",
    "proba_entertainment",
    "proba_health",
    "proba_science_tech",
]


def add_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    probs = out[PROBA_COLUMNS].to_numpy()
    sorted_probs = np.sort(probs, axis=1)
    out["confidence"] = sorted_probs[:, -1]
    out["margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    out["entropy"] = -(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1)
    out["predicted_correct"] = (out["pseudo_label"] == out["label_name"]).astype(int)
    return out


def fit_quality_estimator(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[LogisticRegression, dict]:
    feature_cols = ["confidence", "margin", "entropy"]
    estimator = LogisticRegression(max_iter=1000, class_weight="balanced")
    estimator.fit(train_df[feature_cols], train_df["predicted_correct"])

    val_scores = estimator.predict_proba(val_df[feature_cols])[:, 1]
    metrics = {
        "quality_auc": float(roc_auc_score(val_df["predicted_correct"], val_scores)),
        "num_train_rows": int(len(train_df)),
        "num_val_rows": int(len(val_df)),
    }
    return estimator, metrics
