from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def evaluate_classification(y_true, y_pred, labels):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
    }
    report = pd.DataFrame(classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )).transpose()
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)
    return metrics, report, cm


def make_result_row(model_name: str, task: str, setting: str, metrics: dict, extra: dict | None = None):
    row = {
        "model_name": model_name,
        "task": task,
        "setting": setting,
        **metrics,
    }
    if extra:
        row.update(extra)
    return row