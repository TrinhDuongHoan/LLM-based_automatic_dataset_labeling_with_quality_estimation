from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .io_utils import ensure_dir


def save_classification_outputs(
    output_dir: Path,
    y_true,
    y_pred,
    labels: list[str],
    prefix: str,
) -> None:
    ensure_dir(output_dir)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(output_dir / f"{prefix}_classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(output_dir / f"{prefix}_confusion_matrix.csv")
