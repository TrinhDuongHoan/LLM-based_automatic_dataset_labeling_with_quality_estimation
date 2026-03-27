from __future__ import annotations

import pandas as pd
from .config import TEXT_COL, SENTIMENT_COL, TOPIC_COL, SENTIMENT_LABELS, TOPIC_LABELS


def load_data(train_path: str, val_path: str):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    required = {TEXT_COL, SENTIMENT_COL, TOPIC_COL}
    for name, df in [("train", train_df), ("val", val_df)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
        for col in required:
            df[col] = df[col].astype(str).str.strip()

    return train_df, val_df


def validate_labels(df: pd.DataFrame):
    bad_sent = sorted(set(df[SENTIMENT_COL]) - set(SENTIMENT_LABELS))
    bad_topic = sorted(set(df[TOPIC_COL]) - set(TOPIC_LABELS))
    if bad_sent:
        raise ValueError(f"Unexpected sentiment labels: {bad_sent}")
    if bad_topic:
        raise ValueError(f"Unexpected topic labels: {bad_topic}")


def basic_eda(train_df: pd.DataFrame, val_df: pd.DataFrame):
    overlap = len(set(train_df[TEXT_COL]).intersection(set(val_df[TEXT_COL])))
    return {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "train_sentiment": train_df[SENTIMENT_COL].value_counts().to_dict(),
        "train_topic": train_df[TOPIC_COL].value_counts().to_dict(),
        "val_sentiment": val_df[SENTIMENT_COL].value_counts().to_dict(),
        "val_topic": val_df[TOPIC_COL].value_counts().to_dict(),
        "overlap_sentences": overlap,
    }