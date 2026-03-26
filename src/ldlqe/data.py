from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, DEFAULT_COLUMNS, LABEL_MAP
from .io_utils import ensure_dir, write_json


@dataclass
class SplitConfig:
    source_path: Path
    seed_per_class: int = 80
    val_size: float = 0.1
    test_size: float = 0.2
    random_state: int = 42


def load_newscorpora(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=DEFAULT_COLUMNS)
    df["label_name"] = df["label"].map(LABEL_MAP)
    df["text"] = df["title"].fillna("") + " [SEP] " + df["publisher"].fillna("")
    df = df.dropna(subset=["label_name", "text"]).reset_index(drop=True)
    return df


def build_splits(cfg: SplitConfig) -> dict[str, pd.DataFrame]:
    df = load_newscorpora(cfg.source_path)

    train_pool, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        stratify=df["label_name"],
        random_state=cfg.random_state,
    )
    train_df, val_df = train_test_split(
        train_pool,
        test_size=cfg.val_size,
        stratify=train_pool["label_name"],
        random_state=cfg.random_state,
    )

    seed_parts = []
    unlabeled_parts = []
    for _, group in train_df.groupby("label_name", sort=True):
        seed_group = group.sample(
            n=min(cfg.seed_per_class, len(group)),
            random_state=cfg.random_state,
        )
        unlabeled_group = group.drop(seed_group.index)
        seed_parts.append(seed_group)
        unlabeled_parts.append(unlabeled_group)

    seed_df = pd.concat(seed_parts).sample(frac=1.0, random_state=cfg.random_state)
    unlabeled_df = pd.concat(unlabeled_parts).sample(frac=1.0, random_state=cfg.random_state)

    return {
        "seed": seed_df.reset_index(drop=True),
        "unlabeled_pool": unlabeled_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def save_splits(splits: dict[str, pd.DataFrame], output_dir: Path) -> None:
    ensure_dir(output_dir)
    summary = {}
    for split_name, split_df in splits.items():
        path = output_dir / f"{split_name}.csv"
        split_df.to_csv(path, index=False)
        summary[split_name] = {
            "rows": int(len(split_df)),
            "label_distribution": split_df["label_name"].value_counts().to_dict(),
            "path": str(path),
        }
    write_json(output_dir / "split_summary.json", summary)


def default_source_path() -> Path:
    return DATA_DIR / "newsCorpora.csv"
