#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ldlqe.config import DATA_DIR
from ldlqe.data import SplitConfig, default_source_path, build_splits, save_splits
from ldlqe.io_utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare data splits for labeling experiments.")
    parser.add_argument(
        "--source",
        type=Path,
        default=default_source_path(),
        help="Path to the raw newsCorpora dataset TSV file.",
    )
    parser.add_argument("--seed-per-class", type=int, default=80)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "processed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    splits = build_splits(
        SplitConfig(
            source_path=args.source,
            seed_per_class=args.seed_per_class,
            val_size=args.val_size,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    )
    save_splits(splits, args.output_dir)
    print(f"Saved processed splits to {args.output_dir}")


if __name__ == "__main__":
    main()
