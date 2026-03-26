from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .io_utils import ensure_dir


def plot_model_comparison(results_df: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 6))
    ordered = results_df.sort_values("macro_f1", ascending=False)
    sns.barplot(
        data=ordered,
        x="macro_f1",
        y="experiment_name",
        hue="experiment_name",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title("Downstream Macro-F1 Comparison")
    ax.set_xlabel("Macro-F1")
    ax.set_ylabel("Labeler + Model")
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison_macro_f1.png", dpi=200)
    plt.close(fig)


def plot_quality_scatter(quality_df: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    sns.set_theme(style="ticks")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=quality_df,
        x="confidence",
        y="margin",
        hue="predicted_correct",
        alpha=0.6,
        ax=ax,
    )
    ax.set_title("Pseudo-label Confidence vs Margin")
    fig.tight_layout()
    fig.savefig(output_dir / "quality_scatter.png", dpi=200)
    plt.close(fig)


def plot_labeler_accuracy(labeler_df: pd.DataFrame, output_dir: Path) -> None:
    ensure_dir(output_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = labeler_df.sort_values("pseudo_label_accuracy", ascending=False)
    sns.barplot(
        data=ordered,
        x="pseudo_label_accuracy",
        y="labeler_name",
        hue="labeler_name",
        palette="magma",
        legend=False,
        ax=ax,
    )
    ax.set_title("Pseudo-label Accuracy by Labeler")
    ax.set_xlabel("Accuracy against gold pool labels")
    ax.set_ylabel("Labeler")
    fig.tight_layout()
    fig.savefig(output_dir / "labeler_accuracy.png", dpi=200)
    plt.close(fig)
