from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


@dataclass
class ExperimentResult:
    labeler_name: str
    model_name: str
    train_rows: int
    accuracy: float
    macro_f1: float
    weighted_f1: float


def build_downstream_models() -> dict[str, object]:
    return {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "multinomial_nb": MultinomialNB(),
        "linear_svm": LinearSVC(class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=30,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }


def train_and_score(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    labeler_name: str,
    model_name: str,
    estimator,
) -> tuple[Pipeline, ExperimentResult]:
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000)),
            ("clf", clone(estimator)),
        ]
    )
    pipeline.fit(train_df["text"], train_df["train_label"])
    pred = pipeline.predict(eval_df["text"])

    result = ExperimentResult(
        labeler_name=labeler_name,
        model_name=model_name,
        train_rows=int(len(train_df)),
        accuracy=float(accuracy_score(eval_df["label_name"], pred)),
        macro_f1=float(f1_score(eval_df["label_name"], pred, average="macro")),
        weighted_f1=float(f1_score(eval_df["label_name"], pred, average="weighted")),
    )
    return pipeline, result
