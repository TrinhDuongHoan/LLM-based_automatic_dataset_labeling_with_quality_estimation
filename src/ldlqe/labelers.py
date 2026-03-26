from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .config import KEYWORD_RULES


LABEL_ORDER = ["business", "entertainment", "health", "science_tech"]


@dataclass
class LabelerResult:
    name: str
    predictions: pd.DataFrame


def _normalize_proba(raw_scores: np.ndarray) -> np.ndarray:
    safe = np.clip(raw_scores, 1e-8, None)
    return safe / safe.sum(axis=1, keepdims=True)


class KeywordRuleLabeler:
    name = "keyword_rule"

    def fit(self, seed_df: pd.DataFrame) -> "KeywordRuleLabeler":
        self.label_order_ = LABEL_ORDER
        self.priors_ = (
            seed_df["label_name"].value_counts(normalize=True).reindex(self.label_order_).fillna(0.0).to_numpy()
        )
        return self

    def predict(self, df: pd.DataFrame) -> LabelerResult:
        probs = []
        predictions = []
        for text in df["text"].astype(str).str.lower():
            scores = np.array(self.priors_, dtype=float)
            for label_idx, label_name in enumerate(self.label_order_):
                for keyword in KEYWORD_RULES[label_name]:
                    if keyword in text:
                        scores[label_idx] += 1.0
            norm_scores = scores / scores.sum()
            probs.append(norm_scores)
            predictions.append(self.label_order_[int(np.argmax(norm_scores))])

        pred_df = pd.DataFrame(probs, columns=[f"proba_{c}" for c in self.label_order_])
        pred_df["pseudo_label"] = predictions
        pred_df["labeler_name"] = self.name
        return LabelerResult(name=self.name, predictions=pred_df)


class SeedModelLabeler:
    def __init__(self, name: str, estimator) -> None:
        self.name = name
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)),
                ("clf", clone(estimator)),
            ]
        )

    def fit(self, seed_df: pd.DataFrame) -> "SeedModelLabeler":
        self.pipeline.fit(seed_df["text"], seed_df["label_name"])
        self.label_order_ = list(self.pipeline.named_steps["clf"].classes_)
        return self

    def predict(self, df: pd.DataFrame) -> LabelerResult:
        if hasattr(self.pipeline.named_steps["clf"], "predict_proba"):
            proba = self.pipeline.predict_proba(df["text"])
        else:
            decision = self.pipeline.decision_function(df["text"])
            if decision.ndim == 1:
                decision = np.column_stack([-decision, decision])
            decision = np.exp(decision - decision.max(axis=1, keepdims=True))
            proba = _normalize_proba(decision)

        pred = self.pipeline.predict(df["text"])
        pred_df = pd.DataFrame(proba, columns=[f"proba_{c}" for c in self.label_order_])
        pred_df["pseudo_label"] = pred
        pred_df["labeler_name"] = self.name
        return LabelerResult(name=self.name, predictions=pred_df)


class RetrievalLabeler:
    name = "fewshot_retrieval"

    def fit(self, seed_df: pd.DataFrame) -> "RetrievalLabeler":
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000)
        self.seed_matrix = self.vectorizer.fit_transform(seed_df["text"])
        self.seed_labels = seed_df["label_name"].to_numpy()
        self.label_order_ = sorted(seed_df["label_name"].unique())
        return self

    def predict(self, df: pd.DataFrame) -> LabelerResult:
        target_matrix = self.vectorizer.transform(df["text"])
        sims = cosine_similarity(target_matrix, self.seed_matrix)
        top_idx = sims.argmax(axis=1)
        pred = self.seed_labels[top_idx]

        probs = np.zeros((len(df), len(self.label_order_)))
        label_to_idx = {label: idx for idx, label in enumerate(self.label_order_)}
        top_scores = sims[np.arange(len(df)), top_idx]
        for row_idx, label_name in enumerate(pred):
            probs[row_idx, label_to_idx[label_name]] = max(float(top_scores[row_idx]), 1e-6)
            probs[row_idx] += 1e-6
        probs = _normalize_proba(probs)

        pred_df = pd.DataFrame(probs, columns=[f"proba_{c}" for c in self.label_order_])
        pred_df["pseudo_label"] = pred
        pred_df["labeler_name"] = self.name
        return LabelerResult(name=self.name, predictions=pred_df)


class EnsembleLabeler:
    name = "ensemble_vote"

    def fit(self, seed_df: pd.DataFrame) -> "EnsembleLabeler":
        self.members = [
            KeywordRuleLabeler().fit(seed_df),
            RetrievalLabeler().fit(seed_df),
            SeedModelLabeler(
                name="seed_logreg_member",
                estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
            ).fit(seed_df),
            SeedModelLabeler(name="seed_nb_member", estimator=MultinomialNB()).fit(seed_df),
        ]
        self.label_order_ = LABEL_ORDER
        return self

    def predict(self, df: pd.DataFrame) -> LabelerResult:
        member_outputs = [member.predict(df).predictions for member in self.members]
        probs = np.zeros((len(df), len(self.label_order_)))
        for member_df in member_outputs:
            for label_idx, label_name in enumerate(self.label_order_):
                column = f"proba_{label_name}"
                if column in member_df:
                    probs[:, label_idx] += member_df[column].to_numpy()

        probs = _normalize_proba(probs)
        pred = [self.label_order_[idx] for idx in probs.argmax(axis=1)]
        pred_df = pd.DataFrame(probs, columns=[f"proba_{c}" for c in self.label_order_])
        pred_df["pseudo_label"] = pred
        pred_df["labeler_name"] = self.name
        return LabelerResult(name=self.name, predictions=pred_df)


def build_labelers() -> list:
    return [
        KeywordRuleLabeler(),
        RetrievalLabeler(),
        SeedModelLabeler(
            name="seed_logreg",
            estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
        ),
        SeedModelLabeler(name="seed_nb", estimator=MultinomialNB()),
        EnsembleLabeler(),
    ]
