from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def train_tfidf(train_text, train_y):
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)),
        ("lr", LogisticRegression(max_iter=3000, solver="liblinear", multi_class="ovr")),
    ])
    clf.fit(train_text, train_y)
    return clf