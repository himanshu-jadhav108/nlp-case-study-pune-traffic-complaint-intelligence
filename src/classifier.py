"""TF-IDF complaint classification and held-out evaluation."""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             classification_report)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .data_utils import LABELS, split_data, validate_complaints


def build_models(seed: int = 42):
    return {
        "Logistic Regression": Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
                                          ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))]),
        "Multinomial Naive Bayes": Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
                                              ("classifier", MultinomialNB(alpha=0.5))]),
    }


def _scores(y_true, y_pred):
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision_macro": float(p_macro), "recall_macro": float(r_macro),
            "f1_macro": float(f_macro), "precision_weighted": float(p_weighted), "recall_weighted": float(r_weighted),
            "f1_weighted": float(f_weighted)}


@dataclass
class Evaluation:
    models: dict
    comparison: list[dict]
    selected_name: str
    selected_model: Pipeline
    y_test: list[str]
    y_pred: list[str]
    labels: list[str]
    confusion: np.ndarray
    report: dict


def evaluate_models(frame, seed: int = 42) -> Evaluation:
    frame = validate_complaints(frame)
    train, val, test = split_data(frame, seed)
    if train.category.nunique() < 2:
        raise ValueError("Training split contains fewer than two categories; add more examples per category.")
    models = build_models(seed)
    comparison = []
    for name, model in models.items():
        model.fit(train.complaint_text, train.category)
        row = {"model": name, **_scores(val.category, model.predict(val.complaint_text))}
        comparison.append(row)
    # Validation macro-F1 is the selection rule; deterministic tie-break favors LR.
    selected_name = max(comparison, key=lambda row: (row["f1_macro"], row["accuracy"], row["model"] == "Logistic Regression"))["model"]
    selected_model = models[selected_name]
    # Refit the selected pipeline on train+validation only after selection.
    fit_data = __import__("pandas").concat([train, val])
    selected_model.fit(fit_data.complaint_text, fit_data.category)
    prediction = selected_model.predict(test.complaint_text)
    report = _scores(test.category, prediction)
    report["per_class"] = classification_report(test.category, prediction, labels=LABELS, output_dict=True, zero_division=0)
    return Evaluation(models, comparison, selected_name, selected_model, test.category.tolist(), prediction.tolist(), LABELS,
                      confusion_matrix(test.category, prediction, labels=LABELS), report)


def predict_complaints(model, texts):
    normalized = ["" if text is None else str(text).strip() for text in texts]
    if not normalized or any(not text for text in normalized):
        raise ValueError("Complaint text cannot be blank.")
    return model.predict(normalized)


def predict_complaints_with_proba(model, texts):
    """Return predictions and a dictionary mapping each class to probability per sample."""
    normalized = ["" if text is None else str(text).strip() for text in texts]
    if not normalized or any(not text for text in normalized):
        raise ValueError("Complaint text cannot be blank.")
    preds = model.predict(normalized)
    probas = []
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        raw_probs = model.predict_proba(normalized)
        for row in raw_probs:
            probas.append({cls: float(p) for cls, p in zip(classes, row)})
    else:
        for pred in preds:
            probas.append({pred: 1.0})
    return preds, probas
