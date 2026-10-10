"""End-to-end integration tests for core application pipelines."""
import pandas as pd
import pytest

from src.classifier import evaluate_models, predict_complaints, predict_complaints_with_proba
from src.data_utils import LABELS, load_sample_data, validate_complaints
from src.location_ner import extract_locations, highlight_entities
from src.ner_evaluation import evaluate_ner, load_ner_data


def test_end_to_end_pipeline():
    # 1. Load data
    data = load_sample_data()
    assert len(data) >= 100
    assert set(data["category"].unique()) == set(LABELS)

    # 2. Fit models and evaluate
    evaluation = evaluate_models(data)
    assert evaluation.selected_name in {"Logistic Regression", "Multinomial Naive Bayes"}
    assert len(evaluation.y_test) == len(evaluation.y_pred) > 0
    assert evaluation.confusion.shape == (6, 6)
    assert "accuracy" in evaluation.report

    # 3. Test single complaint prediction & confidence
    sample_text = "Massive traffic jam near Swargate because road construction has blocked two lanes."
    preds, probas = predict_complaints_with_proba(evaluation.selected_model, [sample_text])
    assert len(preds) == 1
    assert preds[0] in LABELS
    assert len(probas) == 1
    assert set(probas[0].keys()) == set(LABELS)

    # 4. Location extraction & HTML highlighting
    loc_result = extract_locations(sample_text)
    assert isinstance(loc_result["spacy"], list)
    assert isinstance(loc_result["gazetteer"], list)
    assert any(g["text"] == "Swargate" for g in loc_result["gazetteer"])
    
    html = highlight_entities(sample_text, loc_result["gazetteer"])
    assert "<mark" in html
    assert "Swargate" in html


def test_batch_prediction_flow():
    data = load_sample_data()
    evaluation = evaluate_models(data)
    
    batch_input = pd.DataFrame({
        "complaint_text": [
            "Potholes on Baner road causing slow traffic",
            "Signal is dead at University Circle",
            "Illegal parking on FC Road",
        ]
    })
    validated_batch = validate_complaints(batch_input, require_category=False)
    preds, probas = predict_complaints_with_proba(evaluation.selected_model, validated_batch["complaint_text"])
    assert len(preds) == 3
    assert all(p in LABELS for p in preds)
    assert len(probas) == 3
