import pandas as pd
import pytest

from src.classifier import build_models, evaluate_models, predict_complaints
from src.data_utils import LABELS, DatasetError, load_sample_data, validate_complaints


def test_all_six_labels_present_and_predictions():
    data = load_sample_data()
    assert set(data.category) == set(LABELS)
    result = evaluate_models(data)
    assert result.selected_name in {"Logistic Regression", "Multinomial Naive Bayes"}
    assert len(result.y_test) == len(result.y_pred) > 0
    assert result.confusion.shape == (6, 6)
    assert all(k in result.report for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted"))


def test_csv_validation_and_blank_handling():
    with pytest.raises(DatasetError, match="Missing"):
        validate_complaints(pd.DataFrame({"category": ["Signal"]}))
    with pytest.raises(DatasetError, match="Unsupported"):
        validate_complaints(pd.DataFrame({"complaint_text": ["road"], "category": ["Other"]}))
    with pytest.raises(DatasetError, match="non-blank"):
        validate_complaints(pd.DataFrame({"complaint_text": [" "], "category": ["Signal"]}))


def test_prediction_rejects_empty_input():
    model = build_models()["Logistic Regression"]
    with pytest.raises(ValueError, match="blank"):
        predict_complaints(model, [""])


def test_prediction_with_proba():
    from src.classifier import predict_complaints_with_proba
    data = load_sample_data()
    eval_res = evaluate_models(data)
    preds, probas = predict_complaints_with_proba(eval_res.selected_model, ["Heavy jam at Swargate junction"])
    assert len(preds) == 1
    assert preds[0] in LABELS
    assert len(probas) == 1
    assert set(probas[0].keys()) == set(LABELS)
    assert abs(sum(probas[0].values()) - 1.0) < 1e-4
