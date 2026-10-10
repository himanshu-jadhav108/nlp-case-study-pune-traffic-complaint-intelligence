import pytest

from src.ner_evaluation import exact_span_metrics


def test_exact_span_metrics_hand_checked():
    gold = [[{"start": 3, "end": 11, "label": "LOC"}, {"start": 20, "end": 25, "label": "LOC"}]]
    predicted = [[{"start": 3, "end": 11, "label": "GPE"}, {"start": 22, "end": 25, "label": "LOC"}]]
    metrics = exact_span_metrics(gold, predicted)
    assert (metrics["true_positive"], metrics["false_positive"], metrics["false_negative"]) == (1, 1, 1)
    assert metrics["precision"] == 0.5 and metrics["recall"] == 0.5 and metrics["f1"] == 0.5


def test_record_count_mismatch_fails():
    with pytest.raises(ValueError, match="counts"):
        exact_span_metrics([[]], [])
