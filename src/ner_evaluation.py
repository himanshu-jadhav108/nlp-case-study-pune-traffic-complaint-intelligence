"""Exact-span entity-level NER evaluation."""
from __future__ import annotations

import json
from pathlib import Path

NER_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "ner_annotated_test.jsonl"


def load_ner_data(path=NER_DATA_PATH):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line_num, line in enumerate(stream, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row.get("text"), str) or not isinstance(row.get("entities"), list):
                    raise ValueError(f"Invalid NER annotation on line {line_num}.")
                rows.append(row)
    return rows


def exact_span_metrics(gold_records, predicted_records):
    """Micro precision/recall/F1 over (start, end, normalized label) tuples."""
    if len(gold_records) != len(predicted_records):
        raise ValueError("Gold and prediction record counts must match.")
    tp = fp = fn = 0
    for gold, predicted in zip(gold_records, predicted_records):
        gold_set = {(int(e["start"]), int(e["end"]), normalize_label(e.get("label", "LOC"))) for e in gold}
        pred_set = {(int(e["start"]), int(e["end"]), normalize_label(e.get("label", "LOC"))) for e in predicted}
        tp += len(gold_set & pred_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn, "precision": precision, "recall": recall, "f1": f1}


def normalize_label(label):
    return "LOC" if str(label).upper() in {"LOC", "GPE", "FAC"} else str(label).upper()


def evaluate_ner(rows, nlp):
    gold, predicted = [], []
    for row in rows:
        gold.append(row["entities"])
        doc = nlp(row["text"])
        predicted.append([{"start": e.start_char, "end": e.end_char, "label": normalize_label(e.label_)}
                          for e in doc.ents if normalize_label(e.label_) == "LOC"])
    return exact_span_metrics(gold, predicted), predicted
