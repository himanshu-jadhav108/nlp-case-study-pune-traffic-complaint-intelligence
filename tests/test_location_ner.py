from types import SimpleNamespace

from src.location_ner import extract_locations, highlight_entities, normalize_label


class DummyNLP:
    def __call__(self, text):
        start = text.index("Swargate")
        return SimpleNamespace(ents=[SimpleNamespace(text="Swargate", label_="GPE", start_char=start, end_char=start+8)])


def test_output_format_and_separate_gazetteer():
    result = extract_locations("Traffic near Swargate", nlp=DummyNLP())
    assert result["spacy"][0]["text"] == "Swargate"
    assert result["spacy"][0]["label"] == "LOC"
    assert result["spacy"][0]["method"] == "spaCy NER"
    assert result["gazetteer"][0]["method"] == "Pune gazetteer fallback"
    assert normalize_label("FAC") == "LOC"
    assert extract_locations("  ", nlp=DummyNLP()) == {"spacy": [], "gazetteer": []}


def test_highlighting_escapes_markup():
    text = "Near Swargate <script>"
    spans = [{"start": 5, "end": 13, "label": "LOC"}]
    html = highlight_entities(text, spans)
    assert "<mark" in html and "&lt;script&gt;" in html


def test_missing_spacy_model_is_clear(monkeypatch):
    from src import location_ner
    monkeypatch.setattr(location_ner, "load_nlp", lambda: (_ for _ in ()).throw(RuntimeError("model missing")))
    result = location_ner.extract_locations("Traffic near Swargate")
    assert result["spacy"] == []
    assert result["gazetteer"][0]["text"] == "Swargate"
