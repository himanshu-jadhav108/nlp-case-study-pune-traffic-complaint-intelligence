"""spaCy location NER with an explicitly separate optional gazetteer."""
from __future__ import annotations

import re
from functools import lru_cache

import spacy

MODEL_NAME = "en_core_web_sm"
PUNE_LOCATIONS = ["Swargate", "Shivajinagar", "Hinjewadi", "Kothrud", "Hadapsar", "Wakad", "Baner",
                  "Koregaon Park", "Deccan", "Pimpri", "Chinchwad", "Aundh", "Katraj", "Kharadi",
                  "Viman Nagar", "Yerawada", "Camp", "Pune Station", "FC Road", "JM Road", "Sinhagad Road"]


@lru_cache(maxsize=1)
def load_nlp():
    """Load the installed small English spaCy model; raises a clear error if absent."""
    try:
        return spacy.load(MODEL_NAME)
    except (OSError, ImportError) as exc:
        raise RuntimeError(f"spaCy model {MODEL_NAME} is unavailable. Install dependencies from requirements.txt.") from exc


def extract_locations(text: str, nlp=None, include_gazetteer: bool = True):
    """Return spaCy entities and gazetteer matches in separate arrays."""
    if text is None or not str(text).strip():
        return {"spacy": [], "gazetteer": []}
    try:
        pipeline = nlp or load_nlp()
        doc = pipeline(str(text))
        entities = [{"text": ent.text, "label": normalize_label(ent.label_), "start": ent.start_char,
                     "end": ent.end_char, "method": "spaCy NER"}
                    for ent in doc.ents if normalize_label(ent.label_) == "LOC"]
    except RuntimeError:
        # A visible gazetteer fallback preserves usability, but is never described as NER.
        entities = []
    gazetteer = []
    if include_gazetteer:
        for place in PUNE_LOCATIONS:
            for match in re.finditer(rf"(?<!\w){re.escape(place)}(?!\w)", str(text), flags=re.IGNORECASE):
                gazetteer.append({"text": match.group(), "label": "LOC", "start": match.start(), "end": match.end(),
                                  "method": "Pune gazetteer fallback"})
    return {"spacy": sorted(entities, key=lambda e: e["start"]), "gazetteer": sorted(gazetteer, key=lambda e: e["start"])}


def normalize_label(label: str) -> str:
    return "LOC" if label.upper() in {"GPE", "LOC", "FAC"} else label.upper()


def highlight_entities(text: str, spans):
    """Return safe HTML with extracted spans highlighted; entity text is escaped."""
    from html import escape
    ordered = sorted(spans, key=lambda item: item["start"])
    out, cursor = [], 0
    for span in ordered:
        start, end = int(span["start"]), int(span["end"])
        if start < cursor or end > len(text) or start >= end:
            continue
        out.append(escape(text[cursor:start]))
        out.append(f'<mark title="{escape(span.get("label", "LOC"))}">{escape(text[start:end])}</mark>')
        cursor = end
    out.append(escape(text[cursor:]))
    return "".join(out)
