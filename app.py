"""Pune Traffic Complaint Intelligence.

A production-ready Streamlit application for traffic complaint categorization
and multi-method location entity extraction (spaCy NER + Pune gazetteer fallback).
"""
from __future__ import annotations

import io
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.classifier import evaluate_models, predict_complaints, predict_complaints_with_proba
from src.data_utils import DATA_PATH, LABELS, DatasetError, load_sample_data, validate_complaints
from src.location_ner import extract_locations, highlight_entities, load_nlp
from src.ner_evaluation import evaluate_ner, load_ner_data

# Category visual tokens (icons, descriptions, and color accents)
CATEGORY_META = {
    "Congestion": {
        "icon": "🚦",
        "color": "#D97706",
        "bg": "rgba(217, 119, 6, 0.12)",
        "desc": "Slow, queued, or gridlocked traffic conditions.",
    },
    "Accident": {
        "icon": "⚠️",
        "color": "#DC2626",
        "bg": "rgba(220, 38, 38, 0.12)",
        "desc": "Vehicle collision or roadside emergency incident.",
    },
    "Signal": {
        "icon": "🛑",
        "color": "#2563EB",
        "bg": "rgba(37, 99, 235, 0.12)",
        "desc": "Faulty, dark, or mis-timed traffic signal lights.",
    },
    "Parking": {
        "icon": "🚗",
        "color": "#7C3AED",
        "bg": "rgba(124, 58, 237, 0.12)",
        "desc": "Illegal, obstructive, or double-parking violations.",
    },
    "Construction": {
        "icon": "🚧",
        "color": "#EA580C",
        "bg": "rgba(234, 88, 12, 0.12)",
        "desc": "Roadworks, metro construction, or utility trenches.",
    },
    "Road Condition": {
        "icon": "🛣️",
        "color": "#0D9488",
        "bg": "rgba(13, 148, 136, 0.12)",
        "desc": "Potholes, broken tarmac, waterlogging, or debris.",
    },
}

SAMPLE_PRESETS = [
    {
        "label": "Swargate Bottleneck",
        "text": "Massive traffic jam near Swargate bus depot because road construction has blocked two major lanes during peak evening hours.",
    },
    {
        "label": "JM Road Signal Fault",
        "text": "The traffic signal lights are completely dark and non-functional at JM Road junction, causing total chaos and near collisions.",
    },
    {
        "label": "Baner Road Potholes",
        "text": "Severe potholes and damaged road surface on Baner Road near the highway turnoff are creating dangerous driving conditions.",
    },
    {
        "label": "FC Road Illegal Parking",
        "text": "Multiple cars and delivery vans double parked illegally outside shops on FC Road, restricting traffic to a single lane.",
    },
    {
        "label": "Hinjewadi Vehicle Crash",
        "text": "A serious two-vehicle collision on Hinjewadi Phase 1 main road has halted all commute traffic towards the IT park.",
    },
]

# Page configuration
st.set_page_config(
    page_title="Pune Traffic Complaint Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Scoped CSS for Clean Professional Design
st.markdown(
    """
    <style>
    /* Professional Typography and Spacing Tokens */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1300px;
    }
    
    /* Entity Highlighting Chip Styling */
    mark {
        background-color: rgba(37, 99, 235, 0.15) !important;
        color: #1d4ed8 !important;
        border: 1px solid rgba(37, 99, 235, 0.35) !important;
        border-radius: 4px !important;
        padding: 2px 7px !important;
        font-weight: 600 !important;
        font-size: 0.95em !important;
        box-decoration-break: clone;
        -webkit-box-decoration-break: clone;
    }
    @media (prefers-color-scheme: dark) {
        mark {
            background-color: rgba(59, 130, 246, 0.25) !important;
            color: #93c5fd !important;
            border: 1px solid rgba(59, 130, 246, 0.5) !important;
        }
    }
    
    /* Header & Badge Styling */
    .civic-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        background: rgba(37, 99, 235, 0.08);
        color: #2563eb;
        border: 1px solid rgba(37, 99, 235, 0.2);
    }
    
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .status-active {
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    
    .status-warning {
        background: #fffbeb;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    
    /* Category Hero Badge */
    .category-hero {
        padding: 12px 18px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
    }
    
    /* Subtle metric text adjustments */
    [data-testid="stMetricValue"] {
        font-size: 1.65rem !important;
        font-weight: 700 !important;
    }
    
    /* Streamlit tabs font size */
    button[data-baseweb="tab"] {
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_sample():
    return load_sample_data()


@st.cache_resource
def get_evaluation():
    return evaluate_models(get_sample())


@st.cache_resource
def get_spacy():
    return load_nlp()


# Load project resources safely
try:
    data = get_sample()
    evaluation = get_evaluation()
except Exception as exc:
    st.error(f"The bundled project dataset could not be loaded: {exc}")
    st.stop()

# Detect spaCy availability safely
spacy_loaded = True
spacy_meta_info = "en_core_web_sm"
try:
    nlp_model = get_spacy()
    spacy_meta_info = f"{nlp_model.meta.get('name', 'spaCy model')} ({nlp_model.meta.get('version', '3.8.0')})"
except Exception:
    spacy_loaded = False
    spacy_meta_info = "Unavailable (Using Pune Gazetteer fallback)"

# ==============================================================================
# SIDEBAR: SYSTEM DASHBOARD & METADATA
# ==============================================================================
with st.sidebar:
    st.markdown("### 🚦 Pune Traffic AI")
    st.caption("Civic NLP & Intelligent Dispatch Assistant")

    st.markdown(
        """
        <div style="margin-bottom: 12px;">
            <span class="civic-badge">📍 Pune Smart Mobility</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("**System Health & Runtime**")
        if spacy_loaded:
            st.markdown(
                '<span class="status-pill status-active">● spaCy NER: Active</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-pill status-warning">▲ spaCy NER: Offline</span>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<span class="status-pill status-active">● Classifier: {evaluation.selected_name}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span class="status-pill status-active">● Records: {len(data)} synthetic</span>',
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.markdown("**Architecture & Models**")
        st.markdown(f"• **Classifier**: {evaluation.selected_name}")
        st.markdown("• **Features**: Word & Bigram TF-IDF (Sublinear)")
        st.markdown(f"• **NER Engine**: {spacy_meta_info}")
        st.markdown("• **Fallback**: Curated Pune Arterial Gazetteer (21 locations)")
        st.markdown("• **Validation**: Balanced stratified split (Seed 42)")

    with st.container(border=True):
        st.markdown("**Academic & Synthetic Notice**")
        st.caption(
            "This application uses bundled synthetic complaints for demonstration. "
            "Data is generated via phrasing templates to model real-world leakage-aware validation. "
            "It does not store user data or require external cloud APIs."
        )

# ==============================================================================
# MAIN APPLICATION HEADER
# ==============================================================================
col_head1, col_head2 = st.columns([0.72, 0.28])
with col_head1:
    st.title("Pune Traffic Complaint Intelligence")
    st.markdown(
        "Automated multi-class classification and hybrid location extraction for civic traffic grievances across Pune metropolitan corridors."
    )
with col_head2:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.caption("Active Champion Pipeline")
        st.markdown(f"**{evaluation.selected_name}**")
        macro_f1 = evaluation.report.get("f1_macro", 0.0)
        st.caption(f"Test Macro-F1: `{macro_f1:.3f}` | Classes: `6`")

# Application Tabs
tabs = st.tabs([
    "🔍 Analyze Complaint",
    "📁 Batch Processing",
    "📊 Classification Evaluation",
    "🎯 Location NER Evaluation",
    "📖 Dataset & Methodology",
])

# ==============================================================================
# TAB 1: ANALYZE COMPLAINT
# ==============================================================================
with tabs[0]:
    st.subheader("Interactive Complaint Diagnostic")
    st.caption("Input or select an example civic traffic grievance to predict its primary category and extract road landmarks.")

    st.markdown("**Quick Preset Examples:**")
    preset_cols = st.columns(len(SAMPLE_PRESETS))
    for idx, preset in enumerate(SAMPLE_PRESETS):
        if preset_cols[idx].button(f"📌 {preset['label']}", use_container_width=True, key=f"pre_{idx}"):
            st.session_state["input_complaint"] = preset["text"]

    default_text = st.session_state.get(
        "input_complaint",
        "Heavy traffic congestion near Swargate junction because continuous metro road construction has blocked two lanes during peak rush hour.",
    )

    complaint_input = st.text_area(
        "Traffic Complaint Narrative",
        value=default_text,
        placeholder="Type or paste a citizen traffic complaint narrative here...",
        height=110,
        help="Enter realistic Pune traffic descriptions mentioning roads, signals, or bottlenecks.",
    )

    analyze_clicked = st.button("🚀 Analyze Complaint", type="primary", use_container_width=False)

    if analyze_clicked or "auto_analyzed" not in st.session_state:
        st.session_state["auto_analyzed"] = True
        cleaned = complaint_input.strip()

        if not cleaned:
            st.warning("Please enter or select a non-empty complaint narrative before analyzing.")
        else:
            with st.spinner("Running NLP feature extraction and pipeline inference..."):
                preds, probas_list = predict_complaints_with_proba(evaluation.selected_model, [cleaned])
                predicted_cat = preds[0]
                category_probs = probas_list[0] if probas_list else {predicted_cat: 1.0}
                loc_result = extract_locations(cleaned)
                cat_meta = CATEGORY_META.get(predicted_cat, {"icon": "🚦", "color": "#2563EB", "bg": "#EFF6FF", "desc": ""})
                top_conf = category_probs.get(predicted_cat, 1.0) * 100.0

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            # Results Section
            res_col1, res_col2 = st.columns([0.48, 0.52])

            with res_col1:
                with st.container(border=True):
                    st.markdown("**Category Classification Result**")
                    st.markdown(
                        f"""
                        <div class="category-hero" style="background-color: {cat_meta['bg']}; border: 1.5px solid {cat_meta['color']};">
                            <div>
                                <span style="font-size: 1.7rem; margin-right: 8px;">{cat_meta['icon']}</span>
                                <span style="font-size: 1.4rem; font-weight: 700; color: {cat_meta['color']};">{predicted_cat}</span>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 1.35rem; font-weight: 700; color: {cat_meta['color']};">{top_conf:.1f}%</div>
                                <div style="font-size: 0.75rem; color: #475569;">Confidence</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.caption(f"**Operational Scope:** {cat_meta['desc']}")

                    # Class Probabilities Breakdown Chart
                    st.markdown("**Model Class Probability Distribution:**")
                    prob_df = pd.DataFrame([
                        {"Category": k, "Probability": v * 100.0, "Color": CATEGORY_META.get(k, {}).get("color", "#2563EB")}
                        for k, v in category_probs.items()
                    ]).sort_values("Probability", ascending=True)

                    fig_prob = px.bar(
                        prob_df,
                        x="Probability",
                        y="Category",
                        orientation="h",
                        text=prob_df["Probability"].apply(lambda p: f"{p:.1f}%"),
                        color="Category",
                        color_discrete_map={k: v["color"] for k, v in CATEGORY_META.items()},
                    )
                    fig_prob.update_layout(
                        showlegend=False,
                        height=210,
                        margin=dict(l=0, r=20, t=10, b=20),
                        xaxis=dict(title="Probability (%)", range=[0, 105]),
                        yaxis=dict(title=""),
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)

            with res_col2:
                with st.container(border=True):
                    st.markdown("**Extracted Location Entities (Corridor & Landmark)**")

                    # Highlighted Narrative
                    all_spans = loc_result["spacy"] if loc_result["spacy"] else loc_result["gazetteer"]
                    if all_spans:
                        st.markdown("**Narrative Span Highlighting:**")
                        highlighted_html = highlight_entities(cleaned, all_spans)
                        st.markdown(
                            f"""
                            <div style="padding: 10px 14px; background: rgba(0,0,0,0.02); border: 1px solid rgba(0,0,0,0.08); border-radius: 8px; line-height: 1.6; margin-bottom: 12px;">
                                {highlighted_html}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Entities Details Subtabs
                    ent_tab1, ent_tab2 = st.tabs(["🤖 spaCy NER Detections", "📚 Pune Gazetteer Fallback"])

                    with ent_tab1:
                        if loc_result["spacy"]:
                            spacy_df = pd.DataFrame(loc_result["spacy"])[["text", "label", "start", "end", "method"]]
                            st.dataframe(spacy_df, hide_index=True, use_container_width=True)
                        else:
                            st.info("No spatial location entities identified by standard spaCy NER in this specific phrasing.")

                    with ent_tab2:
                        if loc_result["gazetteer"]:
                            gaz_df = pd.DataFrame(loc_result["gazetteer"])[["text", "label", "start", "end", "method"]]
                            st.dataframe(gaz_df, hide_index=True, use_container_width=True)
                            st.caption("Pune Gazetteer matches known arterial hubs independent of grammatical parsing.")
                        else:
                            st.info("No exact Pune gazetteer landmarks matched in the input text.")

                    if spacy_loaded:
                        st.caption(f"Engine: {spacy_meta_info} | Normalization: `GPE/LOC/FAC` → `LOC`")
                    else:
                        st.warning("spaCy model is not loaded in this session. Gazetteer fallback operates independently.")

# ==============================================================================
# TAB 2: BATCH PROCESSING
# ==============================================================================
with tabs[1]:
    st.subheader("Batch Complaint Classification & Location Extraction")
    st.caption("Upload a structured CSV file containing a `complaint_text` column to perform batch dispatch categorization and geospatial landmark extraction.")

    upload_col1, upload_col2 = st.columns([0.65, 0.35])

    with upload_col1:
        uploaded_file = st.file_uploader(
            "Upload Complaints CSV File",
            type=["csv"],
            key="batch_uploader",
            help="CSV must contain a 'complaint_text' column. 'category' column is optional.",
        )

    with upload_col2:
        with st.container(border=True):
            st.markdown("**Sample Template CSV**")
            st.caption("Download an example CSV containing formatted test records for batch evaluation.")
            sample_template = data[["complaint_text", "category"]].head(5).copy()
            st.download_button(
                "📥 Download Sample CSV",
                sample_template.to_csv(index=False).encode("utf-8-sig"),
                "sample_traffic_complaints.csv",
                "text/csv",
                use_container_width=True,
            )

    if uploaded_file:
        try:
            raw_input_df = pd.read_csv(uploaded_file)
            batch_df = validate_complaints(raw_input_df, require_category=False)

            with st.spinner(f"Classifying {len(batch_df)} batch records and extracting locations..."):
                preds, probas_list = predict_complaints_with_proba(evaluation.selected_model, batch_df["complaint_text"])
                extracted_all = [extract_locations(t) for t in batch_df["complaint_text"]]

                batch_df["predicted_category"] = preds
                batch_df["confidence_score"] = [
                    round(max(p.values()) * 100.0, 1) if p else 100.0 for p in probas_list
                ]
                batch_df["spacy_locations"] = [
                    "; ".join(f"{e['text']} ({e['label']})" for e in x["spacy"]) if x["spacy"] else "None"
                    for x in extracted_all
                ]
                batch_df["gazetteer_landmarks"] = [
                    "; ".join(e["text"] for e in x["gazetteer"]) if x["gazetteer"] else "None"
                    for x in extracted_all
                ]

            st.success(f"Successfully processed {len(batch_df)} complaint records!")

            # Batch Summary Metrics
            b_metric1, b_metric2, b_metric3, b_metric4 = st.columns(4)
            b_metric1.metric("Total Records", len(batch_df))
            b_metric2.metric("Dominant Category", batch_df["predicted_category"].mode()[0])
            avg_conf = batch_df["confidence_score"].mean()
            b_metric3.metric("Avg Confidence", f"{avg_conf:.1f}%")
            has_locs = (batch_df["spacy_locations"] != "None") | (batch_df["gazetteer_landmarks"] != "None")
            b_metric4.metric("Location Extracted", f"{has_locs.sum()}/{len(batch_df)} ({has_locs.mean()*100:.0f}%)")

            # Batch Visual Analytics & Filter
            v_col1, v_col2 = st.columns([0.6, 0.4])
            with v_col1:
                batch_cat_counts = batch_df["predicted_category"].value_counts().reindex(LABELS, fill_value=0).reset_index()
                batch_cat_counts.columns = ["Category", "Count"]
                fig_batch = px.bar(
                    batch_cat_counts,
                    x="Category",
                    y="Count",
                    color="Category",
                    color_discrete_map={k: v["color"] for k, v in CATEGORY_META.items()},
                    text="Count",
                    title="Batch Predicted Category Distribution",
                )
                fig_batch.update_layout(showlegend=False, height=260, margin=dict(l=0, r=0, t=30, b=10))
                st.plotly_chart(fig_batch, use_container_width=True)

            with v_col2:
                st.markdown("**Filter Batch Results**")
                selected_cat_filter = st.selectbox(
                    "Filter by Predicted Category",
                    ["All Categories"] + LABELS,
                )
                filtered_batch = batch_df if selected_cat_filter == "All Categories" else batch_df[batch_df["predicted_category"] == selected_cat_filter]
                st.caption(f"Displaying {len(filtered_batch)} of {len(batch_df)} records")

            st.dataframe(filtered_batch, hide_index=True, use_container_width=True)

            csv_data = batch_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 Download Complete Predictions CSV",
                csv_data,
                "pune_traffic_predictions.csv",
                "text/csv",
                type="primary",
            )

        except DatasetError as exc:
            st.error(f"Input Validation Error: {exc}")
        except Exception as exc:
            st.error(f"Failed to process CSV file: {exc}")

# ==============================================================================
# TAB 3: CLASSIFICATION EVALUATION
# ==============================================================================
with tabs[2]:
    st.subheader("Model Validation & Held-out Performance Benchmark")
    st.caption(
        "Evaluation results computed strictly on the held-out test split using balanced stratified partitioning (Seed 42) "
        "ensuring all 6 traffic grievance categories are represented."
    )

    with st.container(border=True):
        st.markdown(f"🏆 **Selected Champion Pipeline:** `{evaluation.selected_name}`")
        st.caption(
            "Selection Strategy: Validation macro-F1 is evaluated across competing TF-IDF pipelines; "
            "validation accuracy serves as a deterministic tie-breaker. The selected model is then refit on train + validation."
        )

    # Model comparison table
    st.markdown("**Pipeline Comparison on Validation Split:**")
    comp_df = pd.DataFrame(evaluation.comparison)
    st.dataframe(
        comp_df.style.format({
            "accuracy": "{:.3f}",
            "precision_macro": "{:.3f}",
            "recall_macro": "{:.3f}",
            "f1_macro": "{:.3f}",
            "precision_weighted": "{:.3f}",
            "recall_weighted": "{:.3f}",
            "f1_weighted": "{:.3f}",
        }),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### Held-Out Test Partition Benchmark")

    score_dict = evaluation.report
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Accuracy", f"{score_dict.get('accuracy', 0.0):.3f}")
    m2.metric("Macro Precision", f"{score_dict.get('precision_macro', 0.0):.3f}")
    m3.metric("Macro Recall", f"{score_dict.get('recall_macro', 0.0):.3f}")
    m4.metric("Macro F1-Score", f"{score_dict.get('f1_macro', 0.0):.3f}")

    w1, w2, w3 = st.columns(3)
    w1.metric("Weighted Precision", f"{score_dict.get('precision_weighted', 0.0):.3f}")
    w2.metric("Weighted Recall", f"{score_dict.get('recall_weighted', 0.0):.3f}")
    w3.metric("Weighted F1-Score", f"{score_dict.get('f1_weighted', 0.0):.3f}")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    cm_col, class_col = st.columns([0.52, 0.48])

    with cm_col:
        st.markdown("**Six-Class Test Confusion Matrix**")
        cm_matrix = pd.DataFrame(evaluation.confusion, index=LABELS, columns=LABELS)
        fig_cm = px.imshow(
            cm_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Cases"),
        )
        fig_cm.update_layout(
            xaxis_title="Predicted Category",
            yaxis_title="True Category",
            height=370,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with class_col:
        st.markdown("**Per-Class Precision, Recall & F1**")
        per_class_df = pd.DataFrame(score_dict.get("per_class", {})).T.reindex(LABELS)
        display_per_class = per_class_df[["precision", "recall", "f1-score", "support"]].copy()
        display_per_class["support"] = display_per_class["support"].astype(int)
        st.dataframe(
            display_per_class.style.format({
                "precision": "{:.3f}",
                "recall": "{:.3f}",
                "f1-score": "{:.3f}",
                "support": "{:d}",
            }),
            use_container_width=True,
            height=370,
        )

# ==============================================================================
# TAB 4: NER EVALUATION
# ==============================================================================
with tabs[3]:
    st.subheader("Exact-Span Named Entity Recognition Evaluation")
    st.caption(
        "Evaluation uses held-out hand-annotated test records (ner_annotated_test.jsonl). "
        "A prediction is scored as a True Positive strictly when character start offset, end offset, "
        "and normalized label ('LOC') match gold annotation exactly."
    )

    try:
        nlp_runner = get_spacy()
        ner_test_records = load_ner_data()
        ner_metrics, ner_predictions = evaluate_ner(ner_test_records, nlp_runner)

        n_col1, n_col2, n_col3 = st.columns(3)
        n_col1.metric("Exact-Span Precision", f"{ner_metrics['precision']:.3f}", help="TP / (TP + FP)")
        n_col2.metric("Exact-Span Recall", f"{ner_metrics['recall']:.3f}", help="TP / (TP + FN)")
        n_col3.metric("Exact-Span F1-Score", f"{ner_metrics['f1']:.3f}", help="Harmonic mean of precision and recall")

        with st.container(border=True):
            st.markdown("**Entity Span Confusion Breakdown**")
            kpi_tp, kpi_fp, kpi_fn, kpi_total = st.columns(4)
            kpi_tp.metric("True Positives (Exact Match)", ner_metrics["true_positive"])
            kpi_fp.metric("False Positives (Spurious Span)", ner_metrics["false_positive"])
            kpi_fn.metric("False Negatives (Missed Span)", ner_metrics["false_negative"])
            kpi_total.metric("Annotated Test Records", len(ner_test_records))

        with st.expander("🔍 Inspect All Hand-Annotated Gold Records & Model Predictions", expanded=True):
            rows_detail = []
            for i, gold_item in enumerate(ner_test_records):
                text_content = gold_item["text"]
                gold_spans_str = ", ".join(text_content[e["start"]:e["end"]] for e in gold_item["entities"]) or "None"
                pred_spans_str = ", ".join(text_content[e["start"]:e["end"]] for e in ner_predictions[i]) or "None"
                rows_detail.append({
                    "Index": i + 1,
                    "Complaint Narrative": text_content,
                    "Gold Ground Truth Locations": gold_spans_str,
                    "spaCy NER Predictions": pred_spans_str,
                })
            st.dataframe(pd.DataFrame(rows_detail), hide_index=True, use_container_width=True)

    except RuntimeError as exc:
        st.error(f"NER Engine Error: {exc}")
        st.info("To run full NER evaluation, ensure dependencies from requirements.txt are installed.")

# ==============================================================================
# TAB 5: DATASET & METHODOLOGY
# ==============================================================================
with tabs[4]:
    st.subheader("Dataset Provenance, Taxonomy & Engineering Methodology")

    with st.container(border=True):
        st.warning(
            "⚠️ **Academic Demonstration Notice:** The bundled dataset consists of 180 synthetic complaint records "
            "designed to illustrate NLP categorization, spatial entity extraction, and leakage-controlled splitting. "
            "These are not official Pune Municipal Corporation (PMC) records."
        )

    st.markdown("### 6-Class Traffic Grievance Taxonomy")
    cat_grid = st.columns(3)
    for idx, (cat_name, meta) in enumerate(CATEGORY_META.items()):
        col_target = cat_grid[idx % 3]
        with col_target:
            with st.container(border=True):
                st.markdown(f"**{meta['icon']} {cat_name}**")
                st.caption(meta["desc"])

    st.markdown("---")
    st.markdown("### Class Distribution & Dataset Exploration")

    d_col1, d_col2 = st.columns([0.45, 0.55])
    with d_col1:
        st.markdown(f"**Class Counts (Total: {len(data)} records)**")
        dist_df = data["category"].value_counts().reindex(LABELS, fill_value=0).reset_index()
        dist_df.columns = ["Category", "Count"]
        fig_dist = px.bar(
            dist_df,
            x="Category",
            y="Count",
            color="Category",
            color_discrete_map={k: v["color"] for k, v in CATEGORY_META.items()},
            text="Count",
        )
        fig_dist.update_layout(showlegend=False, height=270, margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig_dist, use_container_width=True)

    with d_col2:
        st.markdown("**Data Partitioning & Evaluation Architecture**")
        st.markdown(
            """
            • **Balanced Stratification**: Splits preserve exact class distributions across Train (60%), Validation (20%), and Test (20%).
            • **Full Class Coverage**: Ensures all 6 grievance categories are evaluated in validation model selection and test scoring.
            • **Sublinear TF-IDF**: Word and bigram term frequency damped with logarithmic scaling (`sublinear_tf=True`) to diminish the dominance of repetitive words.
            • **Independent Evaluation**: Pipeline vectorizer is fitted only on training data, then selected model is evaluated once on held-out test records.
            """
        )

    st.markdown("---")
    st.markdown("### Interactive Dataset Explorer")
    selected_explore_cat = st.selectbox("Filter Explorer by Category", ["All Categories"] + LABELS, key="data_exp_sel")
    filtered_data_view = data if selected_explore_cat == "All Categories" else data[data["category"] == selected_explore_cat]
    st.dataframe(filtered_data_view[["complaint_text", "category"]], hide_index=True, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 12px 0;">
        🚦 <b>Pune Traffic Complaint Intelligence</b> • Built with Streamlit, spaCy & scikit-learn • Ephemeral Cloud Ready
    </div>
    """,
    unsafe_allow_html=True,
)
