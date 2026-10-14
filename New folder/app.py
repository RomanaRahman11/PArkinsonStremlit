"""Parkinson's Disease — multimodal *research* dashboard.

Separate datasets, separate models, no participant-level multimodal fusion.
Feature-only inference is intentionally restricted to vectors produced with the
original upstream preprocessing. Not a medical device or diagnostic service.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from pd_app.core import (
    ROOT, SCHEMA, aggregate_voice_by_subject, infer,
    load_verified_model, template_csv,
)

METRICS = json.loads((ROOT / "data" / "metrics.json").read_text(encoding="utf-8"))

st.set_page_config(
    page_title="NeuroTrace | Parkinson's Research",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root {color-scheme:dark}
#MainMenu {visibility:hidden}
.block-container {max-width: 1260px; padding-top:2rem; padding-bottom:4rem}
[data-testid="stSidebar"] {background:linear-gradient(160deg,#102035 0%,#0b1929 100%); border-right:1px solid #20405a}
[data-testid="stMetric"] {background:#132b40;border-radius:14px;padding:16px;border:1px solid #24516b}
.hero {background:radial-gradient(circle at 85% 12%,rgba(58,197,193,.27),transparent 33%),linear-gradient(105deg,#19364c,#10263a 60%,#123b49);border:1px solid #2b6570;border-radius:23px;padding:35px 37px 31px;margin-bottom:20px}
.eyebrow {letter-spacing:.17em;text-transform:uppercase;color:#86f3e0;font-size:12px;font-weight:700}
.hero h1 {font-size:clamp(32px,4.5vw,54px);margin:8px 0 6px;color:#f1fbff;line-height:1.15;letter-spacing:-1.1px}
.hero p {color:#c7e0e7;font-size:17px;margin-top:8px;max-width:780px}
.subtitle {color:#a4c2cf;font-size:15px;margin-bottom:21px}
.glass {border:1px solid #244458;background:#102338;border-radius:18px;padding:23px;margin:10px 0 20px}
.glass .top {font-size:12px;color:#7be6d3;letter-spacing:.09em;text-transform:uppercase}
.glass h3 {color:#edfaff;font-size:24px;margin:6px 0}
.glass p {color:#b5d0da;margin:0;font-size:14px;line-height:1.6}
.pill {display:inline-block;border:1px solid #2a6b69;border-radius:999px;background:#143a43;padding:7px 12px;color:#8bf5df;font-size:12px;font-weight:600}
.warn {border-left:4px solid #dfad65;padding:14px 18px;background:#352d26;border-radius:0 10px 10px 0;color:#f3e4d1;font-size:14px}
.small {font-size:12px;color:#8eaaba}
footer {visibility:hidden}
</style>
""", unsafe_allow_html=True)


def banner(title: str, detail: str, eyebrow: str = "Research prototype") -> None:
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div>'
        f'<h1>{title}</h1><p>{detail}</p>'
        '<span class="pill">Exploratory · Not clinically validated</span></div>',
        unsafe_allow_html=True,
    )


def note() -> None:
    st.warning(
        "Research prototype only. Scores are **model outputs, not diagnoses or calibrated disease risks**. "
        "Do not use them for clinical decisions, screening, or reassurance."
    )


def show_metric_strip(kind: str) -> None:
    m = METRICS[kind]
    cols = st.columns(4)
    for col, label, value in zip(cols,
            ["Locked holdout", "Accuracy", "Balanced accuracy", "ROC-AUC"],
            [f'{m["n"]} people', f'{m["accuracy"]:.1%}',
             f'{m["balanced_accuracy"]:.1%}', f'{m["roc_auc"]:.3f}']):
        col.metric(label, value)


def show_gallery(kind: str) -> None:
    dirs = ROOT / "assets" / kind
    paths = sorted(dirs.glob("*.png"))
    if not paths:
        return
    with st.expander("Original research figures", expanded=False):
        labels = [p.stem.replace("_", " ") for p in paths]
        selected = st.selectbox("Select figure", labels, key=f"figure_{kind}")
        idx = labels.index(selected)
        st.image(str(paths[idx]), use_container_width=True)
        st.caption("From the original research archive; not generated from your upload.")


@st.cache_resource(show_spinner="Verifying and loading original model…")
def cached_model(kind: str):
    return load_verified_model(kind)


def show_prediction(kind: str, frame: pd.DataFrame) -> None:
    try:
        with st.spinner("Validating feature vectors and running original saved model…"):
            predictions = infer(kind, frame, cached_model(kind))
            grouped = aggregate_voice_by_subject(frame, predictions) if kind == "voice" else pd.DataFrame()
    except Exception as exc:
        st.error(str(exc))
        return

    st.success(f"Processed {len(predictions)} feature row(s). No model training occurred.")
    if len(predictions) == 1:
        p = float(predictions.loc[0, "pd_like_score"])
        st.metric("Research model score (PD-like class)", f"{p:.1%}")
        st.progress(p)
        st.caption(f"Model-assigned pattern at fixed threshold 0.50: {predictions.loc[0, 'model_pattern']}.")
    st.dataframe(predictions, use_container_width=True, hide_index=True)
    st.download_button(
        "Download results (.csv)", predictions.to_csv(index=False).encode("utf-8"),
        f"{kind}_research_predictions.csv", "text/csv", key=f"export_{kind}"
    )
    if not grouped.empty:
        st.markdown("#### Optional multi-recording subject aggregation")
        st.caption("Mean per-recording probability, matching the archived voice protocol. A single recording is not the original test protocol.")
        st.dataframe(grouped, use_container_width=True, hide_index=True)
        st.download_button("Download subject summary", grouped.to_csv(index=False).encode("utf-8"),
                           "voice_subject_summary.csv", "text/csv")
    st.caption("Scores can be affected by measurement setup, data shift, cohort confounding and tiny validation sets.")


def csv_uploader(kind: str, instructions: str) -> None:
    st.markdown("#### Upload extracted features")
    st.write(instructions)
    st.download_button(
        f"Download {len(SCHEMA[kind])}-feature CSV template",
        template_csv(kind), f"{kind}_feature_template.csv", "text/csv",
        key=f"template_{kind}"
    )
    uploaded = st.file_uploader(
        "CSV with exactly named feature columns (1–200 rows; up to 5 MB)",
        type=["csv"], key=f"upload_{kind}",
    )
    if uploaded is not None:
        if uploaded.size > 5 * 1024 * 1024:
            st.error("File exceeds 5 MB.")
            return
        try:
            data = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Cannot read CSV: {exc}")
            return
        st.write(f"**Loaded:** {len(data)} row(s), {len(data.columns)} column(s)")
        if st.button("Run research model", type="primary", key=f"run_{kind}"):
            show_prediction(kind, data)


def manual_entry(kind: str) -> None:
    st.markdown("#### Enter one feature vector manually")
    st.caption("Use the original measurement units. All fields are required; do not invent or approximate missing data.")
    with st.form(f"manual_{kind}"):
        cols = st.columns(2)
        values = {}
        for i, field in enumerate(SCHEMA[kind]):
            with cols[i % 2]:
                values[field] = st.text_input(field, value="", placeholder="Numeric value", key=f"{kind}_{i}")
        submitted = st.form_submit_button("Analyze feature vector")
    if submitted:
        try:
            if any(not value.strip() for value in values.values()):
                raise ValueError("Please complete every field using actual extracted feature values.")
            df = pd.DataFrame([{name: float(value) for name, value in values.items()}])
            if not np.isfinite(df.to_numpy(dtype=float)).all():
                raise ValueError("All values must be finite numbers.")
            show_prediction(kind, df)
        except ValueError as exc:
            st.error(str(exc))


def show_feature_tool(kind: str, instructions: str, *, allow_manual: bool = True) -> None:
    tab_titles = ["CSV upload", "Manual entry"] if allow_manual else ["Advanced feature CSV"]
    tabs = st.tabs(tab_titles)
    with tabs[0]:
        csv_uploader(kind, instructions)
    if allow_manual:
        with tabs[1]:
            manual_entry(kind)


def page_home() -> None:
    banner("NeuroTrace Research Lab",
           "A transparent exploration of three independent Parkinson’s research datasets — voice, handwriting and structural MRI.",
           eyebrow="Multimodal research dashboard · No patient-level fusion")
    note()
    st.markdown("### Independent holdout results")
    columns = st.columns(3)
    for col, key, tag in zip(columns, ("voice", "spiral", "mri"), ("Acoustic features", "Tablet trajectory", "Structural T1 MRI")):
        d = METRICS[key]
        with col:
            st.markdown(
                f'<div class="glass"><div class="top">{tag}</div>'
                f'<h3>{d["name"]}</h3><div style="font-size:38px;font-weight:800;color:#7eecda">{d["accuracy"]:.1%}</div>'
                f'<p>Accuracy · {d["n"]} held-out subjects<br>Balanced accuracy {d["balanced_accuracy"]:.1%} · AUC {d["roc_auc"]:.3f}</p></div>',
                unsafe_allow_html=True,
            )
    st.caption("Different populations, tasks and protocols: these numbers are descriptive and are NOT a direct comparison of modalities.")
    st.markdown("### What this application actually supports")
    c1, c2 = st.columns(2)
    with c1:
        st.info("**Voice** — upload the 22 original acoustic measurements. Saved classifier generates a research score; optional multiple-recording averaging.")
        st.info("**Spiral** — upload 14 precomputed trajectory features from two tests. Severe acquisition confounding remains unresolved.")
    with c2:
        st.info("**MRI** — review original figures, evaluation and a technical replay interface for precomputed 1,536-dimensional features. Raw MRI processing is not packaged.")
        st.warning("**No raw .wav, drawing image, tablet .txt or .nii/.nii.gz upload pathway.** The archive does not contain validated end-to-end feature extraction for those file formats.")
    st.markdown("### Dataset-specific limitations")
    for kind in ("voice", "spiral", "mri"):
        st.write(f"**{METRICS[kind]['name']}:** {METRICS[kind]['caveat']}")


def page_voice() -> None:
    banner("Voice analysis", "A saved Random Forest trained on 22 MDVP-style acoustic measurements. Upload feature vectors — not an audio file.", eyebrow="Modality 01 / acoustic")
    note()
    show_metric_strip("voice")
    st.markdown("---")
    show_feature_tool("voice", "Required inputs are the **22 named acoustic features** from the UCI Parkinson's voice schema. A WAV/MP3 recording cannot be converted here without the original validated extraction settings. An optional `subject_id` column allows averaging multiple recordings per subject.")
    st.markdown("---")
    show_gallery("voice")


def page_spiral() -> None:
    banner("Spiral handwriting", "Original saved 14-feature handwriting Random Forest. Intended for research reproduction, not screening.", eyebrow="Modality 02 / tablet trajectory")
    note()
    st.error("**Major validity issue:** original acquisition timing nearly separates the groups by itself; disease-specific signal has NOT been established. Subject IDs inferred from file names are provisional.")
    show_metric_strip("spiral")
    st.markdown("---")
    show_feature_tool("spiral", "Supply **14 precomputed x/y extent, pressure, and grip features** from tests 0 and 1. These must follow the same units and calculations as the original ParkinsonHW research processing; a photographed drawing or raw TXT cannot be correctly processed by this app.")
    st.markdown("---")
    show_gallery("spiral")


def page_mri() -> None:
    banner("MRI research findings", "Structural T1 results from the NEUROCON cohort. Investigational; poor locked-holdout discrimination.", eyebrow="Modality 03 / neuroimaging")
    note()
    st.error("**Not a reliable disease classifier:** 5/11 correct (45.45%) on the locked holdout, with sensitivity 2/7 (28.57%). Automated registration had unresolved anatomical-QC concerns.")
    show_metric_strip("mri")
    st.markdown("### Registered MRI QC & outcomes")
    c1, c2, c3 = st.columns(3)
    c1.metric("Development cohort", "32")
    c2.metric("Development QC flags", "15 / 32")
    c3.metric("Holdout QC flags", "5 / 11")
    st.caption("Frozen ImageNet ResNet18 features (3 views × 512), training-fold PCA(8), balanced logistic regression. Independent acquisition and image-registration pipeline is not shipped.")
    show_gallery("mri")
    st.markdown("### Expert-only: precomputed 1,536-feature vectors")
    st.warning("**This is a technical replay interface, not MRI diagnosis.** It only works for vectors produced by the *exact* frozen ResNet18 image-processing workflow used in the research. Do NOT upload raw MRI, other embeddings, randomly generated numbers or clinical data. The result is not validated for patients.")
    with st.expander("Open expert feature-vector interface", expanded=False):
        show_feature_tool("mri", "CSV must contain all `resnet18_feature_0000` through `resnet18_feature_1535` columns in their exact original order (reordering is handled by column name). No other representation is accepted.", allow_manual=False)


def page_results() -> None:
    banner("Research results", "A transparent record of the already-locked outcomes. No retraining, test-based model selection, or fusion.", eyebrow="Evidence & reproducibility")
    note()
    rows = []
    for key in ("voice", "spiral", "mri"):
        item = METRICS[key]
        rows.append({
            "Modality": item["name"], "Holdout n": item["n"],
            "Accuracy": item["accuracy"], "Balanced accuracy": item["balanced_accuracy"],
            "ROC-AUC": item["roc_auc"], "Sensitivity": item["sensitivity"],
            "Specificity": item["specificity"],
            "TN": item["tn"], "FP": item["fp"], "FN": item["fn"], "TP": item["tp"],
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={name: st.column_config.NumberColumn(format="%.3f")
                                for name in ["Accuracy", "Balanced accuracy", "ROC-AUC", "Sensitivity", "Specificity"]})
    st.caption("With holdouts of 8, 8, and 11, point estimates have large statistical uncertainty. No independent external validation.")
    st.bar_chart(df.set_index("Modality")[["Accuracy", "Balanced accuracy", "ROC-AUC"]], height=330)
    st.markdown("### Fundamental limitations")
    st.markdown("""
- **No patient-level multimodal fusion:** subjects in the voice, handwriting, and MRI datasets are not matched.
- **Voice:** recording-level input is not the same as a subject-level study; tiny holdout limits inference.
- **Spiral:** acquisition-only controls achieved near-perfect / perfect label separation in this cohort, undermining disease-specific interpretations.
- **MRI:** mean anatomical alignment scores and 5/11 correct are inadequate to support diagnosis; 5 of 11 holdout scans had QC review flags.
- **Privacy:** avoid uploading identifying or sensitive health information to public hosted apps.
""")
    with st.expander("Original consolidated research summary", expanded=False):
        st.markdown((ROOT / "data" / "original_research_summary.md").read_text(encoding="utf-8"))


def page_about() -> None:
    banner("Protocol & provenance", "Technical boundaries matter as much as a model score.", eyebrow="Transparent methods")
    st.markdown("### Artifact provenance")
    st.write("Saved trained models are copied from the supplied Kaggle research ZIP, checked by SHA-256 before loading. scikit-learn is pinned to the original training version (1.6.1).")
    st.code("Voice: 22 features → imputation → scaler → RF\nSpiral: 14 features → imputation → scaler → RF\nMRI: 1536 precomputed ResNet18 features → scaler → PCA(8) → Logistic Regression", language="text")
    st.markdown("### Safe use")
    st.write("This app does not intentionally save uploaded CSVs. Public hosting can involve platform-level logging and data handling; do not upload personally identifiable or medical records. Only submit de-identified feature vectors you have permission to use.")
    st.markdown("### Status")
    st.warning("Research / educational demonstration only. Not FDA/CE-approved, not externally validated, not a medical service. Consult a qualified clinician for health questions.")
    st.write("All original experimental reports, datasets, and images are documented separately in the research archive; only selected figures, model artifacts, and summary data are packaged for web deployment.")
    st.markdown("### Deployment")
    st.code("python -m pip install -r requirements.txt\nstreamlit run app.py", language="bash")


with st.sidebar:
    st.markdown("## 🧠 NeuroTrace")
    st.caption("Independent-modality PD research")
    st.divider()
    selection = st.radio(
        "Explore",
        ["Overview", "Voice", "Spiral handwriting", "MRI", "Research results", "Protocol & safety"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown('<span class="pill">Research prototype v1.0</span>', unsafe_allow_html=True)
    st.caption("No GPU · No training · Fixed historical models")
    st.caption("Not for clinical interpretation")

pages = {
    "Overview": page_home,
    "Voice": page_voice,
    "Spiral handwriting": page_spiral,
    "MRI": page_mri,
    "Research results": page_results,
    "Protocol & safety": page_about,
}
pages[selection]()
st.divider()
st.caption("NeuroTrace · Parkinson’s Research Prototype · Independent datasets · No clinical use")
