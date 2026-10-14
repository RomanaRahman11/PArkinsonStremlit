# NeuroTrace — Parkinson's Research Dashboard

**Status:** Research-only Streamlit prototype, **not a medical device**.

This deployment-ready project was built from the provided `PD_FINAL_THREE_MODALITY_RESEARCH.zip` saved Kaggle research archive. It includes the **original saved models**, selected historical research figures, research metrics, a hash-verification manifest, and a web interface. **There is no retraining or test re-evaluation.**

## What works

| Section | Input | Output | Limitation |
|---|---|---|---|
| Voice | CSV of **22 original MDVP features**, optional `subject_id`; manual 22 fields | Saved Random Forest score; optional multi-recording mean | Audio `.wav` processing not included; 8-person test set |
| Spiral | CSV of **14 exact features** from ParkinsonHW tests 0 and 1; manual fields | Saved Random Forest score | Severe acquisition confounding; raw TXT/drawing/image processing not included |
| MRI | Expert-only **1,536 original ResNet18 feature values** | Frozen PCA+Logistic regression research score | Raw NIfTI preprocessing not included; **45.45% test accuracy**, unacceptable for clinical use |
| Results | None | Archived figures and locked holdout metrics | Populations differ; **no real multimodal fusion** |

**Do not input personal medical data or use any result to determine whether someone has Parkinson's disease.** The score is a model class score, **not** an individualized clinical probability.

## Fast local launch (no GPU)

Using Python **3.13** is recommended (the original models were serialized on Python 3.13 with scikit-learn 1.6.1):

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_artifacts.py
streamlit run app.py
```

Streamlit opens a local browser URL (usually `http://localhost:8501`). This is not a published live URL.

## Deploy to Streamlit Community Cloud

1. Extract this ZIP. Create a GitHub repository (prefer **private** if your research artifacts should not be public). Upload the **contents** of the project folder to the repository root, including `app.py`, `requirements.txt`, `models/`, `pd_app/`, `assets/` and `data/`. Do **not** upload the original large source datasets or patient-identifiable files.
2. Open **https://share.streamlit.io**, connect GitHub, select **Create app** and your repository. Set entrypoint to **`app.py`**.
3. In **Advanced settings**, select **Python 3.13** if available. Deploy. The app will install dependencies from `requirements.txt` and load its saved model files after SHA-256 checks.
4. Open the deployed URL. Test the **Overview**, **Voice**, **Spiral**, **MRI**, **Research results**, and **Protocol** pages. Download a feature template to understand required CSV inputs.

Community Cloud uses GitHub repo source + dependency requirements; more information: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy .

**Important:** Nothing has been deployed to a public website automatically. GitHub authorization and pressing Deploy are still required. If you share the app publicly, remember the limitations and privacy concerns.

## Model file integrity

The three packaged `.joblib` files are SHA-256-pinned in `data/model_manifest.json`. A checksum is verified **before** `joblib.load`; never replace these with random or untrusted joblib files. Joblib/pickle loading executes Python objects; load only authentic, controlled artifacts. Dependencies pin `scikit-learn==1.6.1` to match the original saved models.

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Fixed historical holdout outcomes

| Modality | Holdout | Accuracy | Balanced accuracy | ROC-AUC | Notes |
|---|---:|---:|---:|---:|---|
| Voice | 8 | 87.50% | 75.00% | 1.000 | Very small holdout; uncertain external performance |
| Spiral | 8 | 87.50% | 83.33% | 0.800 | Acquisition-only negative controls explain labels far too well |
| MRI | 11 | 45.45% | 51.79% | 0.536 | 5/11 correct; registration QC incomplete; 5/11 priority flagged |

These are **not** head-to-head scores and **cannot** be interpreted as patient-level fusion performance. Raw source training data is not included in the deployable package. Original research results are embedded without changing validation or retuning.

## Workflow and feature preparation

The voice/spiral models accept numeric features, not raw recordings or pictures. Download the CSV templates inside each application tab and populate all named columns from the **same original acquisition and feature-extraction methods**. The app reorders by exact feature names and validates finite numeric values. The voice model optionally computes mean scores across recordings for a shared `subject_id` field. Raw `.wav`, `.txt`, hand-drawn images, and `.nii.gz` **are not supported** because the provided archive did not include an independently reproducible, clinically validated raw-signal preprocessing service.

MRI expert input requires 1,536 numerical outputs of the exact 3-view ImageNet ResNet18 preprocessing used in the original Kaggle research. **Do not** substitute other feature-generation approaches; upload of a raw brain scan is **not** implemented.

## Privacy & limitations

Uploaded feature data is read in memory for the app session; this code does not intentionally save it to server storage. Deployed hosting providers may have their own logs and retention practices. Avoid personally identifying or health records entirely. No medical claims, patient screening, disease risk calibration, or guaranteed cross-site generalization is provided.

The study has unresolved validity threats: only eight voice and spiral test subjects each; spiral sample IDs are provisional and its acquisition protocol strongly confounded with diagnosis; MRI registration was not anatomically approved, with 15/32 dev and 5/11 test scans priority-flagged and all development optimizers reaching their iteration ceilings.

---

**বাংলায়:** এটি একটি *Research Prototype*, রোগ নির্ণয়ের সফটওয়্যার নয়। Voice ও Spiral-এর জন্য আগে থেকে বের করা feature CSV দিতে হবে। MRI-তে আসল scan আপলোড সমর্থিত নয়—শুধু original 1536-dimensional ResNet18 features দিয়ে গবেষণামূলক replay করা যায়। তিনটি dataset-এর রোগী আলাদা, তাই আসল multimodal fusion করা হয়নি।
