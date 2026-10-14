"""Integrity-checked, feature-only inference for a research prototype.

NOT a medical device. Do not feed audio, raw handwriting trajectories, or raw MRI
into this module: trained feature-extraction algorithms are not distributed here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import sklearn
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILES = {
    "voice": ROOT / "models" / "voice_random_forest.joblib",
    "spiral": ROOT / "models" / "spiral_random_forest.joblib",
    "mri": ROOT / "models" / "mri_resnet18_classifier.joblib",
}
SCHEMA = json.loads((ROOT / "data" / "feature_schemas.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((ROOT / "data" / "model_manifest.json").read_text(encoding="utf-8"))
MAX_ROWS = 200


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified_model(kind: str):
    """Check distributed model checksum before joblib.load (trusted assets only)."""
    if kind not in MODEL_FILES:
        raise ValueError(f"Unknown model: {kind}")
    if sklearn.__version__ != "1.6.1":
        raise RuntimeError(
            f"This archived model requires scikit-learn 1.6.1, but {sklearn.__version__} "
            "is installed. Run: pip install -r requirements.txt"
        )
    path = MODEL_FILES[kind]
    expected = MANIFEST[kind]["sha256"]
    if not path.is_file() or file_sha256(path) != expected:
        raise RuntimeError(f"Model integrity check failed: {kind}")
    model = joblib.load(path)
    if getattr(model, "n_features_in_", None) != len(SCHEMA[kind]):
        raise RuntimeError(f"Model feature-count mismatch: {kind}")
    if hasattr(model, "feature_names_in_"):
        actual = model.feature_names_in_.tolist()
        if actual != SCHEMA[kind]:
            raise RuntimeError(f"Model feature schema mismatch: {kind}")
    if not np.array_equal(model.classes_, [0, 1]):
        raise RuntimeError(f"Model class encoding mismatch: {kind}")
    return model


def validate_features(kind: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Enforce exact numeric feature schema; may accept additional metadata columns."""
    if kind not in SCHEMA:
        raise ValueError("Unknown research modality")
    if not (1 <= len(frame) <= MAX_ROWS):
        raise ValueError(f"Provide 1–{MAX_ROWS} feature rows per request.")
    if not frame.columns.is_unique:
        raise ValueError("Duplicate CSV columns detected; rename them before uploading.")
    expected = SCHEMA[kind]
    missing = [name for name in expected if name not in frame.columns]
    if missing:
        example = ", ".join(missing[:5])
        raise ValueError(f"Missing {len(missing)} required feature columns: {example}")
    numeric = frame[expected].apply(pd.to_numeric, errors="coerce")
    matrix = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        problems = np.argwhere(~np.isfinite(matrix))
        row_idx, col_idx = problems[0]
        raise ValueError(
            f"Invalid/missing feature at row {row_idx + 1}, column '{expected[col_idx]}'. "
            "Supply finite numeric values from the original feature pipeline."
        )
    # Keep pandas feature names for voice and spiral, model expects numpy for MRI.
    return numeric.astype(np.float64)


def infer(kind: str, frame: pd.DataFrame, model=None) -> pd.DataFrame:
    X = validate_features(kind, frame)
    model = model if model is not None else load_verified_model(kind)
    prediction_input = X.to_numpy(dtype=np.float32) if kind == "mri" else X
    scores = np.asarray(model.predict_proba(prediction_input)[:, 1], dtype=np.float64)
    if len(scores) != len(frame) or not np.isfinite(scores).all():
        raise RuntimeError("Invalid model outputs")
    classes = np.where(scores >= 0.5, "PD-like pattern", "Control-like pattern")
    return pd.DataFrame({
        "row": np.arange(1, len(frame) + 1),
        "pd_like_score": np.round(scores, 5),
        "model_pattern": classes,
    })


def aggregate_voice_by_subject(frame: pd.DataFrame, row_results: pd.DataFrame) -> pd.DataFrame:
    """Original research aggregation = mean of recording probabilities by subject."""
    if "subject_id" not in frame:
        return pd.DataFrame()
    ids = frame["subject_id"].astype("string").str.strip()
    if ids.isna().any() or (ids == "").any():
        raise ValueError("The subject_id column contains empty participant IDs")
    data = pd.DataFrame({
        "subject_id": ids.to_numpy(),
        "pd_like_score": row_results["pd_like_score"].to_numpy(),
    })
    groups = data.groupby("subject_id", sort=False).agg(
        recordings=("pd_like_score", "size"),
        pd_like_score=("pd_like_score", "mean"),
    ).reset_index()
    groups["pd_like_score"] = groups["pd_like_score"].round(5)
    groups["model_pattern"] = np.where(
        groups["pd_like_score"] >= 0.5,
        "PD-like pattern",
        "Control-like pattern",
    )
    return groups


def template_csv(kind: str) -> bytes:
    return pd.DataFrame(columns=SCHEMA[kind]).to_csv(index=False).encode("utf-8")
