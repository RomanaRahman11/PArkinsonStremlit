"""Verify bundled saved-model hashes and run a one-row synthetic smoke test."""
from pathlib import Path
import sys
import sklearn
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pd_app.core import SCHEMA, infer, load_verified_model

if sklearn.__version__ != "1.6.1":
    raise RuntimeError("Archived models require scikit-learn 1.6.1. Install pinned requirements.txt first.")

for kind in ("voice", "spiral", "mri"):
    loaded = load_verified_model(kind)
    dummy = pd.DataFrame([{name: 0.1 for name in SCHEMA[kind]}])
    pred = infer(kind, dummy, model=loaded)
    print(f"{kind:7} VERIFIED | feature count={len(SCHEMA[kind])} | smoke score={pred.iloc[0]['pd_like_score']:.4f} (synthetic; not medically meaningful)")
print("ALL MODEL ARTIFACTS VERIFIED")
