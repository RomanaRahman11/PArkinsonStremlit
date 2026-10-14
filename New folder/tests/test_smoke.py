"""Run locally with: python -m unittest discover -s tests -v"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pd_app.core import (
    SCHEMA, aggregate_voice_by_subject, infer, load_verified_model,
    template_csv, validate_features,
)


class InferenceTests(unittest.TestCase):
    def test_feature_counts(self):
        self.assertEqual([len(SCHEMA[x]) for x in ("voice", "spiral", "mri")], [22, 14, 1536])

    @unittest.skipUnless(sklearn.__version__ == "1.6.1", "Requires original saved-model sklearn version 1.6.1")
    def test_model_integrity_and_prediction(self):
        for kind in ("voice", "spiral", "mri"):
            model = load_verified_model(kind)
            frame = pd.DataFrame([{name: 0.5 for name in SCHEMA[kind]}])
            pred = infer(kind, frame, model=model)
            self.assertEqual(len(pred), 1)
            self.assertTrue(0 <= float(pred.loc[0, "pd_like_score"]) <= 1)

    def test_reject_missing_and_bad_values(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            validate_features("voice", pd.DataFrame([{"not_a_feature": 2}]))
        df = pd.DataFrame([{name: 0.5 for name in SCHEMA["spiral"]}])
        df.iloc[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "Invalid/missing"):
            validate_features("spiral", df)

    def test_voice_grouped_mean(self):
        df = pd.DataFrame({"subject_id": ["A", "A", "B"]})
        preds = pd.DataFrame({"pd_like_score": [0.2, 0.6, 0.1]})
        g = aggregate_voice_by_subject(df, preds)
        self.assertEqual(g["recordings"].tolist(), [2, 1])
        self.assertAlmostEqual(float(g.loc[0, "pd_like_score"]), .4)

    def test_templates_headers(self):
        for kind in SCHEMA:
            self.assertEqual(template_csv(kind).decode().strip().split(","), SCHEMA[kind])


if __name__ == "__main__":
    unittest.main()
