# Parkinson's Disease Detection
## Three-Modality Research Summary

**Research status:** Exploratory; not clinically validated.

## Independent Holdout Results

| Modality | N | Accuracy | Balanced Accuracy | ROC-AUC |
|---|---:|---:|---:|---:|
| Voice | 8 | 87.50% | 75.00% | 1.0000 |
| Spiral | 8 | 87.50% | 83.33% | 0.8000 |
| MRI | 11 | 45.45% | 51.79% | 0.5357 |

## 1. Voice
- UCI voice recordings: 195 recordings from 32 subjects.
- Subject-grouped development/testing.
- Final classifier: Random Forest.
- Holdout: 7/8 correct.
- Limitations: Very small holdout and substantial uncertainty.
- ROC-AUC of 1.0 on eight subjects does not establish perfect discrimination.

## 2. Spiral Handwriting
- ParkinsonHW main cohort: 40 handwriting files.
- Final classifier: Random Forest using handwriting features.
- Holdout: 7/8 correct.
- Acquisition timing strongly confounded with diagnosis.
- Acquisition-only negative control performed exceptionally well.
- Filename-based participant IDs are provisional;
  guaranteed subject independence was not established.
- Therefore disease-specific discrimination is not demonstrated.

## 3. Structural Brain MRI
- NEUROCON: 43 participants.
- Development: 32; locked holdout: 11.
- Frozen ImageNet ResNet18 features, PCA(8),
  balanced logistic regression.
- Holdout: 5/11 correct.
- Sensitivity: 2/7; specificity: 3/4.
- Development QC flags: 15/32.
- Holdout QC flags: 5/11.
- Anatomical registration approval remains pending.
- The MRI model did not demonstrate reliable PD classification.

## Methodological Interpretation
These are three independent datasets with different
participants, tasks and acquisition characteristics.

Their scores are descriptive and must not be treated
as a head-to-head modality comparison.

Patient-level multimodal fusion has NOT been performed.

## Overall Conclusion
Voice and spiral results are preliminary and constrained
by very small holdouts. Spiral acquisition confounding
is a major validity concern.

The MRI classifier did not establish useful disease
discrimination on its locked holdout.

No clinical diagnostic performance is claimed.

## Reproducibility
The accompanying archive preserves available research
metrics, selected model artifacts, predictions and figures.

Original source datasets and the external ImageNet
ResNet18 checkpoint may still be required to reproduce
the full pipelines. Raw MRI volumes are not included.
