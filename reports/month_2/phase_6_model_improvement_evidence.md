# Phase 6 — model-improvement evidence

This document records evidence, not implemented fixes.

- **Feature limitations:** 1705 false negatives and 181 bottom-decile false negatives under the primary Top-10% policy. Potential future investigation: evaluate prediction-time-safe complaint, service, or geographic signals under a new governed feature policy.
- **Temporal drift:** target prevalence changes from 0.460 in TRAIN to 0.350 in TEST. Potential future investigation: validate rolling temporal stability and retraining policy.
- **Complaint-category effects:** not measurable within the single Graffiti scope. Potential future investigation: only broaden scope through a separate governed scope decision.
- **Geographic effects:** Supported borough FNR ranges from 0.830 to 0.946. Potential future investigation: assess safe geography features without treating association as cause.
- **Score separation:** raw TEST PR-AUC is documented by Phase 4 as weak; errors span deterministic score ranges. Potential future investigation: compare richer governed feature sets out of sample.
- **Calibration:** isotonic remains frozen but did not improve TEST Brier versus raw in Phase 4. Potential future investigation: reassess calibration only after a new base model is governed.
- **SHAP:** `created_month` dominates the four-feature model and `is_weekend` contributes no reported influence. Potential future investigation: test whether new safe signals improve discrimination before re-explaining.

No item above is a causal conclusion or an implemented model change.
