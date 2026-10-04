# Phase 4 — probability calibration

Frozen Phase 3 classifier: `shallow`; split `20260806T135114Z_9d945cb2da0eecfc`; feature fingerprint `d6620d99301f80884487140bde487479f58fe72c1e209328524fa27e8b5a2271`. The base classifier was reconstructed with its frozen hyperparameters and fitted on TRAIN only.

Validation sample count: 6762. Raw validation behavior: **overconfident overall**. The raw Brier score is 0.24356895. The ten fixed-width bins contain 1 low-count populated bin; it is not interpreted independently.

Exactly RAW, SIGMOID (sklearn logistic mapping), and ISOTONIC (sklearn monotonic mapping) were fitted/evaluated on VALIDATION. Shared project evaluators supplied Brier, PR-AUC, ROC-AUC, and Top-K metrics.

| method | pr_auc | roc_auc | brier_score | recall_at_05 | recall_at_10 | recall_at_20 | precision_at_05 | precision_at_10 | precision_at_20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RAW | 0.46447317 | 0.55447923 | 0.24356895 | 0.05898681 | 0.10860514 | 0.22761971 | 0.50147493 | 0.46233383 | 0.48484848 |
| SIGMOID | 0.46447317 | 0.55447923 | 0.24286062 | 0.05898681 | 0.10860514 | 0.22761971 | 0.50147493 | 0.46233383 | 0.48484848 |
| ISOTONIC | 0.46684851 | 0.56226391 | 0.24108289 | 0.06037474 | 0.11866759 | 0.22831367 | 0.51327434 | 0.50516987 | 0.48632668 |

Calibration-curve evidence:

- RAW: 4 populated bins; maximum absolute bin gap 0.23024448.
- SIGMOID: 3 populated bins; maximum absolute bin gap 0.07627331.
- ISOTONIC: 5 populated bins; maximum absolute bin gap 0.00000000 on the validation data used to fit the flexible monotonic mapping; this is interpreted cautiously rather than as proof of generalization.

Policy: a calibrated method must improve raw Brier by the configured minimum and may not degrade any ranking or Top-K metric beyond the configured absolute guard. Lowest eligible Brier wins; effective ties prefer RAW, then SIGMOID, then ISOTONIC. RAW is explicitly valid.

Selected and frozen method: **ISOTONIC**. The TEST set was not accessed during selection.
