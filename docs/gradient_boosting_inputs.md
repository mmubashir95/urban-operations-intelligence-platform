# Phase 2.1 Frozen Input Verification

Phase 2.1 reuses `load_frozen_baseline_inputs()` and returns the same
`FrozenBaselineInputs` object after verification. It does not rebuild cleaning,
targets, features, preprocessing, or split boundaries, and it does not train a
Gradient Boosting model.

The gate checks matrix/target alignment, the ordered frozen feature schema,
binary targets and both training classes. It then re-runs the Month 1 Phase 9
contract, which validates the authoritative chronological timestamps and the
canonical feature/leakage policy. Sparse matrices are inspected by shape and by
the existing sparse-aware contract; they are never converted to dense arrays.

The test split is checked only for structural contract integrity. Phase 2.1
does not fit on it, create predictions for it, or calculate test metrics.
