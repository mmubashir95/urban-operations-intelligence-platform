# Rare and Unseen Categorical Handling

## Production decision

Phase 4 is **COMPLETE — GOVERNED NO-OP** for the current baseline.
`borough`, `location_type`, and `incident_zip` are supported by the reusable
handler but remain `CONDITIONAL`, prediction-time unresolved, and
`phase_2_allowed: false`. The frozen policy therefore exposes no active
categorical features to fit or transform. Configuration support does not imply
baseline approval. Activation requires both `production_decision: ACTIVE` in
the reviewed cardinality configuration and `APPROVED_CANDIDATE` plus the allow
flag in the frozen feature policy.

## Distinct category semantics

- `__MISSING__` means the source value was absent and is preserved exactly from
  Phase 3.
- `__RARE__` means a genuine category was observed in training but occurred
  fewer times than the fitted minimum-count threshold.
- `__UNKNOWN__` means a genuine later value was absent from the complete
  training vocabulary.

Unseen values are never mapped to missing or rare. Rare and unknown source
token collisions fail unless matching in-memory provenance proves that the
handler generated them.

## Threshold evidence and governed policy

Notebook 10 evaluated absolute count thresholds 10, 25, and 50 and relative
frequency thresholds 0.1%, 0.5%, and 1% using training data only. The evidence
is in `reports/11_split_aware_eda/tables/rare_category_analysis.csv`. The
canonical operator is strict: `count < min_count` maps to `__RARE__`, while
`count >= min_count` is retained.

| Supported field | Training cardinality | Count < 10 result | Decision |
| --- | ---: | ---: | --- |
| `borough` | 5 | 0 categories / 0 rows | No rare grouping indicated |
| `location_type` | 4 | 0 categories / 0 rows | No rare grouping indicated |
| `incident_zip` | 175 | 33 categories / 120 rows (0.506%) | Candidate only; field remains inactive |

The versioned configuration selects `minimum_count` with `min_count: 10` as
the lowest evaluated absolute-count candidate. It is simple, deterministic,
interpretable, and has limited impact in the diagnostic evidence. Because the
only supported high-cardinality field is inactive, this threshold is not an
active production transformation. Any future feature approval must review the
threshold against then-current training evidence before deployment.

## Training-only fit and immutable transform

`urban_ops.features.rare_unseen.fit_rare_unseen_handler` accepts only the
training frame and produces a frozen `FittedRareUnseenState` containing:

- policy and configuration versions;
- active columns;
- deterministic category counts and complete training vocabulary;
- retained and rare category tuples;
- the threshold and all three reserved tokens.

Categories are ordered lexically so identical training input and configuration
produce identical state and fingerprints. Validation, test, and inference are
accepted only by `transform_rare_unseen`; that function has no fitting path and
cannot update the artifact. Later frequencies therefore cannot change rarity
or vocabulary.

For an active field, transformation is:

```text
__MISSING__                       → __MISSING__
known retained training category → unchanged
known rare training category     → __RARE__
category absent from training    → __UNKNOWN__
```

The functions return deep copies and preserve index, order, identifiers,
targets, and unrelated columns. Evidence reports training cardinality,
retained/rare counts, affected training rows, later unseen values and rows, and
semantic reconciliation. No split or feature artifact is written in Phase 4.

## Boundary

Phase 4 does not activate conditional fields, refit Phase 3, infer categories,
use target values, fit on validation/test data, encode categories, build a
`ColumnTransformer`, persist preprocessing, or train a model. Phase 5 consumes
the Phase 4 output for one-hot encoding without changing missing, rare, or
unseen semantics. Latitude and longitude remain conditional with missing-value
handling deferred.

Next work: **Numeric preprocessing** after Phase 5 categorical encoding.
