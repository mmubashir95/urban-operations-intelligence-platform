# Preprocessing Pipeline Composition

## Production decision

Phase 7 is **COMPLETE** when the already-governed categorical and numeric
preprocessing blocks are horizontally composed into one sparse model-ready
matrix.

The current real production state is:

- categorical encoded columns: 0
- numeric pass-through columns: 4
- combined columns: 4

Current combined feature names are derived from the upstream fitted states:

```text
created_hour
created_day_of_week
created_month
is_weekend
```

If categorical features are approved by future policy, their encoded feature
names will be placed before numeric feature names without changing the
composition contract.

## Composition order

The canonical Phase 7 order is:

```text
categorical encoded features
numeric features
```

Matrix composition must match the feature-name order exactly:

```text
combined_matrix = sparse.hstack([categorical_matrix, numeric_matrix])
combined_feature_names = categorical_feature_names + numeric_feature_names
```

This order is frozen for later coefficient interpretation, debugging, model
documentation, and artifact persistence.

## Fitted composition state

`urban_ops.features.preprocessing_composition.build_preprocessing_composition`
creates an immutable `FittedPreprocessingComposition` from the upstream fitted
Phase 5 and Phase 6 states. It records:

- config version and policy version;
- Phase 5 categorical encoder fingerprint;
- Phase 6 numeric preprocessor fingerprint;
- categorical feature names and count;
- numeric feature names and count;
- combined feature names and count;
- matrix type and dtype;
- composition order;
- `learned_statistics = none`;
- deterministic fingerprint.

Phase 7 does not learn category vocabularies, rare mappings, encoder
categories, numeric statistics, model parameters, or thresholds.

## Output contract

The composed output is:

- SciPy `csr_matrix`
- `float64`
- one row per source complaint
- one column per final feature name
- feature names are ordered, unique, and schema-stable
- target, identifiers, leakage fields, and inactive coordinates are absent

Sparse matrices do not carry DataFrame indexes. Row alignment is positional:
combined matrix row `i` corresponds to source frame row `i` at transform time.
Phase 7 validates categorical/numeric row-count equality before stacking and
does not sort or reorder rows.

## Split lifecycle

Training creates upstream fitted states once:

```text
Phase 4 fit on train
Phase 5 fit on train-derived Phase 4 state
Phase 6 fit/validate on train
Phase 7 build composition schema from fitted states
```

Validation, test, and inference are transform-only through those frozen states.
They cannot add columns, remove columns, reorder features, or change upstream
fingerprints.

## Boundary

Phase 7 does not refill missing values, recalculate rare categories, refit
encoder categories, redo numeric validation policy, activate conditional
geography, build a sklearn `Pipeline`, build a `ColumnTransformer`, train
`LogisticRegression`, calculate model metrics, or persist model artifacts.

Phase 8 now verifies this composed output against its targets, identifiers,
chronology, feature policy, sparse format, finite-value contract, train-only
variance, and complete upstream fingerprint chain. A successful check freezes
the schema for baseline modelling. See docs/preprocessing_verification.md.

Next work: **Baseline Modelling** after successful Phase 8 verification.
