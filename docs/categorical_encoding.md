# Categorical Encoding

## Production decision

Phase 5 is **COMPLETE -- GOVERNED NO-OP** for the current baseline.
`borough`, `location_type`, and `incident_zip` are supported by the reusable
encoder but remain `CONDITIONAL`, prediction-time unresolved, and
`phase_2_allowed: false`. The frozen policy therefore exposes no active
categorical features to encode. Configuration support does not imply baseline
approval.

## Strategy

The selected baseline strategy is sklearn `OneHotEncoder` with:

```text
handle_unknown = ignore
drop = None
sparse_output = True
```

One-hot encoding keeps category effects independent and readable for the later
regularized logistic-regression baseline. `drop=None` keeps all categories,
including sentinel columns, so future coefficients remain easier to audit.
Sparse output avoids densifying a potentially wide categorical block.

`handle_unknown="ignore"` is only a defensive fallback. Phase 4 remains the
normal unseen-category policy and should map legitimate later unseen values to
`__UNKNOWN__` before encoding.

## Reserved-token columns

For each active categorical feature, Phase 5 builds the encoder vocabulary from
the training-fitted Phase 4 state:

```text
retained training categories
__MISSING__
__RARE__
__UNKNOWN__
```

The three sentinel columns are explicit even if a token is absent from the
training rows for that fitted split. This preserves the separate meanings:

- `__MISSING__`: the source value was unavailable.
- `__RARE__`: a training-known low-support category grouped by Phase 4.
- `__UNKNOWN__`: a legitimate later value absent from the training vocabulary.

Validation and test data never add categories, columns, or feature names.

## Training-only fitted state

`urban_ops.features.categorical_encoding.fit_categorical_encoder` accepts the
training frame after Phase 4 and the immutable `FittedRareUnseenState`. The
resulting `FittedCategoricalEncoder` records:

- config version and policy version;
- Phase 4 fitted-state fingerprint;
- active columns;
- categories per feature;
- encoded feature names;
- total encoded feature count;
- strategy, `handle_unknown`, `drop`, and sparse-output settings;
- deterministic fingerprint of the fitted encoder schema.

Validation, test, and inference use only `transform_categorical_encoder` or
`transform_split_categorical_encoder`. These functions return a separate sparse
matrix and do not mutate source frames.

## Phase 4 threshold rule

The canonical rare rule is:

```text
count < min_count  -> __RARE__
count >= min_count -> retained
```

Notebook 10 EDA diagnostics and Phase 4 implementation now use this same
strict less-than operator. The count threshold remains inactive in production
because no supported categorical field is approved.

## Boundary

Phase 5 does not activate conditional fields, change Phase 3 missing semantics,
change Phase 4 rare/unseen semantics, scale values, build a
`ColumnTransformer`, create a combined model matrix, persist a model pipeline,
train a model, or evaluate a model.

Latitude and longitude remain conditional with numeric missingness deferred.

Phase 6 handles numeric preprocessing separately as deterministic pass-through
for approved temporal features. Phase 7 composes the categorical and numeric
CSR blocks without refitting this encoder. See
`docs/preprocessing_composition.md`.

Next work after Phase 7: **Final preprocessing verification**.
