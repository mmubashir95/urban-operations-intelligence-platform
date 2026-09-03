# Numeric Preprocessing

## Production decision

Phase 6 is **COMPLETE -- DETERMINISTIC PASS-THROUGH** for the current
baseline.

The real production policy exposes exactly these active numeric model
features:

- `created_hour`
- `created_day_of_week`
- `created_month`
- `is_weekend`

`latitude` and `longitude` remain `CONDITIONAL`, prediction-time `UNRESOLVED`,
`phase_2_allowed: false`, and numeric missingness `DEFERRED`. They are
supported by configuration only as governed deferred fields; support does not
imply baseline approval.

## Per-feature policy

| Feature | Strategy | Reason |
| --- | --- | --- |
| `created_hour` | `pass_through` | Frozen Month 1 temporal representation; raw hour remains interpretable. |
| `created_day_of_week` | `pass_through` | Frozen Month 1 weekday representation; raw integer remains interpretable. |
| `created_month` | `pass_through` | Frozen Month 1 month representation; raw integer remains interpretable. |
| `is_weekend` | `pass_through` | Binary indicator is already model-ready. |
| `latitude` | `deferred` | Conditional geography is not approved for the baseline. |
| `longitude` | `deferred` | Conditional geography is not approved for the baseline. |

No `StandardScaler`, median imputation, coordinate scaling, coordinate
normalization, cyclic sine/cosine expansion, or categorical encoding of the
temporal integers is active in Phase 6.

## Cyclic semantics

`created_hour`, `created_day_of_week`, and `created_month` are cyclic by
nature. The Month 1 baseline deliberately keeps the frozen raw temporal
features for simplicity, interpretability, and direct alignment with the
approved feature policy. Cyclic sine/cosine alternatives may be evaluated in a
future governed modelling comparison, but adding those columns would reopen
feature-policy and deterministic-feature decisions.

## Validation behavior

Active deterministic temporal features must be present, non-null, numeric,
finite, integer-like where applicable, and inside their governed domains:

- `created_hour`: 0 through 23
- `created_day_of_week`: 0 through 6
- `created_month`: 1 through 12
- `is_weekend`: 0 or 1 after numeric materialization

Nulls or invalid ranges fail loudly. They are treated as upstream invariant
violations from deterministic feature creation, not values to impute.

## Fitted state

`urban_ops.features.numeric_preprocessing.fit_numeric_preprocessor` creates an
explicit immutable `FittedNumericPreprocessor` containing:

- config version and policy version;
- active numeric columns;
- deterministic feature order;
- per-feature preprocessing rule;
- output feature names;
- matrix type and dtype;
- `learned_statistics = none`;
- deterministic fingerprint.

There are no learned numeric statistics in the current production state.
Training-only fitting is therefore trivially satisfied: validation, test, and
inference can validate and transform only; they cannot alter feature order,
schema, or fitted state.

## Output and alignment

`transform_numeric_preprocessor` returns a separate SciPy CSR matrix with
`float64` values. It does not mutate the source DataFrame and does not include
targets, identifiers, leakage fields, latitude, or longitude.

Sparse matrices do not carry DataFrame indexes. The row-alignment contract is
positional: matrix row `i` corresponds to source frame row `i` at transform
time. Phase 7 must preserve this positional alignment when composing
preprocessing blocks.

## Boundary

Phase 6 does not build a `ColumnTransformer`, fit a model pipeline, train a
model, evaluate a model, activate conditional geography, or implement
coordinate imputation.

Phase 7 now composes this numeric CSR block with the separate categorical CSR
block without changing numeric semantics. See
`docs/preprocessing_composition.md`.

Next work after Phase 7: **Final preprocessing verification**.
