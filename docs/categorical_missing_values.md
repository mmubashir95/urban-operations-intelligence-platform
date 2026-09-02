# Categorical Missing-Value Handling

## Purpose and governance boundary

Categorical nulls use one deterministic representation: `__MISSING__`. The
token means only that the source record did not provide a value. It does not
mean that a category is rare, invalid, or unseen during training.

The constant and supported categorical columns are configured in
`configs/features/resolution_risk_categorical_missing.yaml`. Feature eligibility
continues to come only from `configs/features/resolution_risk_baseline.yaml`.
Configuration support does not activate a field.

`borough`, `location_type`, and `incident_zip` remain `CONDITIONAL`, so the
current baseline applies no categorical replacement to the authoritative split
views. The reusable handler and its tests establish the behavior that will be
used if governance later approves a categorical field. The frozen eligibility
policy is not changed.

## Constant rule

For a policy-approved categorical field:

```text
null → __MISSING__
```

The handler returns a copy, preserves row identity and ordering, converts the
selected values to pandas' nullable string dtype, and learns no train, target,
validation, or test statistic. Existing non-null categories are preserved.
`incident_zip` is categorical text; means, medians, and interpolation have no
valid meaning for ZIP identifiers.

A real source value equal to `__MISSING__` is rejected as an ambiguous token
collision. DataFrame `attrs` provenance distinguishes a correctly transformed
result, so idempotency is explicitly **in-memory idempotency**. CSV, Parquet,
or another serialization/reload boundary may discard that provenance. If a
reloaded transformed field contains `__MISSING__` without provenance, repeat
application fails loudly on the apparent collision. That is an intentional
fail-safe; Phase 3 does not add a persistence protocol for provenance.

Raw categorical cleanup occurs before feature preprocessing. In particular,
the cleaning policy trims `incident_zip` text and converts empty or
whitespace-only ZIP values to null. It preserves valid ZIP text, including
leading zeroes, and never coerces ZIP identifiers to numbers. The categorical
feature handler then represents a legitimate remaining null as `__MISSING__`
only when feature policy activates the field.

## Separate meanings

- `__MISSING__`: the source supplied no categorical value.
- `__RARE__`: a low-support category observed in training; not implemented.
- `__UNKNOWN__`: a real category unseen during training; not implemented.

The handler does not infer `borough` from ZIP or coordinates, and it does not
replace unexpectedly missing deterministic calendar fields. Such calendar
missingness violates the validated `created_date` derivation contract and must
fail rather than be hidden.

## Invalid-value and preprocessing boundary

Validation and cleaning own raw-value correctness. They handle malformed raw
values, trim whitespace, convert blank strings to null, validate coordinate
ranges, identify impossible coordinate pairs, normalize invalid values when an
approved cleaning rule requires it, and record audit counts. Existing
coordinate validation remains authoritative; inactive geography does not
justify expanding Phase 3 into a coordinate-cleaning redesign.

Feature preprocessing starts only after that boundary. It handles legitimate
missing values that remain after cleaning. It does not decide whether a raw
value is valid, infer a borough or other category, or infer geographic values.

## Governed numeric deferral

The machine-readable decision is recorded under
`implementation_boundary.numeric_missingness` in
`configs/features/resolution_risk_baseline.yaml`:

| Feature | Current feature status | Prediction-time status | Missing-value handling |
| --- | --- | --- | --- |
| `latitude` | `CONDITIONAL` | `UNRESOLVED` | `DEFERRED` |
| `longitude` | `CONDITIONAL` | `UNRESOLVED` | `DEFERRED` |

The project does not define fitted preprocessing for features that are not
approved for the baseline. Numeric missing-value handling is revisited only if
latitude or longitude becomes `APPROVED_CANDIDATE`.

If that trigger occurs, the governed intended design is to:

1. Validate the coordinate pair upstream first.
2. Normalize invalid or impossible coordinates to missing in validation or
   cleaning and record audit counts.
3. Fit numeric imputation values using training data only.
4. Apply the training-fitted values to validation, test, and inference data.
5. Evaluate `latitude_missing` and `longitude_missing` indicators.
6. Treat any imputed coordinate as a technical placeholder, never as the
   complaint's true location.
7. Persist the fitted preprocessing with the model pipeline.

No latitude/longitude imputation or missingness indicator is implemented or
activated in Phase 3.

No rare grouping, unseen-category mapping, numeric filling, encoding, scaling,
combined transformer, model matrix, or model training is implemented here.
