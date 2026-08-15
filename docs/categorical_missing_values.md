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
collision. In-memory provenance distinguishes a correctly transformed result,
making repeat application idempotent without accepting an unmarked collision.

## Separate meanings

- `__MISSING__`: the source supplied no categorical value.
- `__RARE__`: a low-support category observed in training; not implemented.
- `__UNKNOWN__`: a real category unseen during training; not implemented.

The handler does not infer `borough` from ZIP or coordinates, and it does not
replace unexpectedly missing deterministic calendar fields. Such calendar
missingness violates the validated `created_date` derivation contract and must
fail rather than be hidden.

## Deferred numeric handling

`latitude` and `longitude` are continuous numeric fields and remain
`CONDITIONAL`. They are not modified. If later approved, numeric handling must
use a training-derived placeholder, likely the median, and may separately
preserve a missingness indicator. A median would be a technical placeholder,
not a claim that a complaint occurred at the median coordinate.

Coordinates also form a pair. Later work must distinguish both coordinates
missing from only one coordinate missing; a partial pair may be an upstream
data-quality issue and must be checked before any numeric replacement.

No rare grouping, unseen-category mapping, numeric filling, encoding, scaling,
combined transformer, model matrix, or model training is implemented here.
