# Feature Dictionary

The authoritative row-level inventory is generated from
`urban_ops.features.feature_roles` into
`reports/06_target_and_leakage/tables/feature_role_inventory.csv`.

Each row contains `source_column`, `internal_feature_name`, `data_type`, `role`,
`available_at_creation`, `mutable_after_creation`, `allowed_for_baseline`,
`null_policy`, `leakage_reason`, `decision_status`, and notes.

## Decision summary

| Role | Handling |
| --- | --- |
| `SAFE_FEATURE` | Baseline use allowed; derive calendar values from `created_date` |
| `CONDITIONAL_FEATURE` | Block unless the pipeline explicitly approves it |
| `IDENTIFIER` | Block |
| `TARGET_INPUT` | Use for label construction only |
| `POST_CREATION_FIELD` | Block |
| `TARGET_DERIVED` | Block |
| `EXCLUDED` | Block |

Conditional fields are deliberately not described as safe. See
`docs/leakage_policy.md` for the prediction boundary and due-date decision.

`status` is classified as `POST_CREATION_FIELD` (always blocked), not
`CONDITIONAL_FEATURE`, because only its final value is captured and it is
mutable after creation by definition; there is no creation-time snapshot of
status to conditionally approve.

## Categorical preprocessing state

The Phase 4 handler supports `borough`, `location_type`, and `incident_zip`,
but support is not activation. All three remain `CONDITIONAL`, so the current
baseline fits no categorical vocabulary and transforms none of them. If policy
later approves one, training-fitted categories remain literal, training-known
low-count categories use `__RARE__`, later unseen categories use `__UNKNOWN__`,
and Phase 3 missing values remain `__MISSING__`. See
`docs/rare_and_unseen_categories.md`.

The Phase 5 encoder supports the same categorical fields and remains inactive
for the same reason. If a field is later approved, one-hot categories are fitted
from the training-fitted Phase 4 state only, sparse output is produced, and
`__MISSING__`, `__RARE__`, and `__UNKNOWN__` receive explicit encoded columns.
See `docs/categorical_encoding.md`.

## Numeric preprocessing state

Phase 6 actively preprocesses only the approved deterministic temporal numeric
features: `created_hour`, `created_day_of_week`, `created_month`, and
`is_weekend`. The strategy is pass-through with domain validation and
deterministic ordering; no numeric statistics are learned.

`latitude` and `longitude` remain conditional/deferred and are excluded from
the numeric output. Targets, identifiers, and leakage fields are excluded by
allowlist rather than broad numeric dtype discovery. See
`docs/numeric_preprocessing.md`.

## Composed preprocessing state

Phase 7 composes the Phase 5 categorical CSR block and Phase 6 numeric CSR
block into one final sparse model-ready matrix. The frozen order is categorical
encoded feature names first, followed by numeric feature names. In the current
production policy this yields the four numeric temporal features only:
`created_hour`, `created_day_of_week`, `created_month`, and `is_weekend`.

The composed state records upstream fingerprints and a deterministic final
feature-name schema. It does not train a model or relearn any upstream
preprocessing state. See `docs/preprocessing_composition.md`.

## Final verified model-input state

Phase 8 reconciles every final column to explicit fitted-state lineage and the
frozen feature policy. The authoritative current schema is:

| Index | Feature | Source branch | Source feature |
| ---: | --- | --- | --- |
| 0 | created_hour | numeric | created_hour |
| 1 | created_day_of_week | numeric | created_day_of_week |
| 2 | created_month | numeric | created_month |
| 3 | is_weekend | numeric | is_weekend |

Targets and unique_key identifiers remain separate, positionally aligned
vectors. Conditional geography and all leakage/excluded fields remain absent.
See docs/preprocessing_verification.md.
