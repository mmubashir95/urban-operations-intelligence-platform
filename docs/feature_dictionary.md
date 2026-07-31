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
