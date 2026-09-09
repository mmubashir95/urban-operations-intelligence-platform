# Final Preprocessing Verification

## Production decision

Phase 8 independently verifies preprocessing Phases 2–7 and freezes the result
as **FROZEN / MODEL-READY**. It adds no transformations and does not train or
evaluate a model.

The current production schema, derived from the fitted Phase 7 composition, is:

| Column | Feature | Branch | Source feature | Stage |
| ---: | --- | --- | --- | --- |
| 0 | created_hour | numeric | created_hour | phase_6_pass_through |
| 1 | created_day_of_week | numeric | created_day_of_week | phase_6_pass_through |
| 2 | created_month | numeric | created_month | phase_6_pass_through |
| 3 | is_weekend | numeric | is_weekend | phase_6_pass_through |

Row counts are not hard-coded. They come from the authoritative chronological
splits and are recorded in each verified contract.

## Final X / y / identifier contract

For train, validation, and test, Phase 8 receives:

- X as a two-dimensional SciPy CSR matrix with float64 values;
- y as the one-dimensional authoritative missed_resolution_target;
- unique_key as a separate one-dimensional traceability series;
- created_date as the UTC chronology series;
- the exact feature-name tuple from frozen Phase 7 state.

Matrix row i, target position i, identifier position i, and timestamp position i
refer to the same complaint. Phase 8 checks counts and pandas indexes and never
sorts, shuffles, or repairs these inputs. The identifier remains available for
evaluation and debugging but never enters X.

## Verification gate

The verify_preprocessing_contract function fails closed unless:

- every required split exists and remains chronological;
- all matrices are CSR, float64, finite, non-empty, and share one exact schema;
- target and identifier counts match matrix rows;
- targets are non-null, binary, and contain both classes in training;
- identifiers are non-null, unique, and disjoint across splits;
- final names are non-empty and unique;
- every final column traces through fitted Phase 5 or 6 state to an approved,
  creation-time available, leakage-safe source feature;
- target, identifier, leakage, excluded, and conditional geography are absent;
- Phase 5 references the supplied Phase 4 fingerprint;
- Phase 7 references the supplied Phase 5 and Phase 6 fingerprints;
- no training feature is all-zero or otherwise zero-variance.

The finite-value check inspects sparse matrix data without densifying the full
matrix. Train-only variation is also computed from sparse columns. An
unexpected zero-variance feature is reported and fails the gate; it is never
silently removed.

## Frozen artifact and evidence

VerifiedPreprocessingContract records policy and verification versions, Phase
4–7 fingerprints, ordered feature names and count, matrix type and dtype, split
order and row counts, target/identifier/timestamp names, schema fingerprint,
status, and a deterministic contract fingerprint. Its to_dict and from_dict
methods provide stable JSON persistence without embedding matrices.

Reusable evidence builders produce the ordered schema table, per-split
readiness evidence, and train-only variance evidence. Notebook 11 displays
these tables in memory and does not overwrite prior reports or split artifacts.

## Freeze and inference semantics

After a successful Phase 8 gate, feature names and order, missing tokens,
training-fitted rare/unseen vocabulary, encoder vocabulary, numeric strategy,
and composition order are frozen. Future changes must deliberately reopen the
relevant earlier phase and rebuild downstream fingerprints.

Inference must apply the same deterministic temporal creation, frozen
categorical missing rule, Phase 4 vocabulary, Phase 5 encoder, Phase 6 numeric
pass-through, and Phase 7 order. Validation, test, and inference remain
transform-only and may not learn preprocessing state.

## Scope boundary

Phase 8 does not add features, scaling, imputation, coordinate processing,
categorical activation, a ColumnTransformer, an sklearn Pipeline, model
fitting, probability prediction, metrics, threshold selection, or model
persistence.

Next work: **Baseline Modelling**.
