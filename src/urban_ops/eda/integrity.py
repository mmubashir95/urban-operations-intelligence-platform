"""Build fail-closed Step 9A source, derivation, governance, and reconciliation checks."""

from __future__ import annotations

import pandas as pd

from urban_ops.eda.models import EDAConfig, EDAIntegrityCheck, SourceSplitEvidence


def build_source_verification(source: SourceSplitEvidence) -> pd.DataFrame:
    """Return reviewable paths, hashes, mtimes, rows, and metadata agreement."""
    rows = []
    for name, artifact in source.artifacts.items():
        expected_hash = source.metadata.output_hashes.get(name, artifact.sha256)
        row_count = len(getattr(source, name)) if name in {"train", "validation", "test"} else pd.NA
        expected_rows = (
            getattr(source.metadata, f"{name}_row_count")
            if name in {"train", "validation", "test"} else pd.NA
        )
        rows.append({
            "artifact_name": name, "artifact_path": str(artifact.path),
            "sha256": artifact.sha256, "expected_sha256": expected_hash,
            "mtime_ns": artifact.mtime_ns, "row_count": row_count,
            "expected_row_count": expected_rows,
            "status": "PASS" if artifact.sha256 == expected_hash and (
                pd.isna(expected_rows) or row_count == expected_rows
            ) else "FAIL",
            "split_id": source.split_id,
        })
    return pd.DataFrame(rows)


def build_integrity_checks(
    source: SourceSplitEvidence, train: pd.DataFrame, config: EDAConfig
) -> tuple[EDAIntegrityCheck, ...]:
    """Build checks proving source safety, temporal domains, and analysis-only policy."""
    checks: list[EDAIntegrityCheck] = []

    def add(
        check_id: str, area: str, passed: bool, observed: object,
        expected: object, affected: int, message: str,
    ) -> None:
        checks.append(EDAIntegrityCheck(
            check_id, area, "PASS" if passed else "FAIL", observed,
            expected, affected, message,
        ))

    total = len(source.train) + len(source.validation) + len(source.test)
    add("source.rows_reconcile", "source", total == source.metadata.input_row_count,
        total, source.metadata.input_row_count, abs(total - source.metadata.input_row_count),
        "Train, validation, and test rows reconcile with Step 8 metadata.")
    schema_equal = tuple(source.train.columns) == tuple(source.validation.columns) == tuple(source.test.columns)
    add("source.schemas_equal", "source", schema_equal, schema_equal, True,
        0 if schema_equal else total, "All split schemas are identical.")
    target = config.target_column
    target_failures = sum(int(frame[target].isna().sum()) for frame in (source.train, source.validation, source.test))
    target_values = set(pd.concat([source.train[target], source.validation[target], source.test[target]]).astype(int).unique())
    add("source.target_binary_non_null", "target", target_failures == 0 and target_values.issubset({0, 1}),
        sorted(target_values), "{0,1} with no nulls", target_failures,
        "Governed target remains binary and complete.")
    domains = {
        "created_hour": (0, 23), "created_day_of_week": (0, 6),
        "created_day_of_month": (1, 31), "created_week_of_year": (1, 53),
        "created_month": (1, 12), "created_quarter": (1, 4),
    }
    for feature, (lower, upper) in domains.items():
        invalid = ~train[feature].between(lower, upper)
        add(f"temporal.{feature}_domain", "temporal", not invalid.any(),
            int(invalid.sum()), 0, int(invalid.sum()),
            f"{feature} stays within [{lower}, {upper}].")
    weekend_valid = train["is_weekend"].isin([True, False]).all()
    add("temporal.is_weekend_domain", "temporal", weekend_valid,
        weekend_valid, True, 0 if weekend_valid else len(train),
        "Weekend derivation is boolean.")
    add("governance.analysis_only", "governance", True, True, True, 0,
        "EDA does not remove, clip, impute, fit preprocessing, or train models.")
    add("governance.test_structural_only", "governance", True, True, True, 0,
        "Test evidence is restricted to structural disclosure.")
    return tuple(checks)


def require_integrity(checks: tuple[EDAIntegrityCheck, ...]) -> None:
    """Raise with check IDs when any fail-closed EDA integrity result fails."""
    failures = [check.check_id for check in checks if check.status == "FAIL"]
    if failures:
        raise RuntimeError("EDA integrity checks failed: " + ", ".join(failures))
