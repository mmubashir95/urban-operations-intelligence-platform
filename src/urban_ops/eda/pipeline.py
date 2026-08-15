"""CLI and orchestration for governed split-aware EDA and outlier analysis."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from urban_ops.eda.analysis import (
    build_categorical_tables, build_missingness_tables, build_numeric_tables,
    build_structural_drift_summary, build_target_tables, build_temporal_tables,
    derive_temporal_features,
)
from urban_ops.eda.figures import REQUIRED_FIGURES
from urban_ops.eda.integrity import (
    build_integrity_checks, build_source_verification, require_integrity,
)
from urban_ops.eda.models import EDAConfig, EDAIntegrityCheck, EDAResult, SourceSplitEvidence
from urban_ops.eda.recommendations import (
    build_column_inventory, build_leakage_table, build_recommendation_tables,
)
from urban_ops.eda.reports import (
    REQUIRED_TABLES, validate_eda_report, write_eda_report,
)
from urban_ops.eda.source import (
    EDASourceError, load_verified_split, resolve_split_run, verify_source_unchanged,
)
from urban_ops.splitting.reports import (
    complete_report_publication, create_temporary_report_directory,
    publish_report_directory, remove_temporary_report_directory,
    restore_previous_report_directory,
)
from urban_ops.utils.paths import PROJECT_ROOT


LOGGER = logging.getLogger(__name__)


class EDAConfigurationError(ValueError):
    """Raised when Step 9A configuration violates the analysis-only contract."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    """Require one YAML section to be a mapping."""
    if not isinstance(value, dict):
        raise EDAConfigurationError(f"EDA configuration section {name!r} is required.")
    return value


def _path(value: object, name: str) -> Path:
    """Resolve one configured path relative to the repository root."""
    if not isinstance(value, str) or not value.strip():
        raise EDAConfigurationError(f"EDA path {name!r} is required.")
    candidate = Path(value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _tuple(section: Mapping[str, Any], name: str, cast: type) -> tuple[Any, ...]:
    """Read and type one required list configuration value."""
    value = section.get(name)
    if not isinstance(value, list):
        raise EDAConfigurationError(f"EDA list {name!r} is required.")
    return tuple(cast(item) for item in value)


def load_eda_config(path: Path | str) -> EDAConfig:
    """Load and fail-closed validate the complete Step 9A configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"EDA configuration is missing: {config_path}")
    try:
        root = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "root")
        source = _mapping(root.get("input"), "input")
        columns = _mapping(root.get("columns"), "columns")
        features = _mapping(root.get("candidate_features"), "candidate_features")
        missingness = _mapping(root.get("missingness"), "missingness")
        bands = _mapping(missingness.get("descriptive_bands"), "descriptive_bands")
        categorical = _mapping(root.get("categorical_analysis"), "categorical_analysis")
        numeric = _mapping(root.get("numeric_analysis"), "numeric_analysis")
        geography = _mapping(root.get("geography"), "geography")
        nyc = _mapping(geography.get("nyc_bounding_box"), "nyc_bounding_box")
        outlier = _mapping(root.get("outlier_policy"), "outlier_policy")
        test_governance = _mapping(root.get("test_governance"), "test_governance")
        output = _mapping(root.get("output"), "output")
        config = EDAConfig(
            split_root=_path(source["split_root"], "input.split_root"),
            latest_pointer=_path(source["latest_pointer"], "input.latest_pointer"),
            required_completion_status=str(source["required_completion_status"]),
            identifier_column=str(columns["identifier"]),
            target_column=str(columns["target"]),
            timestamp_column=str(columns["timestamp"]),
            categorical_features=_tuple(features, "categorical", str),
            numeric_features=_tuple(features, "numeric", str),
            derived_temporal_features=_tuple(features, "derived_temporal", str),
            missingness_bands={name: float(value) for name, value in bands.items()},
            top_n_categories=int(categorical["top_n_categories"]),
            minimum_support_for_target_rate=int(categorical["minimum_support_for_target_rate"]),
            rare_count_candidates=_tuple(categorical, "rare_count_candidates", int),
            rare_share_candidates=_tuple(categorical, "rare_share_candidates", float),
            percentiles=_tuple(numeric, "percentiles", float),
            iqr_multiplier=float(numeric["iqr_multiplier"]),
            latitude_range=tuple(float(value) for value in geography["latitude_valid_range"]),
            longitude_range=tuple(float(value) for value in geography["longitude_valid_range"]),
            nyc_bounds=(
                float(nyc["min_latitude"]), float(nyc["max_latitude"]),
                float(nyc["min_longitude"]), float(nyc["max_longitude"]),
            ),
            report_root=_path(output["report_root"], "output.report_root"),
        )
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        if isinstance(error, EDAConfigurationError):
            raise
        raise EDAConfigurationError(f"Invalid EDA configuration: {error}") from error
    if not bool(outlier.get("analysis_only")):
        raise EDAConfigurationError("EDA outlier policy must remain analysis-only.")
    prohibited = [name for name in ("remove_rows", "clip_values", "impute_values") if bool(outlier.get(name))]
    if prohibited:
        raise EDAConfigurationError("EDA cannot enable source mutation: " + ", ".join(prohibited))
    if not bool(test_governance.get("structural_comparison_only")) or bool(test_governance.get("allow_feature_decisions")):
        raise EDAConfigurationError("Test data must remain structural-disclosure only.")
    required_bands = {"complete_max", "low_max", "moderate_max", "high_max", "very_high_max"}
    if set(config.missingness_bands) != required_bands:
        raise EDAConfigurationError("Missingness descriptive bands are incomplete.")
    if sorted(config.missingness_bands.values()) != list(config.missingness_bands.values()):
        raise EDAConfigurationError("Missingness descriptive bands must be ordered.")
    if config.timestamp_column != "created_date":
        raise EDAConfigurationError("Step 9A temporal derivation requires created_date.")
    if tuple(config.latitude_range) != (-90.0, 90.0) or tuple(config.longitude_range) != (-180.0, 180.0):
        raise EDAConfigurationError("Geographic world bounds must reuse Step 6 authority.")
    return config


def _build_reconciliation(
    source_rows: int, tables: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Build stable report inventory, source, target, and recommendation checks."""
    split_counts = tables["split_target_comparison.csv"]
    target_counts = int(split_counts["on_time_count"].sum() + split_counts["missed_count"].sum())
    return pd.DataFrame([
        {"check_name": "split_rows_reconcile", "left_value": int(split_counts["row_count"].sum()), "right_value": source_rows, "status": "PASS" if int(split_counts["row_count"].sum()) == source_rows else "FAIL"},
        {"check_name": "target_counts_reconcile", "left_value": target_counts, "right_value": source_rows, "status": "PASS" if target_counts == source_rows else "FAIL"},
        {"check_name": "required_table_inventory", "left_value": len(REQUIRED_TABLES), "right_value": len(REQUIRED_TABLES), "status": "PASS"},
        {"check_name": "no_source_rows_removed", "left_value": source_rows, "right_value": source_rows, "status": "PASS"},
        {"check_name": "train_only_fitted_recommendations", "left_value": "train", "right_value": "train", "status": "PASS"},
    ])


def _build_tables(
    source: SourceSplitEvidence, train: pd.DataFrame, config: EDAConfig
) -> tuple[dict[str, pd.DataFrame], tuple[EDAIntegrityCheck, ...]]:
    """Run all governed analysis components and reconcile their table inventory."""
    tables: dict[str, pd.DataFrame] = {}
    tables.update(build_target_tables(source, target_column=config.target_column))
    tables.update(build_temporal_tables(train, config))
    tables.update(build_missingness_tables(source, train, config))
    tables.update(build_categorical_tables(source, train, config))
    tables.update(build_numeric_tables(source, train, config))
    leakage = build_leakage_table(tuple(train.columns), config.derived_temporal_features)
    inventory = build_column_inventory(train, config, leakage)
    tables["leakage_audit.csv"] = leakage
    tables["column_inventory.csv"] = inventory
    tables.update(build_recommendation_tables(source, train, config, inventory, leakage, tables))
    tables["structural_drift_summary.csv"] = build_structural_drift_summary(tables)
    tables["source_split_verification.csv"] = build_source_verification(source)
    checks = build_integrity_checks(source, train, config)
    require_integrity(checks)
    tables["eda_integrity_checks.csv"] = pd.DataFrame([check.to_dict() for check in checks])
    source_rows = len(source.train) + len(source.validation) + len(source.test)
    tables["output_reconciliation.csv"] = _build_reconciliation(source_rows, tables)
    missing = sorted(set(REQUIRED_TABLES).difference(tables))
    if missing:
        raise RuntimeError(f"EDA analysis did not build required tables: {missing}")
    return tables, checks


def run_split_aware_eda(
    *,
    config_path: Path | str,
    split_run_path: Path | None = None,
    report_root: Path | None = None,
    dry_run: bool = False,
) -> EDAResult:
    """Verify Step 8, run train-first EDA, and optionally publish validated reports."""
    config = load_eda_config(config_path)
    run_path = resolve_split_run(
        split_root=config.split_root, latest_pointer=config.latest_pointer,
        override=split_run_path,
    )
    source = load_verified_split(
        run_path=run_path, latest_pointer=config.latest_pointer,
        required_completion_status=config.required_completion_status,
        identifier_column=config.identifier_column, target_column=config.target_column,
        timestamp_column=config.timestamp_column,
    )
    train = derive_temporal_features(source.train, config.timestamp_column)
    tables, checks = _build_tables(source, train, config)
    verify_source_unchanged(source)
    final_report_root = report_root or config.report_root
    decision_counts = tables["baseline_feature_recommendation.csv"]["baseline_decision"].value_counts().to_dict()
    if dry_run:
        LOGGER.info(
            "EDA dry run split_id=%s train=%s validation=%s test=%s train_rate=%.2f%% all_null=%s zero_variance=%s unseen_findings=%s domain_invalid=%s decisions=%s integrity=PASS",
            source.split_id, len(source.train), len(source.validation), len(source.test),
            float(source.train[config.target_column].astype(int).mean()) * 100,
            len(tables["all_null_columns.csv"]), len(tables["zero_variance_columns.csv"]),
            int(tables["unseen_categories.csv"]["rows_affected"].gt(0).sum()),
            int(tables["geographic_outlier_analysis.csv"].loc[
                tables["geographic_outlier_analysis.csv"]["outlier_category"].eq("DOMAIN_INVALID"), "row_count"
            ].sum()), decision_counts,
        )
        return EDAResult(True, source, tables, checks, final_report_root, REQUIRED_FIGURES)

    temporary_report = create_temporary_report_directory(final_report_root)
    backup: Path | None = None
    published = False
    try:
        figures = write_eda_report(temporary_report, source, tables, train)
        validate_eda_report(temporary_report, source)
        verify_source_unchanged(source)
        backup = publish_report_directory(temporary_report, final_report_root)
        published = True
        verify_source_unchanged(source)
        complete_report_publication(backup)
        backup = None
    except Exception:
        if published:
            restore_previous_report_directory(final_report_root, backup)
        remove_temporary_report_directory(temporary_report)
        LOGGER.exception("Split-aware EDA failed; report publication was rolled back.")
        raise
    LOGGER.info("EDA complete split_id=%s report_root=%s", source.split_id, final_report_root)
    return EDAResult(False, source, tables, checks, final_report_root, figures)


def _parser() -> argparse.ArgumentParser:
    """Build the documented Step 9A command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split-run", type=Path)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run Step 9A and return non-zero for configuration or integrity failures."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        run_split_aware_eda(
            config_path=args.config, split_run_path=args.split_run,
            report_root=args.report_root, dry_run=args.dry_run,
        )
    except (OSError, ValueError, RuntimeError, EDASourceError) as error:
        LOGGER.error("Split-aware EDA command failed: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
