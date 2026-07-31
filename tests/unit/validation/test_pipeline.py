"""Unit tests for validation configuration, artifact guards, and CLI errors."""

from pathlib import Path

import pytest
import yaml

from urban_ops.validation.pipeline import (
    ValidationConfigurationError, ValidationInputError, load_validation_config,
    main, run_validation,
)


def config_payload(tmp_path: Path) -> dict:
    """Return a minimal valid Step 6 configuration mapping."""
    return {
        "input": {
            "raw_root": str(tmp_path / "raw"), "expected_dataset_id": "erm2-nwe9",
            "supported_format": "parquet", "scope_authority_file": str(tmp_path / "scope.csv"),
            "scope_extraction_metadata_file": str(tmp_path / "scope_metadata.csv"),
        },
        "output": {"report_root": str(tmp_path / "reports")},
        "schema": {"require_column_order": True, "required_columns": ["unique_key"]},
        "timestamps": {"columns": ["created_date"], "timezone": "UTC"},
        "categories": {
            "columns": ["status"], "rare_category_max_rows": 1,
            "high_cardinality_threshold": 100, "maximum_profile_values_per_column": 100,
        },
        "geography": {
            "latitude_valid_range": [-90, 90], "longitude_valid_range": [-180, 180],
            "nyc_bounding_box": {
                "min_latitude": 40.4, "max_latitude": 41.0,
                "min_longitude": -74.3, "max_longitude": -73.6,
            },
        },
        "missingness": {"null_like_strings": ["", "null"]},
    }


def write_config(tmp_path: Path, payload: dict | None = None) -> Path:
    path = tmp_path / "validation.yaml"
    path.write_text(yaml.safe_dump(payload or config_payload(tmp_path)), encoding="utf-8")
    return path


def test_valid_config_loads_typed_paths(tmp_path: Path) -> None:
    config = load_validation_config(write_config(tmp_path))
    assert config.raw_root == tmp_path / "raw"
    assert config.supported_format == "parquet"
    assert config.nyc_box.min_latitude == 40.4


def test_missing_config_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="configuration is missing"):
        load_validation_config(tmp_path / "missing.yaml")


def test_unsupported_format_is_rejected(tmp_path: Path) -> None:
    payload = config_payload(tmp_path)
    payload["input"]["supported_format"] = "csv"
    with pytest.raises(ValidationConfigurationError, match="Parquet"):
        load_validation_config(write_config(tmp_path, payload))


def test_missing_raw_root_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No successful raw extraction"):
        run_validation(config_path=write_config(tmp_path))


@pytest.mark.parametrize("present", [(), ("metadata.json",), ("metadata.json", "query.sql")])
def test_incomplete_explicit_run_fails(tmp_path: Path, present: tuple[str, ...]) -> None:
    run = tmp_path / "run_id=test"
    run.mkdir()
    for name in present:
        (run / name).write_text("{}" if name.endswith("json") else "query", encoding="utf-8")
    with pytest.raises(ValidationInputError, match="missing|Unable to read"):
        run_validation(config_path=write_config(tmp_path), raw_run_path=run)


def test_cli_returns_two_for_input_failure(tmp_path: Path) -> None:
    assert main(["--config", str(write_config(tmp_path))]) == 2
