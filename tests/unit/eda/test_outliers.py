"""Tests separating domain-invalid, statistical, and operational outliers."""

from pathlib import Path

import pandas as pd

from urban_ops.eda.pipeline import run_split_aware_eda
from tests.unit.eda.conftest import build_eda_fixture, make_eda_frame


def test_valid_geographic_extremes_are_retained_and_cyclic_iqr_is_not_applied(eda_fixture) -> None:
    tables = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True).tables
    geographic = tables["geographic_outlier_analysis.csv"]
    assert geographic.loc[geographic["outlier_category"].eq("DOMAIN_INVALID"), "row_count"].sum() == 0
    numeric = tables["numeric_outlier_analysis.csv"].set_index("feature_name")
    assert numeric.loc["created_day_of_month", "recommended_action"] == "NOT_APPLICABLE"
    assert numeric.loc["latitude", "recommended_action"] == "RETAIN"


def test_partial_and_impossible_coordinates_are_domain_invalid(tmp_path: Path) -> None:
    frame = make_eda_frame()
    frame.loc[0, "latitude"] = "999"
    frame.loc[1, "longitude"] = pd.NA
    fixture = build_eda_fixture(tmp_path, frame)
    tables = run_split_aware_eda(config_path=fixture.config, dry_run=True).tables
    geographic = tables["geographic_outlier_analysis.csv"]
    assert geographic.loc[geographic["outlier_category"].eq("DOMAIN_INVALID"), "row_count"].sum() >= 2
    assert len(tables["temporal_outlier_analysis.csv"]) == 6
