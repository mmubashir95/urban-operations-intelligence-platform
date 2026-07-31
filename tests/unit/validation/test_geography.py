"""Tests for coordinate range and ZIP string validation."""

import pandas as pd
import pytest

from urban_ops.validation.geography import BoundingBox, validate_geography


BOX = BoundingBox(40.40, 41.00, -74.30, -73.60)


def validate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return validate_geography(
        frame, latitude_range=(-90, 90), longitude_range=(-180, 180), nyc_box=BOX
    )


@pytest.mark.parametrize(("column", "value", "metric"), [
    ("latitude", "-91", "outside_world_range"),
    ("latitude", "91", "outside_world_range"),
    ("longitude", "-181", "outside_world_range"),
    ("longitude", "181", "outside_world_range"),
    ("latitude", "not-numeric", "invalid_numeric"),
    ("longitude", "not-numeric", "invalid_numeric"),
])
def test_invalid_coordinate_is_reported(column: str, value: str, metric: str) -> None:
    row = {"unique_key": "1", "latitude": "40.7", "longitude": "-73.9", "incident_zip": "01234"}
    row[column] = value
    summary, outliers = validate(pd.DataFrame([row]))
    count = summary.set_index(["field", "metric"]).loc[(column, metric), "row_count"]
    assert count == 1
    assert metric in set(outliers["issue_type"])


def test_valid_coordinate_and_leading_zero_zip_are_preserved() -> None:
    frame = pd.DataFrame([{
        "unique_key": "1", "latitude": "40.75", "longitude": "-73.90",
        "incident_zip": "01234",
    }])
    original = frame.copy(deep=True)
    summary, outliers = validate(frame)
    assert outliers.empty
    assert summary.loc[summary["metric"].eq("unexpected_length"), "row_count"].iloc[0] == 0
    pd.testing.assert_frame_equal(frame, original)


def test_missing_coordinates_and_zip_are_counted() -> None:
    summary, _ = validate(pd.DataFrame([{
        "unique_key": "1", "latitude": None, "longitude": None, "incident_zip": None,
    }]))
    indexed = summary.set_index(["field", "metric"])["row_count"]
    assert indexed[("latitude", "missing")] == 1
    assert indexed[("longitude", "missing")] == 1
    assert indexed[("incident_zip", "missing")] == 1


def test_unspecified_borough_is_reported_without_correction() -> None:
    frame = pd.DataFrame([{
        "unique_key": "1", "borough": "Unspecified", "latitude": "40.7",
        "longitude": "-73.9", "incident_zip": "12345",
    }])
    original = frame.copy(deep=True)
    summary, outliers = validate(frame)
    assert summary.set_index(["field", "metric"]).loc[
        ("borough", "unspecified_or_null_like"), "row_count"
    ] == 1
    assert "unspecified_or_null_like" in set(outliers["issue_type"])
    pd.testing.assert_frame_equal(frame, original)


def test_point_outside_nyc_box_is_warning_not_world_error() -> None:
    summary, outliers = validate(pd.DataFrame([{
        "unique_key": "1", "latitude": "35", "longitude": "-100", "incident_zip": "12345",
    }]))
    indexed = summary.set_index(["field", "metric"])["row_count"]
    assert indexed[("coordinate_pair", "outside_nyc_bounding_box")] == 1
    assert indexed[("latitude", "outside_world_range")] == 0
    assert outliers.iloc[0].severity == "WARNING"


@pytest.mark.parametrize(("zip_value", "issue"), [
    ("1234", "unexpected_length"),
    ("12345.0", "decimal_like"),
    ("12A45", "non_digit"),
])
def test_zip_format_issues_are_reported(zip_value: str, issue: str) -> None:
    _, outliers = validate(pd.DataFrame([{
        "unique_key": "1", "latitude": "40.7", "longitude": "-73.9",
        "incident_zip": zip_value,
    }]))
    assert issue in set(outliers["issue_type"])
