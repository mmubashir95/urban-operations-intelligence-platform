"""Validate coordinates and ZIP formatting without correcting geographic data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BoundingBox:
    """Broad coordinate bounds used only to flag possible NYC outliers."""

    min_latitude: float
    max_latitude: float
    min_longitude: float
    max_longitude: float


OUTLIER_COLUMNS = ["unique_key", "field", "raw_value", "issue_type", "severity"]


def validate_geography(
    frame: pd.DataFrame,
    *,
    latitude_range: tuple[float, float],
    longitude_range: tuple[float, float],
    nyc_box: BoundingBox,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return geographic metric summary and row-level outlier evidence."""
    total = len(frame)
    keys = frame.get("unique_key", pd.Series(frame.index, index=frame.index))
    lat_raw = frame.get("latitude", pd.Series(pd.NA, index=frame.index))
    lon_raw = frame.get("longitude", pd.Series(pd.NA, index=frame.index))
    lat = pd.to_numeric(lat_raw, errors="coerce")
    lon = pd.to_numeric(lon_raw, errors="coerce")
    lat_invalid_numeric = lat_raw.notna() & lat.isna()
    lon_invalid_numeric = lon_raw.notna() & lon.isna()
    lat_world = lat.notna() & ~lat.between(*latitude_range)
    lon_world = lon.notna() & ~lon.between(*longitude_range)
    valid_pair = lat.notna() & lon.notna() & ~lat_world & ~lon_world
    outside_nyc = valid_pair & ~(
        lat.between(nyc_box.min_latitude, nyc_box.max_latitude)
        & lon.between(nyc_box.min_longitude, nyc_box.max_longitude)
    )
    zip_raw = frame.get("incident_zip", pd.Series(pd.NA, index=frame.index))
    zip_text = zip_raw.astype("string")
    zip_non_digit = zip_raw.notna() & ~zip_text.str.fullmatch(r"\d+", na=False)
    zip_decimal = zip_raw.notna() & zip_text.str.fullmatch(r"\d+\.0+", na=False)
    zip_bad_length = zip_raw.notna() & zip_text.str.fullmatch(r"\d+", na=False) & zip_text.str.len().ne(5)
    borough_raw = frame.get("borough", pd.Series(pd.NA, index=frame.index))
    borough_normalized = borough_raw.astype("string").str.strip().str.casefold()
    borough_unspecified = borough_normalized.isin({"", "unknown", "unspecified", "n/a", "na"})
    metrics = [
        ("borough", "missing", int(borough_raw.isna().sum()), "WARNING"),
        ("borough", "unspecified_or_null_like", int(borough_unspecified.sum()), "WARNING"),
        ("latitude", "missing", int(lat_raw.isna().sum()), "WARNING"),
        ("latitude", "parseable", int(lat.notna().sum()), "INFO"),
        ("latitude", "invalid_numeric", int(lat_invalid_numeric.sum()), "ERROR"),
        ("latitude", "outside_world_range", int(lat_world.sum()), "ERROR"),
        ("longitude", "missing", int(lon_raw.isna().sum()), "WARNING"),
        ("longitude", "parseable", int(lon.notna().sum()), "INFO"),
        ("longitude", "invalid_numeric", int(lon_invalid_numeric.sum()), "ERROR"),
        ("longitude", "outside_world_range", int(lon_world.sum()), "ERROR"),
        ("coordinate_pair", "outside_nyc_bounding_box", int(outside_nyc.sum()), "WARNING"),
        ("incident_zip", "missing", int(zip_raw.isna().sum()), "WARNING"),
        ("incident_zip", "non_digit", int(zip_non_digit.sum()), "WARNING"),
        ("incident_zip", "unexpected_length", int(zip_bad_length.sum()), "WARNING"),
        ("incident_zip", "decimal_like", int(zip_decimal.sum()), "WARNING"),
    ]
    summary = pd.DataFrame([
        {
            "field": field, "metric": metric, "row_count": count,
            "row_rate": count / total if total else 0.0, "severity": severity,
        }
        for field, metric, count, severity in metrics
    ])
    outliers: list[dict[str, object]] = []
    masks = [
        ("latitude", lat_invalid_numeric, "invalid_numeric", "ERROR", lat_raw),
        ("latitude", lat_world, "outside_world_range", "ERROR", lat_raw),
        ("longitude", lon_invalid_numeric, "invalid_numeric", "ERROR", lon_raw),
        ("longitude", lon_world, "outside_world_range", "ERROR", lon_raw),
        ("coordinate_pair", outside_nyc, "outside_nyc_bounding_box", "WARNING",
         lat_raw.astype("string") + "," + lon_raw.astype("string")),
        ("incident_zip", zip_non_digit, "non_digit", "WARNING", zip_raw),
        ("incident_zip", zip_bad_length, "unexpected_length", "WARNING", zip_raw),
        ("incident_zip", zip_decimal, "decimal_like", "WARNING", zip_raw),
        ("borough", borough_unspecified, "unspecified_or_null_like", "WARNING", borough_raw),
    ]
    for field, mask, issue, severity, raw_values in masks:
        for index in frame.index[mask.fillna(False)]:
            outliers.append({
                "unique_key": keys.loc[index], "field": field,
                "raw_value": raw_values.loc[index], "issue_type": issue,
                "severity": severity,
            })
    return summary, pd.DataFrame(outliers, columns=OUTLIER_COLUMNS)
