"""Unit tests for deterministic Phase 6 error-analysis behavior."""
from math import ceil

import numpy as np
import pandas as pd
import pytest

from urban_ops.models.advanced_error_analysis import (
    aggregate_zip, canonical_error_table, classify_errors, confusion_summary, failure_modes,
    group_metrics, load_error_analysis_config, score_range_metrics,
    select_severity_cases, top_k_predictions,
)


def test_error_classification_and_confusion_are_exact():
    actual = [1, 0, 0, 1]; predicted = [1, 0, 1, 0]
    assert classify_errors(actual, predicted).tolist() == ["TP", "TN", "FP", "FN"]
    row = confusion_summary(actual, predicted, decision_rule="unit").iloc[0]
    assert row[["TP", "TN", "FP", "FN"]].tolist() == [1, 1, 1, 1]
    assert row.sample_count == 4 and row.recall == row.false_positive_rate == row.false_negative_rate == .5


def test_top_k_uses_ceiling_and_stable_original_order_for_ties():
    predicted, ranks = top_k_predictions([.8, .9, .8, .1], .26)
    assert predicted.sum() == ceil(4 * .26) == 2
    assert ranks.tolist() == [2, 1, 3, 4]
    assert predicted.tolist() == [1, 1, 0, 0]


def _canonical_fixture():
    config = load_error_analysis_config()
    config = config.__class__(**{**config.__dict__, "capacity": .25, "score_quantiles": 4,
                                "minimum_sample_count": 3, "minimum_positive_count": 2})
    frame = pd.DataFrame({"created_date": pd.date_range("2025-01-01", periods=8, freq="D", tz="UTC"),
                          "agency": ["A"] * 8, "complaint_type": ["C"] * 8,
                          "borough": ["X"] * 4 + ["Y"] * 4,
                          "incident_zip": ["1", "1", "2", "2", "3", None, "4", "5"],
                          "location_type": ["L"] * 8})
    table = canonical_error_table(frame, identifiers=[f"id-{i}" for i in range(8)], targets=[0, 1, 1, 0, 1, 0, 1, 1],
                                  raw_scores=[.9, .8, .7, .6, .5, .4, .3, .2], calibrated_scores=[.5] * 8, config=config)
    return config, table


def test_canonical_table_score_bins_and_group_support_are_deterministic():
    config, table = _canonical_fixture()
    assert table.operational_selection.tolist() == [1, 1, 0, 0, 0, 0, 0, 0]
    assert table.primary_error_type.tolist() == ["FP", "TP", "FN", "TN", "FN", "TN", "FN", "FN"]
    assert table.score_range.nunique() == 4
    ranges = score_range_metrics(table)
    assert ranges.sample_count.sum() == len(table) and ranges[["TP", "TN", "FP", "FN"]].sum().sum() == len(table)
    groups = group_metrics(table, "borough", minimum_sample_count=config.minimum_sample_count,
                           minimum_positive_count=config.minimum_positive_count)
    assert groups.support_status.tolist() == ["SUPPORTED", "SUPPORTED"]
    assert groups.sample_count.sum() == len(table)


def test_low_support_missing_metadata_and_zip_remainder_are_clean():
    _, table = _canonical_fixture()
    groups = group_metrics(table, "not_present", minimum_sample_count=2, minimum_positive_count=1)
    assert groups.empty
    borough = group_metrics(table, "borough", minimum_sample_count=5, minimum_positive_count=3)
    assert set(borough.support_status) == {"LOW SUPPORT"}
    zips = aggregate_zip(table, 2)
    assert "MISSING" in set(zips) and "OTHER" in set(zips)
    assert list(zips).count("1") == 2


def test_error_severity_selection_is_deterministic():
    config, table = _canonical_fixture()
    severity = select_severity_cases(table, config)
    assert set(severity.error_severity) >= {"HIGH_CONFIDENCE_FP", "BORDERLINE_FN", "SEVERE_FN"}
    assert severity.loc[severity.record_identifier.eq("id-0"), "error_severity"].item() == "HIGH_CONFIDENCE_FP"


def test_failure_mode_generation_is_deterministic_and_evidence_based():
    config, table = _canonical_fixture()
    severity = select_severity_cases(table, config)
    table = table.merge(severity[["record_identifier", "error_severity"]], on="record_identifier", how="left")
    table["error_severity"] = table.error_severity.fillna("NOT_ERROR")
    table.loc[:3, "created_period"] = "2025-01"; table.loc[4:, "created_period"] = "2025-02"
    geography = group_metrics(table, "borough", minimum_sample_count=3, minimum_positive_count=2).rename(columns={"borough": "group"})
    geography.insert(0, "dimension", "borough")
    first = failure_modes(table, geography)
    assert first.equals(failure_modes(table, geography))
    assert "Temporal selection concentration" in set(first.failure_mode)
    assert set(first.severity) <= {"HIGH", "MEDIUM", "LOW"}


@pytest.mark.parametrize("actual,predicted", [([0, 1], [0]), ([0, 2], [0, 1]), ([], [])])
def test_invalid_classification_inputs_fail(actual, predicted):
    with pytest.raises(ValueError):
        classify_errors(actual, predicted)
