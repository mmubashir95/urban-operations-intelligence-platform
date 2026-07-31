"""Tests for nullable missed-resolution labels."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from urban_ops.features.target import build_missed_resolution_target


def test_target_boundaries_na_dtype_and_no_mutation() -> None:
    frame = pd.DataFrame({
        "due_date": ["2024-01-10"] * 4,
        "closed_date": ["2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"],
        "target_eligible": [True, True, True, False],
    })
    original = frame.copy(deep=True)
    target = build_missed_resolution_target(frame)
    assert target.iloc[:3].tolist() == [0, 0, 1]
    assert pd.isna(target.iloc[3])
    assert str(target.dtype) == "Int8"
    assert set(target.dropna().astype(int)) <= {0, 1}
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("column", ["closed_date", "due_date"])
def test_invalid_eligible_target_input_raises(column: str) -> None:
    frame = pd.DataFrame({
        "due_date": ["2024-01-10"], "closed_date": ["2024-01-11"],
        "target_eligible": [True],
    })
    frame.loc[0, column] = None
    with pytest.raises(ValueError, match="Eligible rows require parseable"):
        build_missed_resolution_target(frame)
