"""Tests for complete figure generation and matplotlib resource cleanup."""

import matplotlib.pyplot as plt

from urban_ops.eda.figures import REQUIRED_FIGURES
from urban_ops.eda.pipeline import run_split_aware_eda


def test_full_run_writes_all_figures_and_closes_them(eda_fixture) -> None:
    result = run_split_aware_eda(config_path=eda_fixture.config)
    assert result.figure_names == REQUIRED_FIGURES
    assert all((eda_fixture.reports / "figures" / name).is_file() for name in REQUIRED_FIGURES)
    assert plt.get_fignums() == []
