"""Tests for EDA configuration, dry/full orchestration, rollback, CLI, and safety."""

from pathlib import Path

import pytest
import yaml

import urban_ops.eda.pipeline as pipeline_module
from urban_ops.eda.pipeline import EDAConfigurationError, main, run_split_aware_eda


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    } if root.exists() else {}


def test_dry_run_writes_no_reports_and_full_run_preserves_source(eda_fixture) -> None:
    source_before = _tree_bytes(eda_fixture.split_run)
    dry = run_split_aware_eda(config_path=eda_fixture.config, dry_run=True)
    assert dry.dry_run and not eda_fixture.reports.exists()
    full = run_split_aware_eda(config_path=eda_fixture.config)
    assert not full.dry_run and (eda_fixture.reports / "eda_summary.md").is_file()
    assert _tree_bytes(eda_fixture.split_run) == source_before
    assert not list(eda_fixture.reports.parent.glob(f".{eda_fixture.reports.name}.tmp-*"))


@pytest.mark.parametrize("setting", ["remove_rows", "clip_values", "impute_values"])
def test_configuration_rejects_source_mutation(eda_fixture, setting: str) -> None:
    payload = yaml.safe_load(eda_fixture.config.read_text())
    payload["outlier_policy"][setting] = True
    eda_fixture.config.write_text(yaml.safe_dump(payload))
    with pytest.raises(EDAConfigurationError, match="source mutation"):
        run_split_aware_eda(config_path=eda_fixture.config, dry_run=True)


def test_configuration_rejects_test_feature_decisions(eda_fixture) -> None:
    payload = yaml.safe_load(eda_fixture.config.read_text())
    payload["test_governance"]["allow_feature_decisions"] = True
    eda_fixture.config.write_text(yaml.safe_dump(payload))
    with pytest.raises(EDAConfigurationError, match="structural-disclosure"):
        run_split_aware_eda(config_path=eda_fixture.config, dry_run=True)


def test_late_source_recheck_failure_preserves_previous_reports(
    eda_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_split_aware_eda(config_path=eda_fixture.config)
    previous = _tree_bytes(eda_fixture.reports)
    calls = 0
    original = pipeline_module.verify_source_unchanged

    def fail_after_report_generation(source) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("forced late source failure")
        original(source)

    monkeypatch.setattr(pipeline_module, "verify_source_unchanged", fail_after_report_generation)
    with pytest.raises(RuntimeError, match="late source"):
        run_split_aware_eda(config_path=eda_fixture.config)
    assert _tree_bytes(eda_fixture.reports) == previous
    assert not list(eda_fixture.reports.parent.glob(f".{eda_fixture.reports.name}.tmp-*"))


def test_report_publication_failure_restores_previous_reports(
    eda_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_split_aware_eda(config_path=eda_fixture.config)
    previous = _tree_bytes(eda_fixture.reports)
    original_replace = Path.replace

    def fail_staged_replace(self: Path, target: Path) -> Path:
        if self.name.startswith(f".{eda_fixture.reports.name}.tmp-") and Path(target) == eda_fixture.reports:
            raise OSError("forced report publication failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_staged_replace)
    with pytest.raises(OSError, match="report publication"):
        run_split_aware_eda(config_path=eda_fixture.config)
    assert _tree_bytes(eda_fixture.reports) == previous


def test_no_network_preprocessing_or_model_artifacts_and_cli_codes(
    eda_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def network_forbidden(*args, **kwargs):
        raise AssertionError("EDA must not call the network")
    monkeypatch.setattr("urllib.request.urlopen", network_forbidden)
    assert main(["--config", str(eda_fixture.config), "--dry-run"]) == 0
    assert main(["--config", str(eda_fixture.config.parent / "missing.yaml"), "--dry-run"]) == 1
    run_split_aware_eda(config_path=eda_fixture.config, dry_run=True)
    forbidden = ["model", "preprocessor", "feature_matrix"]
    assert not any(any(token in path.name.casefold() for token in forbidden) for path in eda_fixture.config.parent.rglob("*"))
