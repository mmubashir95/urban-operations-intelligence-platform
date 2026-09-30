"""Train the initial XGBoost risk model on the frozen training split only.

This module performs no prediction, evaluation, tuning, persistence, or model
selection. Phase 2.1 remains responsible for validating all frozen inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from urban_ops.models.baseline_workflow import EDA_CONFIG_PATH, FrozenBaselineInputs
from urban_ops.models.gradient_boosting import GradientBoostedRiskModel
from urban_ops.models.gradient_boosting_inputs import (
    load_and_verify_gradient_boosting_inputs,
)


@dataclass(frozen=True)
class GradientBoostingTrainingMetadata:
    """Descriptive, non-evaluative metadata for one train-only fit."""

    implementation: str
    model_class: str
    training_split: str
    split_id: str
    training_row_count: int
    feature_count: int
    feature_names: tuple[str, ...]
    negative_class_count: int
    positive_class_count: int
    positive_class_rate: float
    configuration_version: int
    random_state: int

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe descriptive training metadata."""
        payload = asdict(self)
        payload["feature_names"] = list(self.feature_names)
        return payload


@dataclass(frozen=True)
class GradientBoostingTrainingResult:
    """Fitted in-memory wrapper and its train-only descriptive metadata."""

    model: GradientBoostedRiskModel
    metadata: GradientBoostingTrainingMetadata


def train_gradient_boosted_risk_model(
    inputs: FrozenBaselineInputs,
) -> GradientBoostingTrainingResult:
    """Fit the approved model using only the verified frozen train split."""
    if "train" not in inputs.matrices:
        raise ValueError("Frozen Gradient Boosting inputs are missing the train matrix.")
    if "train" not in inputs.targets:
        raise ValueError("Frozen Gradient Boosting inputs are missing the train target.")

    X_train = inputs.matrices["train"]
    y_train = inputs.targets["train"]
    feature_names = inputs.feature_names

    model = GradientBoostedRiskModel()
    model.fit(X_train, y_train, feature_names=feature_names)
    if model.feature_names_ != feature_names:
        raise ValueError("Fitted model did not preserve the frozen feature-name order.")
    if model.feature_count_ != len(feature_names):
        raise ValueError("Fitted model feature count differs from the frozen contract.")

    negative_count = int(y_train.eq(0).sum())
    positive_count = int(y_train.eq(1).sum())
    training_row_count = len(y_train)
    metadata = GradientBoostingTrainingMetadata(
        implementation="XGBoost",
        model_class=type(model).__name__,
        training_split="train",
        split_id=inputs.split_id,
        training_row_count=training_row_count,
        feature_count=len(feature_names),
        feature_names=feature_names,
        negative_class_count=negative_count,
        positive_class_count=positive_count,
        positive_class_rate=float(positive_count / training_row_count),
        configuration_version=model.config.selection_version,
        random_state=model.config.random_state,
    )
    return GradientBoostingTrainingResult(model=model, metadata=metadata)


def load_and_train_gradient_boosted_risk_model(
    *,
    eda_config_path: Path | str = EDA_CONFIG_PATH,
    split_run_path: Path | None = None,
) -> GradientBoostingTrainingResult:
    """Verify frozen Month 1 inputs, then fit exclusively on their train split."""
    inputs = load_and_verify_gradient_boosting_inputs(
        eda_config_path=eda_config_path,
        split_run_path=split_run_path,
    )
    return train_gradient_boosted_risk_model(inputs)
