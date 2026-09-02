"""Creation-time feature governance and target-label construction."""

from urban_ops.features.categorical_encoding import (
    fit_categorical_encoder,
    transform_categorical_encoder,
)
from urban_ops.features.categorical_missing import (
    replace_policy_approved_categorical_missing,
)
from urban_ops.features.eligibility import evaluate_target_eligibility
from urban_ops.features.leakage import validate_feature_columns
from urban_ops.features.policy import load_feature_policy
from urban_ops.features.rare_unseen import (
    fit_rare_unseen_handler,
    transform_rare_unseen,
)
from urban_ops.features.target import build_missed_resolution_target
from urban_ops.features.temporal import derive_approved_temporal_features

__all__ = [
    "build_missed_resolution_target",
    "derive_approved_temporal_features",
    "evaluate_target_eligibility",
    "fit_categorical_encoder",
    "load_feature_policy",
    "replace_policy_approved_categorical_missing",
    "fit_rare_unseen_handler",
    "transform_categorical_encoder",
    "transform_rare_unseen",
    "validate_feature_columns",
]
