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
from urban_ops.features.numeric_preprocessing import (
    active_numeric_feature_names,
    build_numeric_preprocessing_evidence,
    fit_numeric_preprocessor,
    load_numeric_preprocessing_config,
    transform_split_numeric_preprocessor,
    transform_numeric_preprocessor,
)
from urban_ops.features.policy import load_feature_policy
from urban_ops.features.preprocessing_composition import (
    build_preprocessing_composition,
    build_preprocessing_composition_evidence,
    compose_preprocessing_blocks,
    compose_split_preprocessing_blocks,
    load_preprocessing_composition_config,
)
from urban_ops.features.rare_unseen import (
    fit_rare_unseen_handler,
    transform_rare_unseen,
)
from urban_ops.features.target import build_missed_resolution_target
from urban_ops.features.temporal import derive_approved_temporal_features

__all__ = [
    "build_missed_resolution_target",
    "active_numeric_feature_names",
    "build_numeric_preprocessing_evidence",
    "build_preprocessing_composition",
    "build_preprocessing_composition_evidence",
    "compose_preprocessing_blocks",
    "compose_split_preprocessing_blocks",
    "derive_approved_temporal_features",
    "evaluate_target_eligibility",
    "fit_categorical_encoder",
    "fit_numeric_preprocessor",
    "load_feature_policy",
    "load_numeric_preprocessing_config",
    "load_preprocessing_composition_config",
    "replace_policy_approved_categorical_missing",
    "fit_rare_unseen_handler",
    "transform_categorical_encoder",
    "transform_numeric_preprocessor",
    "transform_split_numeric_preprocessor",
    "transform_rare_unseen",
    "validate_feature_columns",
]
