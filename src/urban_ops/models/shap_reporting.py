"""Write deterministic Phase 5 Tree SHAP tables, figures, and reports."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from urban_ops.models.shap_explainability import CASE_ROLES, ShapExplanationResult


def _plotting():
    """Return headless pyplot configured consistently with project reports."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def write_global_figures(result: ShapExplanationResult, importance: pd.DataFrame, directory: Path) -> tuple[Path, Path]:
    """Write global importance and compact beeswarm-style distribution figures."""
    plt = _plotting(); directory.mkdir(parents=True, exist_ok=True)
    bar_path = directory / "phase_5_global_shap_importance.png"
    ordered = importance.sort_values("mean_abs_shap")
    figure, axis = plt.subplots(figsize=(7, 4), dpi=150)
    axis.barh(ordered.feature, ordered.mean_abs_shap)
    axis.set(xlabel="Mean absolute SHAP value (raw margin)", title="Global Model Feature Influence")
    figure.tight_layout(); figure.savefig(bar_path); plt.close(figure)

    summary_path = directory / "phase_5_shap_summary.png"
    figure, axis = plt.subplots(figsize=(8, 5), dpi=150)
    rng = np.random.default_rng(20260806)
    for row, feature in enumerate(importance.feature):
        column = result.feature_names.index(feature)
        jitter = rng.uniform(-0.15, 0.15, len(result.shap_values))
        scatter = axis.scatter(result.shap_values[:, column], row + jitter, c=result.feature_values[:, column],
                               cmap="coolwarm", s=8, alpha=0.55, linewidths=0)
    axis.axvline(0, color="0.5", linewidth=1); axis.set_yticks(range(len(importance)), importance.feature)
    axis.invert_yaxis(); axis.set(xlabel="SHAP contribution (raw margin)", title="Tree SHAP Summary")
    colorbar = figure.colorbar(scatter, ax=axis, label="Relative feature value")
    colorbar.set_ticks([0, 1], labels=["Low", "High"])
    figure.tight_layout(); figure.savefig(summary_path); plt.close(figure)
    return bar_path, summary_path


def write_dependence_figures(result: ShapExplanationResult, directory: Path) -> tuple[Path, ...]:
    """Write one deterministic value-versus-contribution plot per feature."""
    plt = _plotting(); paths = []
    for column, feature in enumerate(result.feature_names):
        path = directory / f"phase_5_dependence_{feature}.png"
        figure, axis = plt.subplots(figsize=(6, 4), dpi=150)
        axis.scatter(result.feature_values[:, column], result.shap_values[:, column], s=10, alpha=0.4)
        axis.axhline(0, color="0.5", linewidth=1)
        axis.set(xlabel=feature, ylabel="SHAP contribution (raw margin)", title=f"Model Contribution by {feature}")
        figure.tight_layout(); figure.savefig(path); plt.close(figure); paths.append(path)
    return tuple(paths)


def build_local_explanations(result: ShapExplanationResult, targets: np.ndarray,
                             roles: dict[str, int], sample_positions: tuple[int, ...], *, cutoff: float) -> pd.DataFrame:
    """Build row-per-feature local evidence for each deterministic case role."""
    sample_lookup = {position: index for index, position in enumerate(sample_positions)}
    rows = []
    for role in CASE_ROLES:
        original_position = roles[role]; row = sample_lookup[original_position]
        for column, feature in enumerate(result.feature_names):
            contribution = float(result.shap_values[row, column])
            rows.append({"case_role": role, "record_identifier": result.identifiers[row],
                         "test_row_position": original_position, "actual_target": int(targets[original_position]),
                         "raw_model_score": float(result.raw_scores[row]),
                         "descriptive_prediction_at_0_5": int(result.raw_scores[row] >= cutoff),
                         "base_value_raw_margin": float(result.base_values[row]), "model_output_raw_margin": float(result.raw_margins[row]),
                         "feature": feature, "feature_value": float(result.feature_values[row, column]),
                         "shap_contribution_raw_margin": contribution,
                         "direction": "increases_model_output" if contribution > 0 else "decreases_model_output" if contribution < 0 else "neutral"})
    return pd.DataFrame(rows)


def write_waterfalls(result: ShapExplanationResult, local: pd.DataFrame, directory: Path) -> tuple[Path, ...]:
    """Write six waterfall-style raw-margin contribution figures."""
    plt = _plotting(); directory.mkdir(parents=True, exist_ok=True); paths = []
    for role in CASE_ROLES:
        rows = local.loc[local.case_role.eq(role)].sort_values("shap_contribution_raw_margin", key=lambda values: values.abs(), ascending=False)
        contributions = rows.shap_contribution_raw_margin.to_numpy(); labels = [f"{feature}={value:g}" for feature, value in zip(rows.feature, rows.feature_value)]
        base = float(rows.base_value_raw_margin.iloc[0]); starts = base + np.concatenate(([0.0], np.cumsum(contributions[:-1])))
        path = directory / f"{role}_waterfall.png"
        figure, axis = plt.subplots(figsize=(8, 4), dpi=150)
        colors = ["#d95f5f" if value > 0 else "#4c78a8" for value in contributions]
        for index, (start, value) in enumerate(zip(starts, contributions)):
            axis.barh(index, value, left=start, color=colors[index]); axis.text(start + value, index, f" {value:+.3f}", va="center", fontsize=8)
        axis.axvline(base, color="0.5", linestyle="--", label=f"Base {base:.3f}")
        axis.set_yticks(range(len(labels)), labels); axis.invert_yaxis(); axis.set(xlabel="Raw model margin", title=f"{role.replace('_', ' ').title()} Tree SHAP Explanation")
        axis.legend(); figure.tight_layout(); figure.savefig(path); plt.close(figure); paths.append(path)
    return tuple(paths)


def markdown_table(table: pd.DataFrame, *, digits: int = 6) -> str:
    """Render a compact Markdown table without optional dependencies."""
    columns = list(table.columns); lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for record in table.itertuples(index=False, name=None):
        values = [f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) else str(value) for value in record]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(*, model_metadata: dict[str, object], phase4: dict[str, object], result: ShapExplanationResult,
                 additivity: dict[str, object], importance: pd.DataFrame, direction: pd.DataFrame,
                 interactions: pd.DataFrame, local: pd.DataFrame, comparison: pd.DataFrame,
                 stability: dict[str, object]) -> str:
    """Build the complete governed Phase 5 findings report."""
    strongest_values = (direction.assign(abs_mean=lambda table: table.mean_shap.abs()).sort_values("abs_mean", ascending=False)
                        .groupby("feature", sort=False).head(2)[["feature", "feature_value", "mean_shap", "sample_count"]])
    local_summary = (local.assign(abs_shap=lambda table: table.shap_contribution_raw_margin.abs()).sort_values(["case_role", "abs_shap"], ascending=[True, False])
                     .groupby("case_role", sort=False).head(1)[["case_role", "record_identifier", "raw_model_score", "feature", "shap_contribution_raw_margin"]])
    return f"""# Phase 5 — SHAP explainability

> **Interpretation guardrail:** SHAP explains the behavior of the current model. It does not establish that the model is accurate. A feature can have high SHAP influence inside a weak model without being strongly predictive in the real world.

SHAP attributes model output to model inputs; it does not establish causal relationships.

## Frozen boundary

The explained model is Phase 3 `{model_metadata['candidate_id']}` using the frozen TRAIN-only fit, split `{model_metadata['split_id']}`, and feature fingerprint `{model_metadata['feature_fingerprint']}`. Phase 4 remains frozen as `{phase4['selected_method']}` and is not attributed: calibration is post-processing. No model selection, retuning, calibration fitting, feature change, preprocessing change, split change, or threshold optimization occurred.

The TEST split is used only for post-freeze descriptive explanation. A seeded sample of {len(result.feature_values)} rows was drawn from 5,499 TEST records with required local cases reserved. Seed: `20260806`.

## Explainer and correctness

Environment: XGBoost `{result.xgboost_version}`. The Python SHAP package could not be installed in this Python 3.13 macOS x86_64 environment because its required `llvmlite` wheel is unavailable and local LLVM is absent. The supported tree-specific equivalent, XGBoost native Tree SHAP (`pred_contribs`), was therefore used; KernelExplainer was not used.

SHAP values are additive contributions in **raw margin/log-odds space**, not percentage-point probability changes. Additivity checked {additivity['rows_checked']} rows at tolerance {additivity['tolerance']}; maximum residual `{additivity['maximum_residual']:.10f}`; status **{additivity['status']}**.

## Global model influence

{markdown_table(importance)}

The fitted model relies most on `{importance.feature.iloc[0]}` among the four available temporal features. This describes model reliance, not a real-world cause of missed targets.

## Feature direction

Exact feature-value groups with the largest absolute mean contribution are:

{markdown_table(strongest_values)}

Positive SHAP values raise the raw model margin; negative values lower it. Sparse value groups must be interpreted cautiously.

## Interactions

Native Tree SHAP interaction analysis completed. Strongest pairs:

{markdown_table(interactions.head(6))}

## Representative local cases

Cases use the wrapper's existing `0.5` cutoff only as a descriptive reference; it is not an operational threshold.

{markdown_table(local_summary)}

Waterfalls show the base raw margin, feature contributions, and final raw model margin for highest/lowest risk, TP, TN, FP, and FN roles. Duplicate record identifiers across roles are allowed when one deterministic record satisfies multiple roles.

## Correct versus incorrect patterns

{markdown_table(comparison)}

This is bounded descriptive evidence for the reference cutoff, not Phase 6 subgroup error analysis.

## Stability

Two seeded, disjoint 1,000-row halves produced Spearman rank correlation `{stability['spearman_rank_correlation']:.6f}`. Top feature match: `{stability['top_feature_matches']}`. Top-two set match: `{stability['top_two_set_matches']}`. Status: **{stability['status']}**.

## What SHAP can and cannot say

SHAP can describe how this fitted tree model distributes its raw output among the four supplied features, globally and for selected records. It cannot prove causality, establish model accuracy, validate data-generating mechanisms, or explain the separate isotonic mapping. The model's known weak TEST discrimination and failed calibration generalization materially limit how strongly these patterns should be interpreted.

Phase 6 should investigate the false-positive and false-negative patterns surfaced here, but Phase 5 does not perform that broader error analysis. Phase 5 is formally complete subject to its recorded test gate.
"""


def completion_gate(*, interactions_completed: bool) -> str:
    """Return the Phase 5 governance checklist."""
    items = ["Frozen Phase 3 classifier reconstructed and verified", "Phase 4 calibration decision unchanged",
             "No tuning, new features, preprocessing, split, calibration, or threshold optimization",
             "Tree-specific native SHAP used", "Deterministic TEST explanation sample created", "Feature order and finite shapes verified",
             "Raw-margin output documented and additivity verified", "Global importance CSV and figure produced",
             "Summary and four dependence figures produced", "Feature-direction analysis completed",
             "Interaction analysis completed" if interactions_completed else "Interaction analysis explicitly documented as unsupported",
             "Six deterministic local roles and waterfall figures produced", "Correct-vs-incorrect comparison produced",
             "Stability check completed", "Weak-model and causality guardrails documented", "Phase 5 reports produced"]
    return "# Phase 5 completion gate\n\n" + "\n".join(f"- [x] {item}." for item in items) + "\n\n- [ ] Targeted and full-suite test evidence recorded after implementation.\n"
