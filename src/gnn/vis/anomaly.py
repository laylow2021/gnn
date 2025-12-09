"""Visualization helpers for anomaly scores."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None  # type: ignore


def plot_anomaly_distributions(df_scored: pd.DataFrame, score_col: str, counterparty_col: str, top_n: int = 10):
    """Plot histogram of anomaly scores and top-N barplot by counterparty."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(df_scored[score_col], bins=20, ax=axes[0])
    axes[0].set_title("Anomaly score distribution")
    axes[0].set_xlabel(score_col)

    top = df_scored.nlargest(top_n, score_col)
    sns.barplot(x=top[score_col], y=top[counterparty_col], ax=axes[1])
    axes[1].set_title(f"Top {top_n} anomalies by counterparty")
    axes[1].set_xlabel(score_col)
    axes[1].set_ylabel(counterparty_col)

    plt.tight_layout()
    return fig, axes


def shap_bar_for_top(
    model,
    features: pd.DataFrame,
    top_indices: pd.Index,
    max_display: int = 10,
    random_state: Optional[int] = None,
):
    """Plot SHAP bar values for the selected indices if shap is available."""
    if shap is None:
        raise ImportError("shap is not installed; install shap to use SHAP plots.")
    explainer = shap.Explainer(model)
    shap_values = explainer(features.loc[top_indices])
    shap.plots.bar(shap_values, max_display=max_display)
    return shap_values
