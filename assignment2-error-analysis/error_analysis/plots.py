"""Figures for the error-analysis chapters.

The visual style is imported from Assignment 1 rather than redefined, so both
notebooks share one palette, grid treatment and typography.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from .data_setup import ASSIGNMENT1_DIR  # noqa: F401  (ensures A1 is on the path)
from src.plotting import (  # noqa: E402
    CATEGORICAL_COLOURS,
    DIVERGING_POLES,
    GRID_COLOUR,
    INK_SECONDARY,
    PRIMARY_COLOUR,
    apply_house_style,
)

__all__ = [
    "apply_house_style",
    # chapter 2
    "plot_residuals_vs_predicted",
    "plot_residual_distribution",
    "plot_feature_error_grid",
    "plot_error_by_subgroup",
    "plot_extreme_errors",
    # chapter 3
    "plot_tree_depth_curve",
    "plot_metric_comparison",
    "plot_fold_stability",
    "plot_predicted_vs_actual",
    "plot_explanation_comparison",
]

# Matplotlib cannot resolve the "semibold" weight for the default font and
# falls back to bold, which is fine, but the notice clutters every figure.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

ACCENT_COLOUR = DIVERGING_POLES[1]


def _binned_summary(x: pd.Series, y: pd.Series, bins: int = 12) -> pd.DataFrame:
    """Average y within equal-count bins of x, for overlaying on a scatter."""
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    frame["bin"] = pd.qcut(frame["x"], bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True)
    return pd.DataFrame(
        {
            "centre": grouped["x"].median(),
            "mean": grouped["y"].mean(),
            "sd": grouped["y"].std(),
        }
    ).reset_index(drop=True)


def plot_residuals_vs_predicted(frame: pd.DataFrame) -> plt.Figure:
    """Residuals against predicted values, with a binned spread band.

    The band shows the standard deviation of the residuals within each bin of
    the prediction, which is what makes non-constant variance visible rather
    than merely arguable.
    """
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.scatter(
        frame["predicted"], frame["residual"],
        s=12, alpha=0.4, color=PRIMARY_COLOUR, edgecolor="none",
    )
    axis.axhline(0, color=INK_SECONDARY, linewidth=1.2)

    summary = _binned_summary(frame["predicted"], frame["residual"])
    axis.plot(summary["centre"], summary["mean"], color=ACCENT_COLOUR,
              linewidth=2, label="Binned mean residual")
    axis.fill_between(
        summary["centre"], summary["mean"] - summary["sd"], summary["mean"] + summary["sd"],
        color=ACCENT_COLOUR, alpha=0.15, label="±1 SD within bin",
    )

    axis.set_xlabel("Predicted log$_{10}$(revenue)")
    axis.set_ylabel("Residual (observed − predicted, log$_{10}$ units)")
    axis.set_title("Residuals against predicted values: the spread widens with the prediction")
    axis.legend(loc="upper left")
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_residual_distribution(residuals: pd.Series) -> plt.Figure:
    """Histogram of residuals with a fitted normal curve for reference."""
    figure, axis = plt.subplots(figsize=(9, 4.6))
    axis.hist(residuals, bins=50, density=True, color=PRIMARY_COLOUR, alpha=0.85,
              label="Observed residuals")

    grid = np.linspace(residuals.min(), residuals.max(), 400)
    axis.plot(grid, norm.pdf(grid, residuals.mean(), residuals.std()),
              color=ACCENT_COLOUR, linewidth=2,
              label="Normal with the same mean and SD")
    axis.axvline(0, color=INK_SECONDARY, linewidth=1.2)

    axis.set_xlabel("Residual (log$_{10}$ units)")
    axis.set_ylabel("Density")
    axis.set_title("Residual distribution: centred on zero, right-skewed, heavier-tailed than normal")
    axis.legend()
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_feature_error_grid(
    frame: pd.DataFrame, features: list[str], quantity: str, title: str
) -> plt.Figure:
    """Small multiples of each feature against a chosen error quantity.

    Args:
        frame: Frame carrying the features and the error column.
        features: Numeric features to plot.
        quantity: Either "residual" or "absError".
        title: Figure title.
    """
    columns = 3
    rows = int(np.ceil(len(features) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3.4 * rows))

    for axis, feature in zip(axes.ravel(), features):
        subset = frame[[feature, quantity]].dropna()
        axis.scatter(subset[feature], subset[quantity], s=8, alpha=0.3,
                     color=PRIMARY_COLOUR, edgecolor="none")
        summary = _binned_summary(subset[feature], subset[quantity], bins=8)
        axis.plot(summary["centre"], summary["mean"], color=ACCENT_COLOUR,
                  linewidth=2, marker="o", markersize=4)
        if quantity == "residual":
            axis.axhline(0, color=INK_SECONDARY, linewidth=1.0)
        axis.set_xlabel(feature)
        axis.set_ylabel("Residual" if quantity == "residual" else "Absolute error")

    for axis in axes.ravel()[len(features):]:
        axis.set_visible(False)

    figure.suptitle(title, fontsize=13, fontweight="semibold", y=1.0)
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_error_by_subgroup(frame: pd.DataFrame, columns: list[str]) -> plt.Figure:
    """Box plots of residuals within the levels of each categorical column."""
    figure, axes = plt.subplots(1, len(columns), figsize=(4.2 * len(columns), 4.4))
    axes = np.atleast_1d(axes)

    for axis, column in zip(axes, columns):
        # Ordered categoricals carry their own meaningful order; anything else
        # falls back to sorting, so bands are never shown alphabetically.
        if isinstance(frame[column].dtype, pd.CategoricalDtype):
            present = set(frame[column].dropna().unique())
            levels = [c for c in frame[column].cat.categories if c in present]
        else:
            levels = sorted(frame[column].dropna().unique(), key=str)
        data = [frame.loc[frame[column] == level, "residual"].dropna() for level in levels]
        axis.boxplot(
            data, tick_labels=[str(level) for level in levels], widths=0.5,
            patch_artist=True,
            boxprops={"facecolor": PRIMARY_COLOUR, "edgecolor": INK_SECONDARY, "linewidth": 0.9},
            medianprops={"color": "white", "linewidth": 1.4},
            whiskerprops={"color": INK_SECONDARY, "linewidth": 0.9},
            capprops={"color": INK_SECONDARY, "linewidth": 0.9},
            flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": INK_SECONDARY,
                        "markeredgecolor": "none", "alpha": 0.4},
        )
        axis.axhline(0, color=ACCENT_COLOUR, linewidth=1.1)
        axis.set_xlabel(column)
        axis.set_ylabel("Residual (log$_{10}$ units)")
        axis.tick_params(axis="x", rotation=30)

    figure.suptitle("Residual distribution within subpopulations", fontsize=13,
                    fontweight="semibold", y=1.02)
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_extreme_errors(frame: pd.DataFrame, threshold: float) -> plt.Figure:
    """Observed against predicted revenue, with the largest 5% of errors marked."""
    extreme = frame["absError"] >= threshold
    figure, axis = plt.subplots(figsize=(8.5, 6))

    axis.scatter(frame.loc[~extreme, "predicted"], frame.loc[~extreme, "logRevenue"],
                 s=12, alpha=0.35, color=PRIMARY_COLOUR, edgecolor="none",
                 label=f"Within the largest 95% ({(~extreme).sum():,})")
    axis.scatter(frame.loc[extreme, "predicted"], frame.loc[extreme, "logRevenue"],
                 s=26, alpha=0.85, color=ACCENT_COLOUR, edgecolor="white", linewidth=0.4,
                 label=f"Top 5% absolute error ({extreme.sum()})")

    limits = [frame["predicted"].min() - 0.2, frame["logRevenue"].max() + 0.2]
    axis.plot(limits, limits, color=INK_SECONDARY, linewidth=1.2, label="Perfect prediction")

    axis.set_xlabel("Predicted log$_{10}$(revenue)")
    axis.set_ylabel("Observed log$_{10}$(revenue)")
    axis.set_title("Where the largest errors sit: overwhelmingly above the diagonal")
    axis.legend(loc="upper left")
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_tree_depth_curve(sweep: pd.DataFrame) -> plt.Figure:
    """Train and test R2 against tree depth, showing where overfitting begins."""
    labels = [str(index) for index in sweep.index]
    positions = np.arange(len(labels))

    figure, axis = plt.subplots(figsize=(9, 4.6))
    axis.plot(positions, sweep["Train R2"], color=ACCENT_COLOUR, marker="o",
              markersize=5, label="Training R²")
    axis.plot(positions, sweep["Test R2"], color=PRIMARY_COLOUR, marker="o",
              markersize=5, label="Out-of-fold R²")
    axis.fill_between(positions, sweep["Test R2"], sweep["Train R2"],
                      color=ACCENT_COLOUR, alpha=0.10, label="Generalisation gap")
    axis.axhline(0, color=INK_SECONDARY, linewidth=1.0)

    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_xlabel("Decision tree max_depth")
    axis.set_ylabel("R²")
    axis.set_title("The tree memorises the training folds long before it predicts well")
    axis.legend(loc="center right")
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_metric_comparison(metrics: pd.DataFrame) -> plt.Figure:
    """Grouped bars for the error metrics and a separate panel for R2.

    R2 is kept on its own axis because it runs in the opposite direction to the
    error metrics and can be negative; plotting them together would need two
    scales on one chart.
    """
    models = list(metrics.index)
    positions = np.arange(len(models))

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    width = 0.27
    for offset, (metric, colour) in enumerate(
        zip(["MAE", "RMSE", "MSE"], CATEGORICAL_COLOURS)
    ):
        axes[0].bar(positions + (offset - 1) * width, metrics[metric], width,
                    color=colour, label=metric, edgecolor="white", linewidth=1.5)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(models, rotation=20, ha="right")
    axes[0].set_ylabel("Error (log$_{10}$ units, lower is better)")
    axes[0].set_title("Error metrics")
    axes[0].legend()

    colours = [ACCENT_COLOUR if value < 0 else PRIMARY_COLOUR for value in metrics["R2"]]
    bars = axes[1].bar(positions, metrics["R2"], 0.55, color=colours,
                       edgecolor="white", linewidth=1.5)
    for bar, value in zip(bars, metrics["R2"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     value + (0.015 if value >= 0 else -0.045),
                     f"{value:.3f}", ha="center", fontsize=9, color=INK_SECONDARY)
    axes[1].axhline(0, color=INK_SECONDARY, linewidth=1.1)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(models, rotation=20, ha="right")
    axes[1].set_ylabel("R² (higher is better)")
    axes[1].set_title("Explained variance")

    figure.suptitle("Model comparison on identical features and folds", fontsize=13,
                    fontweight="semibold", y=1.02)
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_fold_stability(fold_scores: pd.DataFrame) -> plt.Figure:
    """Per-fold R2 for each model, showing how stable the comparison is."""
    figure, axis = plt.subplots(figsize=(9, 4.6))
    models = list(fold_scores.columns)

    for position, model in enumerate(models):
        values = fold_scores[model]
        axis.scatter([position] * len(values), values, s=55, alpha=0.75,
                     color=PRIMARY_COLOUR, edgecolor="white", linewidth=0.6,
                     zorder=3)
        axis.plot([position - 0.18, position + 0.18], [values.mean()] * 2,
                  color=ACCENT_COLOUR, linewidth=2.4, zorder=4)

    axis.axhline(0, color=INK_SECONDARY, linewidth=1.0)
    axis.set_xticks(np.arange(len(models)))
    axis.set_xticklabels(models, rotation=20, ha="right")
    axis.set_ylabel("R² on the held-out fold")
    axis.set_title("Each point is one fold; the bar marks the mean")
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_predicted_vs_actual(
    predictions: dict, observed: pd.Series, models: list[str]
) -> plt.Figure:
    """Observed against predicted values, one panel per model."""
    figure, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.6),
                                sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    limits = [observed.min() - 0.2, observed.max() + 0.2]

    for axis, model in zip(axes, models):
        axis.scatter(predictions[model], observed, s=10, alpha=0.35,
                     color=PRIMARY_COLOUR, edgecolor="none")
        axis.plot(limits, limits, color=INK_SECONDARY, linewidth=1.2)
        axis.set_xlabel("Predicted log$_{10}$(revenue)")
        axis.set_title(model, fontsize=10)
    axes[0].set_ylabel("Observed log$_{10}$(revenue)")

    figure.suptitle("None of the models reaches the top of the observed range",
                    fontsize=13, fontweight="semibold", y=1.02)
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_explanation_comparison(explanations: pd.DataFrame, top_n: int = 8) -> plt.Figure:
    """Linear coefficients beside forest importances for the leading features."""
    subset = explanations.head(top_n).iloc[::-1]
    positions = np.arange(len(subset))

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)

    colours = [ACCENT_COLOUR if value < 0 else PRIMARY_COLOUR
               for value in subset["Linear coefficient"]]
    axes[0].barh(positions, subset["Linear coefficient"], 0.6, color=colours,
                 edgecolor="white", linewidth=1.2)
    axes[0].axvline(0, color=INK_SECONDARY, linewidth=1.1)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(subset.index)
    axes[0].set_xlabel("Linear coefficient (signed)")
    axes[0].set_title("Linear Regression: direction and size")

    axes[1].barh(positions, subset["Random Forest importance"], 0.6,
                 color=PRIMARY_COLOUR, edgecolor="white", linewidth=1.2)
    axes[1].set_xlabel("Impurity-based importance (unsigned)")
    axes[1].set_title("Random Forest: reliance, without direction")

    figure.suptitle("The two models explain the same data differently",
                    fontsize=13, fontweight="semibold", y=1.02)
    figure.tight_layout()
    plt.close(figure)
    return figure
