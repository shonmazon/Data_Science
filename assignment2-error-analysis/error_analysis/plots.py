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
    "plot_residuals_vs_predicted",
    "plot_residual_distribution",
    "plot_feature_error_grid",
    "plot_error_by_subgroup",
    "plot_extreme_errors",
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
