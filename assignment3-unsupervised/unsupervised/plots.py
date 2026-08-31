"""Figures for the final homework.

The visual style is imported from Assignment 1 so that all three notebooks share
one palette, grid treatment and typography.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis, skew

from .data_setup import ASSIGNMENT1_DIR  # noqa: F401  (puts A1 on the import path)
from src.plotting import (  # noqa: E402
    DIVERGING_COLOURMAP,
    INK_PRIMARY,
    INK_SECONDARY,
    PRIMARY_COLOUR,
    SEQUENTIAL_COLOURMAP,
    apply_house_style,
)

__all__ = [
    "apply_house_style",
    "plot_distribution_grid",
    "plot_boxplot_grid",
    "plot_transform_effect",
    "plot_correlation_pair",
]

# Matplotlib cannot resolve the "semibold" weight for the default font and falls
# back to bold, which is fine, but the notice clutters every figure.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

ACCENT_COLOUR = "#eb6834"


def _axis_label(column: str, logged: bool) -> str:
    return f"log$_{{10}}$({column})" if logged else column


def plot_distribution_grid(
    matrix: pd.DataFrame, logged_features: list[str], title: str
) -> plt.Figure:
    """Histogram of every feature, annotated with its skewness and kurtosis."""
    columns = list(matrix.columns)
    figure, axes = plt.subplots(2, 4, figsize=(15, 7))

    for axis, column in zip(axes.ravel(), columns):
        values = matrix[column].dropna()
        bins = 2 if values.nunique() == 2 else 45
        axis.hist(values, bins=bins, color=PRIMARY_COLOUR)
        axis.axvline(values.median(), color=ACCENT_COLOUR, linewidth=1.6)
        axis.set_xlabel(_axis_label(column, column in logged_features))
        axis.set_ylabel("Games")
        axis.set_title(
            f"skew {skew(values):+.2f}   kurtosis {kurtosis(values):+.1f}",
            fontsize=9.5, color=INK_SECONDARY,
        )

    figure.suptitle(title, fontsize=13, fontweight="semibold", y=1.01)
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_boxplot_grid(
    matrix: pd.DataFrame, logged_features: list[str], title: str
) -> plt.Figure:
    """Box plot of every feature, each on its own axis."""
    columns = list(matrix.columns)
    figure, axes = plt.subplots(1, len(columns), figsize=(15, 4.4))

    for axis, column in zip(axes, columns):
        axis.boxplot(
            matrix[column].dropna(), widths=0.5, patch_artist=True,
            boxprops={"facecolor": PRIMARY_COLOUR, "edgecolor": INK_SECONDARY, "linewidth": 0.9},
            medianprops={"color": "white", "linewidth": 1.4},
            whiskerprops={"color": INK_SECONDARY, "linewidth": 0.9},
            capprops={"color": INK_SECONDARY, "linewidth": 0.9},
            flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": INK_SECONDARY,
                        "markeredgecolor": "none", "alpha": 0.4},
        )
        axis.set_xticks([])
        axis.set_title(_axis_label(column, column in logged_features), fontsize=9.5)

    figure.suptitle(title, fontsize=13, fontweight="semibold", y=1.03)
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_transform_effect(
    raw_matrix: pd.DataFrame, log_matrix: pd.DataFrame, features: list[str]
) -> plt.Figure:
    """The heavy-tailed features before and after the log transform.

    This figure is the evidence for the preprocessing decision rather than an
    illustration of it: the top row shows distributions no distance-based method
    can work with, the bottom row shows the same variables made usable.
    """
    figure, axes = plt.subplots(2, len(features), figsize=(4 * len(features), 7))

    for column_index, column in enumerate(features):
        raw = raw_matrix[column].dropna()
        logged = log_matrix[column].dropna()

        axes[0, column_index].hist(raw, bins=45, color=PRIMARY_COLOUR)
        axes[0, column_index].set_title(f"{column}\nraw: skew {skew(raw):+.1f}", fontsize=10)
        axes[0, column_index].set_xlabel(column)
        axes[0, column_index].set_ylabel("Games")

        axes[1, column_index].hist(logged, bins=45, color=ACCENT_COLOUR)
        axes[1, column_index].set_title(f"log$_{{10}}$: skew {skew(logged):+.2f}", fontsize=10)
        axes[1, column_index].set_xlabel(_axis_label(column, True))
        axes[1, column_index].set_ylabel("Games")

    figure.suptitle(
        "The same four variables before (top) and after (bottom) the log transform",
        fontsize=13, fontweight="semibold", y=1.0,
    )
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_correlation_pair(raw_matrix: pd.DataFrame, log_matrix: pd.DataFrame) -> plt.Figure:
    """Correlation matrices on both scales, on one shared colour scale."""
    figure, axes = plt.subplots(1, 2, figsize=(15, 6))

    for axis, matrix, label in [
        (axes[0], raw_matrix.corr(), "Raw scale"),
        (axes[1], log_matrix.corr(), "Analysis scale (heavy tails logged)"),
    ]:
        sns.heatmap(
            matrix, annot=True, fmt=".2f", cmap=DIVERGING_COLOURMAP, vmin=-1, vmax=1,
            center=0, square=True, linewidths=2, linecolor="white",
            cbar_kws={"label": "Pearson r", "shrink": 0.8}, annot_kws={"fontsize": 8},
            ax=axis,
        )
        strongest = matrix.where(~np.eye(len(matrix), dtype=bool)).abs().max().max()
        axis.set_title(f"{label}   (strongest |r| = {strongest:.2f})", fontsize=11)
        axis.grid(False)

    figure.suptitle(
        "Heavy tails suppress correlation: the same data, two scales",
        fontsize=13, fontweight="semibold", y=1.02,
    )
    figure.tight_layout()
    plt.close(figure)
    return figure
