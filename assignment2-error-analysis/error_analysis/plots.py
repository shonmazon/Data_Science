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
    INK_PRIMARY,
    INK_SECONDARY,
    PRIMARY_COLOUR,
    SEQUENTIAL_COLOURMAP,
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
    # chapter 4
    "plot_confusion_matrices",
    "plot_confidence_distributions",
    "plot_feature_outcome_distributions",
    "plot_threshold_curves",
    "plot_fbeta_curves",
    "plot_roc_curves",
    "plot_error_rate_by_decile",
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


def plot_confusion_matrices(matrices: dict) -> plt.Figure:
    """Confusion matrix heatmaps, annotated with counts and row percentages."""
    figure, axes = plt.subplots(1, len(matrices), figsize=(5.4 * len(matrices), 4.6))
    axes = np.atleast_1d(axes)
    labels = ["Not standout", "Standout"]

    for axis, (name, matrix) in zip(axes, matrices.items()):
        axis.imshow(matrix / matrix.sum(axis=1, keepdims=True),
                    cmap=SEQUENTIAL_COLOURMAP, vmin=0, vmax=1)
        for row in range(2):
            for column in range(2):
                count = matrix[row, column]
                share = count / matrix[row].sum()
                axis.text(column, row, f"{count:,}\n{share:.1%} of row",
                          ha="center", va="center", fontsize=11,
                          color="white" if share > 0.55 else INK_PRIMARY)
        axis.set_xticks([0, 1], labels)
        axis.set_yticks([0, 1], labels)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_title(name, fontsize=11)
        axis.grid(False)

    figure.suptitle("Confusion matrices at a 0.5 threshold, from out-of-fold predictions",
                    fontsize=13, fontweight="semibold", y=1.03)
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_confidence_distributions(
    confidence: np.ndarray, correct: np.ndarray
) -> plt.Figure:
    """Confidence when the model is right against when it is wrong."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    bins = np.linspace(0.5, 1.0, 26)
    axes[0].hist(confidence[correct], bins=bins, alpha=0.75, density=True,
                 color=PRIMARY_COLOUR, label=f"Correct (n={correct.sum():,})")
    axes[0].hist(confidence[~correct], bins=bins, alpha=0.75, density=True,
                 color=ACCENT_COLOUR, label=f"Incorrect (n={(~correct).sum():,})")
    axes[0].set_xlabel("Confidence in the predicted class")
    axes[0].set_ylabel("Density")
    axes[0].set_title("The distributions overlap heavily")
    axes[0].legend()

    centres, rates, counts = [], [], []
    for low, high in zip(bins[:-1], bins[1:]):
        inside = (confidence >= low) & (confidence < high)
        if inside.sum() >= 10:
            centres.append((low + high) / 2)
            rates.append(1 - correct[inside].mean())
            counts.append(inside.sum())
    axes[1].plot(centres, rates, color=ACCENT_COLOUR, marker="o", markersize=5)
    axes[1].set_xlabel("Confidence in the predicted class")
    axes[1].set_ylabel("Observed error rate")
    axes[1].set_title("Error rate falls with confidence, but never reaches zero")
    axes[1].set_ylim(bottom=0)

    figure.suptitle("Are confident predictions more reliable?", fontsize=13,
                    fontweight="semibold", y=1.02)
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_feature_outcome_distributions(
    frame: pd.DataFrame, features: list[str], correct: pd.Series
) -> plt.Figure:
    """Feature distributions split by whether the prediction was right."""
    figure, axes = plt.subplots(1, len(features), figsize=(4.4 * len(features), 4.4))
    axes = np.atleast_1d(axes)

    for axis, feature in zip(axes, features):
        subset = frame[[feature]].assign(correct=correct).dropna()
        data = [subset.loc[subset["correct"], feature], subset.loc[~subset["correct"], feature]]
        parts = axis.violinplot(data, showmedians=True, widths=0.7)
        for body, colour in zip(parts["bodies"], [PRIMARY_COLOUR, ACCENT_COLOUR]):
            body.set_facecolor(colour)
            body.set_alpha(0.75)
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color(INK_SECONDARY)
        axis.set_xticks([1, 2], ["Correct", "Incorrect"])
        axis.set_ylabel(feature)
        axis.set_title(feature, fontsize=10)

    figure.suptitle("Misclassified games are more expensive and played for longer",
                    fontsize=13, fontweight="semibold", y=1.02)
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_threshold_curves(sweep: pd.DataFrame) -> plt.Figure:
    """Metrics against threshold, and the false-positive/false-negative trade-off."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    thresholds = sweep.index

    for metric, colour in zip(["Precision", "Recall", "F1", "MCC"],
                              CATEGORICAL_COLOURS + ["#eda100"]):
        axes[0].plot(thresholds, sweep[metric], marker="o", markersize=5,
                     color=colour, label=metric)
    best_mcc = sweep["MCC"].idxmax()
    axes[0].axvline(best_mcc, color=INK_SECONDARY, linewidth=1.1)
    axes[0].annotate(f"MCC peaks at {best_mcc:.1f}", xy=(best_mcc, sweep["MCC"].max()),
                     xytext=(best_mcc + 0.06, sweep["MCC"].max() + 0.14),
                     fontsize=9, color=INK_SECONDARY,
                     arrowprops={"arrowstyle": "->", "color": INK_SECONDARY, "linewidth": 1})
    axes[0].set_xlabel("Decision threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Performance across thresholds")
    axes[0].legend(loc="upper right")

    axes[1].plot(thresholds, sweep["FP"], marker="o", markersize=5,
                 color=ACCENT_COLOUR, label="False positives")
    axes[1].plot(thresholds, sweep["FN"], marker="o", markersize=5,
                 color=PRIMARY_COLOUR, label="False negatives")
    axes[1].set_xlabel("Decision threshold")
    axes[1].set_ylabel("Number of games")
    axes[1].set_title("The trade-off, in counts rather than rates")
    axes[1].legend()

    figure.suptitle("Threshold sensitivity from 0.1 to 0.9", fontsize=13,
                    fontweight="semibold", y=1.02)
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_fbeta_curves(grid: pd.DataFrame) -> plt.Figure:
    """F-beta against beta, one line per threshold, with the optimum marked."""
    betas = [float(column.split("=")[1]) for column in grid.columns]
    figure, axis = plt.subplots(figsize=(9, 5))

    palette = plt.cm.viridis(np.linspace(0.15, 0.9, len(grid.index)))
    for colour, (threshold, row) in zip(palette, grid.iterrows()):
        axis.plot(betas, row.to_numpy(), marker="o", markersize=4,
                  color=colour, label=f"t = {threshold:.1f}")

    best_per_beta = [grid[column].idxmax() for column in grid.columns]
    axis.plot(betas, [grid.loc[t, c] for t, c in zip(best_per_beta, grid.columns)],
              color=INK_PRIMARY, linewidth=2.4, linestyle=":", label="Best threshold")

    axis.set_xlabel(r"$\beta$  (weight on recall relative to precision)")
    axis.set_ylabel(r"$F_\beta$ score")
    axis.set_title(r"As $\beta$ rises, the best threshold falls")
    axis.legend(ncol=2, fontsize=8, loc="upper left")
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_roc_curves(curves: dict) -> plt.Figure:
    """ROC curves for each classifier, with the chance diagonal."""
    figure, axis = plt.subplots(figsize=(6.4, 6))

    for colour, (name, (false_positive_rate, true_positive_rate, area)) in zip(
        CATEGORICAL_COLOURS, curves.items()
    ):
        axis.plot(false_positive_rate, true_positive_rate, color=colour, linewidth=2,
                  label=f"{name} (AUC = {area:.3f})")
    axis.plot([0, 1], [0, 1], color=INK_SECONDARY, linewidth=1.2, label="Chance (AUC = 0.5)")

    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("ROC curves from out-of-fold probabilities")
    axis.legend(loc="lower right")
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_error_rate_by_decile(decile_table: pd.DataFrame, boundary_decile: int) -> plt.Figure:
    """Error rate against revenue decile, marking where the class boundary falls."""
    figure, axis = plt.subplots(figsize=(9.5, 4.6))
    colours = [ACCENT_COLOUR if index >= boundary_decile else PRIMARY_COLOUR
               for index in decile_table.index]
    axis.bar(decile_table.index, decile_table["Error rate"], 0.65, color=colours,
             edgecolor="white", linewidth=1.5)

    axis.axvline(boundary_decile - 0.5, color=INK_SECONDARY, linewidth=1.4)
    axis.annotate("class boundary\n(75th percentile)",
                  xy=(boundary_decile - 0.5, decile_table["Error rate"].max() * 0.92),
                  xytext=(boundary_decile - 3.6, decile_table["Error rate"].max() * 0.92),
                  fontsize=9, color=INK_SECONDARY,
                  arrowprops={"arrowstyle": "->", "color": INK_SECONDARY, "linewidth": 1})

    axis.set_xticks(decile_table.index)
    axis.set_xlabel("Revenue decile (1 = lowest earning)")
    axis.set_ylabel("Error rate")
    axis.set_title("Errors concentrate immediately around the class boundary")
    figure.tight_layout()
    plt.close(figure)
    return figure
