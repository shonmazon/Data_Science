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
    GRID_COLOUR,
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
    "plot_scree",
    "plot_projection_2d",
    "plot_projection_3d",
    "plot_loadings",
    "plot_reconstruction_error",
    "plot_component_stability",
    "plot_kmeans_sweep",
    "plot_k_distance",
    "plot_dendrogram",
    "plot_clusters_on_pca",
    "plot_metric_disagreement",
    "plot_feature_space",
    "plot_zscore_diagnostics",
    "plot_score_distribution",
    "plot_lof_sensitivity",
    "plot_anomalies_on_pca",
    "plot_method_agreement",
    "plot_sensitivity_curves",
    "plot_runtime",
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


def plot_scree(variance: pd.DataFrame) -> plt.Figure:
    """Scree plot beside the cumulative explained variance."""
    positions = np.arange(1, len(variance) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    axes[0].bar(positions, variance["Explained variance"], 0.6, color=PRIMARY_COLOUR,
                edgecolor="white", linewidth=1.5)
    axes[0].plot(positions, variance["Explained variance"], color=ACCENT_COLOUR,
                 marker="o", markersize=5, linewidth=1.6)
    axes[0].set_xticks(positions)
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Share of total variance")
    axes[0].set_title("Scree plot: no sharp elbow")

    axes[1].plot(positions, variance["Cumulative"], color=PRIMARY_COLOUR, marker="o",
                 markersize=5, linewidth=2)
    for level, style in [(0.80, ":"), (0.90, "--")]:
        axes[1].axhline(level, color=INK_SECONDARY, linewidth=1.0, linestyle=style)
        axes[1].annotate(f"{level:.0%}", xy=(0.6, level + 0.012), fontsize=9,
                         color=INK_SECONDARY)
    axes[1].set_xticks(positions)
    axes[1].set_ylim(0, 1.03)
    axes[1].set_xlabel("Number of components retained")
    axes[1].set_ylabel("Cumulative variance")
    axes[1].set_title("Cumulative explained variance")

    figure.suptitle("Variance is spread across components rather than concentrated",
                    fontsize=13, fontweight="semibold", y=1.02)
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_projection_2d(scores: np.ndarray, frame: pd.DataFrame, variance: pd.DataFrame) -> plt.Figure:
    """The 2D projection, annotated two ways with variables PCA never saw."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    labels = [f"PC{i} ({variance['Explained variance'].iloc[i-1]:.1%})" for i in (1, 2)]

    order = ["Indie", "AA", "AAA"]
    colours = ["#2a78d6", "#eb6834", "#1baf7a"]
    for name, colour in zip(order, colours):
        mask = (frame["plotClass"].astype(str) == name).to_numpy()
        axes[0].scatter(scores[mask, 0], scores[mask, 1], s=14, alpha=0.6, color=colour,
                        edgecolor="none", label=f"{name} (n={mask.sum():,})")
    axes[0].legend(title="Publisher class", loc="upper left")
    axes[0].set_title("Coloured by studio scale, which PCA never saw")

    revenue = np.log10(frame["revenue"].to_numpy())
    points = axes[1].scatter(scores[:, 0], scores[:, 1], s=14, alpha=0.7, c=revenue,
                             cmap=SEQUENTIAL_COLOURMAP, edgecolor="none")
    figure.colorbar(points, ax=axes[1], shrink=0.85).set_label("log$_{10}$(revenue)")
    axes[1].set_title("Coloured by revenue: PC1 is a commercial-scale axis")

    for axis in axes:
        axis.set_xlabel(labels[0])
        axis.set_ylabel(labels[1])

    figure.suptitle("Projection onto the first two principal components",
                    fontsize=13, fontweight="semibold", y=1.01)
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_projection_3d(scores: np.ndarray, frame: pd.DataFrame, variance: pd.DataFrame) -> plt.Figure:
    """The 3D projection, viewed from two angles."""
    figure = plt.figure(figsize=(14, 6))
    order = ["Indie", "AA", "AAA"]
    colours = ["#2a78d6", "#eb6834", "#1baf7a"]
    cumulative = variance["Cumulative"].iloc[2]

    for position, (elevation, azimuth) in enumerate([(18, 45), (18, 135)], start=1):
        axis = figure.add_subplot(1, 2, position, projection="3d")
        for name, colour in zip(order, colours):
            mask = (frame["plotClass"].astype(str) == name).to_numpy()
            axis.scatter(scores[mask, 0], scores[mask, 1], scores[mask, 2],
                         s=16, alpha=0.75, color=colour, edgecolor="none",
                         depthshade=False, label=name)
        axis.view_init(elev=elevation, azim=azimuth)
        # The default translucent panes wash the points out at this density.
        for pane in (axis.xaxis, axis.yaxis, axis.zaxis):
            pane.pane.set_alpha(0.0)
            pane.pane.set_edgecolor(GRID_COLOUR)
        axis.set_xlabel("PC1"); axis.set_ylabel("PC2"); axis.set_zlabel("PC3")
        axis.set_title(f"view {position}", fontsize=10)
        if position == 1:
            axis.legend(title="Publisher class", loc="upper left", fontsize=8)

    figure.suptitle(
        f"Projection onto the first three components, together {cumulative:.1%} of variance",
        fontsize=13, fontweight="semibold", y=0.98,
    )
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_loadings(loadings: pd.DataFrame) -> plt.Figure:
    """Heatmap of how each original feature contributes to each component."""
    figure, axis = plt.subplots(figsize=(8, 5.4))
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap=DIVERGING_COLOURMAP, center=0,
                vmin=-0.8, vmax=0.8, linewidths=2, linecolor="white",
                cbar_kws={"label": "Loading", "shrink": 0.85},
                annot_kws={"fontsize": 9}, ax=axis)
    axis.set_title("Feature loadings on the leading components")
    axis.grid(False)
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_reconstruction_error(errors: pd.DataFrame) -> plt.Figure:
    """Per-feature reconstruction error as components are added back."""
    figure, axis = plt.subplots(figsize=(9.5, 5))
    features = [c for c in errors.columns if c != "Variance kept"]
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(features)))

    for colour, feature in zip(palette, features):
        axis.plot(errors.index, errors[feature], marker="o", markersize=5,
                  color=colour, label=feature)

    axis.set_xticks(list(errors.index))
    axis.set_xlabel("Components retained")
    axis.set_ylabel("Reconstruction RMSE (standard deviations)")
    axis.set_title("What a projection discards, feature by feature")
    axis.legend(ncol=2, fontsize=8.5)
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_component_stability(similarities: dict) -> plt.Figure:
    """Distribution of bootstrap agreement with each full-sample component axis."""
    figure, axis = plt.subplots(figsize=(9, 4.8))
    data = [np.array(similarities[i]) for i in sorted(similarities)]
    labels = [f"PC{i + 1}" for i in sorted(similarities)]

    parts = axis.violinplot(data, showmedians=True, widths=0.7)
    for body, colour in zip(parts["bodies"], ["#2a78d6", "#2a78d6", "#eb6834"]):
        body.set_facecolor(colour); body.set_alpha(0.8)
    for key in ("cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color(INK_SECONDARY)

    axis.axhline(0.9, color=INK_SECONDARY, linewidth=1.1, linestyle="--")
    axis.annotate("0.9", xy=(0.55, 0.905), fontsize=9, color=INK_SECONDARY)
    axis.set_xticks(range(1, len(labels) + 1), labels)
    axis.set_ylabel("|cosine| with the full-sample axis")
    axis.set_title("PC1 and PC2 survive resampling; PC3 does not")
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_kmeans_sweep(sweep: pd.DataFrame) -> plt.Figure:
    """Inertia and silhouette across k: the elbow and the quality criterion."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    axes[0].plot(sweep.index, sweep["Inertia"], color=PRIMARY_COLOUR, marker="o", markersize=5)
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Within-cluster sum of squares")
    axes[0].set_title("Inertia declines smoothly: no elbow")

    axes[1].plot(sweep.index, sweep["Silhouette"], color=ACCENT_COLOUR, marker="o", markersize=5)
    axes[1].axhline(0.25, color=INK_SECONDARY, linewidth=1.1, linestyle="--")
    axes[1].annotate("0.25: conventional floor for 'some structure'",
                     xy=(sweep.index[2], 0.257), fontsize=9, color=INK_SECONDARY)
    axes[1].set_ylim(0, 0.42)
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Mean silhouette")
    axes[1].set_title("Silhouette never reaches the conventional floor")

    figure.suptitle("K-Means across k", fontsize=13, fontweight="semibold", y=1.02)
    figure.tight_layout(); plt.close(figure)
    return figure


def plot_k_distance(curve: np.ndarray, eps: float, n_neighbors: int) -> plt.Figure:
    """Sorted k-th nearest-neighbour distance, used to choose DBSCAN's eps."""
    figure, axis = plt.subplots(figsize=(9, 4.6))
    axis.plot(np.arange(len(curve)), curve, color=PRIMARY_COLOUR, linewidth=2)
    axis.axhline(eps, color=ACCENT_COLOUR, linewidth=1.6)
    axis.annotate(f"chosen eps = {eps}", xy=(len(curve) * 0.04, eps + 0.09),
                  fontsize=9.5, color=ACCENT_COLOUR)
    axis.set_xlabel("Games, sorted by distance")
    axis.set_ylabel(f"Distance to the {n_neighbors}th nearest neighbour")
    axis.set_title("The k-distance curve bends gradually rather than at a knee")
    figure.tight_layout(); plt.close(figure)
    return figure


def plot_dendrogram(tree, labels=None, title="", colour_threshold=None) -> plt.Figure:
    """Dendrogram of a hierarchical clustering."""
    from scipy.cluster.hierarchy import dendrogram

    figure, axis = plt.subplots(figsize=(11, 4.8))
    dendrogram(
        tree, ax=axis, labels=labels, color_threshold=colour_threshold,
        above_threshold_color=INK_SECONDARY, no_labels=labels is None,
    )
    axis.set_ylabel("Merge distance")
    axis.set_title(title)
    if labels is not None:
        axis.tick_params(axis="x", rotation=35, labelsize=9)
    figure.tight_layout(); plt.close(figure)
    return figure


def plot_clusters_on_pca(scores: np.ndarray, labellings: dict, variance: pd.DataFrame) -> plt.Figure:
    """Each method's partition drawn on the same two principal components."""
    figure, axes = plt.subplots(1, len(labellings), figsize=(5.2 * len(labellings), 5))
    axes = np.atleast_1d(axes)
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]

    for axis, (name, labels) in zip(axes, labellings.items()):
        for value in sorted(set(labels)):
            mask = labels == value
            is_noise = value == -1
            axis.scatter(
                scores[mask, 0], scores[mask, 1], s=11,
                alpha=0.28 if is_noise else 0.65,
                color=INK_SECONDARY if is_noise else palette[value % len(palette)],
                edgecolor="none",
                label=f"noise ({mask.sum()})" if is_noise else f"cluster {value} ({mask.sum()})",
            )
        axis.set_xlabel(f"PC1 ({variance['Explained variance'].iloc[0]:.1%})")
        axis.set_ylabel(f"PC2 ({variance['Explained variance'].iloc[1]:.1%})")
        axis.set_title(name, fontsize=10)
        axis.legend(fontsize=8, loc="upper right")

    figure.suptitle("The same data, three partitions, one projection",
                    fontsize=13, fontweight="semibold", y=1.02)
    figure.tight_layout(); plt.close(figure)
    return figure


def plot_metric_disagreement(sweep: pd.DataFrame, ratios: pd.Series) -> plt.Figure:
    """The named variance ratio against silhouette as k grows."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    axes[0].plot(ratios.index, ratios.to_numpy(), color=ACCENT_COLOUR, marker="o", markersize=5)
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Within-cluster variance / global variance")
    axes[0].set_title("The named metric improves without limit as k grows")

    axes[1].plot(sweep.index, sweep["Silhouette"], color=PRIMARY_COLOUR, marker="o", markersize=5)
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Mean silhouette")
    axes[1].set_title("Silhouette, which penalises over-splitting, declines")

    figure.suptitle("Why the variance ratio cannot choose k", fontsize=13,
                    fontweight="semibold", y=1.02)
    figure.tight_layout(); plt.close(figure)
    return figure


def plot_feature_space(distance: pd.DataFrame, tree) -> plt.Figure:
    """Feature distance matrix beside the dendrogram of the transposed matrix."""
    from scipy.cluster.hierarchy import dendrogram

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.6))

    sns.heatmap(distance, annot=True, fmt=".2f", cmap=SEQUENTIAL_COLOURMAP, vmin=0, vmax=1,
                square=True, linewidths=2, linecolor="white",
                cbar_kws={"label": "1 - |correlation|", "shrink": 0.8},
                annot_kws={"fontsize": 7.5}, ax=axes[0])
    axes[0].set_title("Distance between features")
    axes[0].grid(False)

    dendrogram(tree, ax=axes[1], labels=list(distance.index), color_threshold=0.7,
               above_threshold_color=INK_SECONDARY)
    axes[1].set_ylabel("Merge distance (1 - |r|)")
    axes[1].set_title("Hierarchy of the feature space")
    axes[1].tick_params(axis="x", rotation=35, labelsize=9)

    figure.suptitle("Clustering the transpose: which features carry the same information",
                    fontsize=13, fontweight="semibold", y=1.02)
    figure.tight_layout(); plt.close(figure)
    return figure


def plot_zscore_diagnostics(diagnostics: pd.DataFrame, threshold: float = 3.0) -> plt.Figure:
    """Which features can produce a Z-score flag at all, and which cannot."""
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    order = diagnostics["Max |z| observed"].sort_values()
    colours = [INK_SECONDARY if v < threshold else PRIMARY_COLOUR for v in order]
    axes[0].barh(np.arange(len(order)), order.to_numpy(), 0.62, color=colours,
                 edgecolor="white", linewidth=1.2)
    axes[0].axvline(threshold, color=ACCENT_COLOUR, linewidth=1.6)
    axes[0].annotate(f"threshold |z| = {threshold:g}", xy=(threshold + 0.1, 0.1),
                     fontsize=9, color=ACCENT_COLOUR)
    axes[0].set_yticks(np.arange(len(order)), order.index)
    axes[0].set_xlabel("Largest standardised deviation observed")
    axes[0].set_title("Three features never reach the threshold")

    counts = diagnostics[f"Games with |z| > {threshold:g}"].sort_values()
    axes[1].barh(np.arange(len(counts)), counts.to_numpy(), 0.62, color=PRIMARY_COLOUR,
                 edgecolor="white", linewidth=1.2)
    axes[1].set_yticks(np.arange(len(counts)), counts.index)
    axes[1].set_xlabel("Games flagged by this feature alone")
    axes[1].set_title("All flags come from five of the eight features")

    figure.suptitle("Diagnostics for the feature-wise Z-score rule", fontsize=13,
                    fontweight="semibold", y=1.02)
    figure.tight_layout(); plt.close(figure)
    return figure


def plot_score_distribution(scores: np.ndarray, flags: np.ndarray, title: str,
                            score_label: str) -> plt.Figure:
    """Anomaly score distribution with the decision cut-off marked."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    axes[0].hist(scores, bins=60, color=PRIMARY_COLOUR)
    axes[0].axvline(scores[flags].min(), color=ACCENT_COLOUR, linewidth=1.8)
    axes[0].annotate("cut-off", xy=(scores[flags].min(), axes[0].get_ylim()[1] * 0.8),
                     xytext=(scores[flags].min() - (scores.max() - scores.min()) * 0.30,
                             axes[0].get_ylim()[1] * 0.8),
                     fontsize=9, color=ACCENT_COLOUR,
                     arrowprops={"arrowstyle": "->", "color": ACCENT_COLOUR, "linewidth": 1})
    axes[0].set_xlabel(score_label); axes[0].set_ylabel("Games")
    axes[0].set_title("The score distribution is continuous, with no natural break")

    ordered = np.sort(scores)[::-1]
    axes[1].plot(np.arange(1, len(ordered) + 1), ordered, color=PRIMARY_COLOUR, linewidth=2)
    axes[1].axvline(flags.sum(), color=ACCENT_COLOUR, linewidth=1.6)
    axes[1].annotate(f"{int(flags.sum())} flagged", xy=(flags.sum() + 40, ordered[0] * 0.98),
                     fontsize=9, color=ACCENT_COLOUR)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Games, ranked by score (log scale)"); axes[1].set_ylabel(score_label)
    axes[1].set_title("Where the threshold falls on the ranking")

    figure.suptitle(title, fontsize=13, fontweight="semibold", y=1.02)
    figure.tight_layout(); plt.close(figure)
    return figure


def plot_lof_sensitivity(sweep: pd.DataFrame, reference_k: int) -> plt.Figure:
    """Agreement of the flagged set with a reference k, across k."""
    figure, axis = plt.subplots(figsize=(9, 4.6))
    column = f"Jaccard with k={reference_k}"
    axis.plot(sweep.index, sweep[column], color=PRIMARY_COLOUR, marker="o", markersize=6)
    axis.axvline(reference_k, color=ACCENT_COLOUR, linewidth=1.4)
    axis.annotate(f"reference k = {reference_k}", xy=(reference_k + 2, 0.5),
                  fontsize=9, color=ACCENT_COLOUR)
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("n_neighbors")
    axis.set_ylabel(f"Jaccard overlap with the k={reference_k} flag set")
    axis.set_title("The set of flagged games changes substantially with k")
    figure.tight_layout(); plt.close(figure)
    return figure


def plot_anomalies_on_pca(scores_2d: np.ndarray, flags: pd.DataFrame,
                          variance: pd.DataFrame) -> plt.Figure:
    """Each method's flagged points on the shared projection."""
    figure, axes = plt.subplots(1, flags.shape[1], figsize=(5.2 * flags.shape[1], 5))
    axes = np.atleast_1d(axes)

    for axis, method in zip(axes, flags.columns):
        mask = flags[method].to_numpy()
        axis.scatter(scores_2d[~mask, 0], scores_2d[~mask, 1], s=10, alpha=0.3,
                     color=PRIMARY_COLOUR, edgecolor="none", label=f"normal ({(~mask).sum():,})")
        axis.scatter(scores_2d[mask, 0], scores_2d[mask, 1], s=30, alpha=0.9,
                     color=ACCENT_COLOUR, edgecolor="white", linewidth=0.4,
                     label=f"flagged ({mask.sum()})")
        axis.set_xlabel(f"PC1 ({variance['Explained variance'].iloc[0]:.1%})")
        axis.set_ylabel(f"PC2 ({variance['Explained variance'].iloc[1]:.1%})")
        axis.set_title(method, fontsize=10)
        axis.legend(fontsize=8.5, loc="upper right")

    figure.suptitle(
        "The same 75 games' worth of flags, chosen three different ways",
        fontsize=13, fontweight="semibold", y=1.02,
    )
    figure.tight_layout(); plt.close(figure)
    return figure


def plot_method_agreement(agreement: pd.DataFrame, counts: np.ndarray) -> plt.Figure:
    """Pairwise agreement between methods, and how many methods flagged each game."""
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    sns.heatmap(agreement, annot=True, fmt=".3f", cmap=SEQUENTIAL_COLOURMAP, vmin=0, vmax=1,
                square=True, linewidths=2, linecolor="white",
                cbar_kws={"label": "Jaccard overlap", "shrink": 0.8},
                annot_kws={"fontsize": 10}, ax=axes[0])
    axes[0].set_title("Pairwise agreement between methods")
    axes[0].grid(False)
    axes[0].tick_params(axis="x", rotation=20)

    distribution = pd.Series(counts).value_counts().sort_index()
    flagged = distribution.drop(0, errors="ignore")
    colours = [ACCENT_COLOUR if n == 1 else PRIMARY_COLOUR for n in flagged.index]
    bars = axes[1].bar(flagged.index.astype(str), flagged.to_numpy(), 0.55, color=colours,
                       edgecolor="white", linewidth=1.5)
    for bar, value in zip(bars, flagged.to_numpy()):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value}",
                     ha="center", va="bottom", fontsize=10, color=INK_SECONDARY)
    axes[1].set_xlabel("Number of methods that flagged the game")
    axes[1].set_ylabel("Games")
    axes[1].set_title("Most flagged games are flagged by only one method")

    figure.suptitle("How much do the three methods agree?", fontsize=13,
                    fontweight="semibold", y=1.02)
    figure.tight_layout(); plt.close(figure)
    return figure


def plot_sensitivity_curves(dimensionality: pd.DataFrame, noise: pd.DataFrame) -> plt.Figure:
    """Flag-set survival as dimensions and noise are added."""
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    colours = ["#2a78d6", "#eb6834", "#1baf7a"]
    methods = [c for c in dimensionality.columns if c != "Total dimensions"]

    for colour, method in zip(colours, methods):
        axes[0].plot(dimensionality["Total dimensions"], dimensionality[method],
                     marker="o", markersize=5, color=colour, label=method)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Total dimensions (log scale)")
    axes[0].set_ylabel("Jaccard with the 8-dimensional result")
    axes[0].set_title("Adding pure-noise dimensions")
    axes[0].set_ylim(0, 1.05); axes[0].legend(fontsize=9)

    for colour, method in zip(colours, noise.columns):
        axes[1].plot(noise.index, noise[method], marker="o", markersize=5,
                     color=colour, label=method)
    axes[1].set_xlabel("Noise added to every value (standard deviations)")
    axes[1].set_ylabel("Jaccard with the unperturbed result")
    axes[1].set_title("Perturbing the data")
    axes[1].set_ylim(0, 1.05); axes[1].legend(fontsize=9)

    figure.suptitle("How stable is each method's answer?", fontsize=13,
                    fontweight="semibold", y=1.02)
    figure.tight_layout(); plt.close(figure)
    return figure


def plot_runtime(runtime: pd.DataFrame) -> plt.Figure:
    """Fit-and-score time against dataset size."""
    figure, axis = plt.subplots(figsize=(9, 4.8))
    colours = ["#2a78d6", "#eb6834", "#1baf7a"]
    for colour, method in zip(colours, runtime.columns):
        axis.plot(runtime.index, runtime[method].clip(lower=1e-4), marker="o",
                  markersize=5, color=colour, label=method)
    axis.set_xscale("log"); axis.set_yscale("log")
    axis.set_xlabel("Rows (log scale)")
    axis.set_ylabel("Seconds to fit and score (log scale)")
    axis.set_title("Runtime, at eight dimensions")
    axis.legend(fontsize=9)
    figure.tight_layout(); plt.close(figure)
    return figure
