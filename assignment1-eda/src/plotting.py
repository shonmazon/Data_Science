"""Chart styling and figure builders.

All figures are produced here rather than inline in the notebook, so that every
chart shares one visual language and no styling decision is repeated. Colours
come from a fixed, pre-validated categorical palette and are assigned to
entities in a fixed order, never cycled by rank.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import probplot

# Categorical slots, in fixed assignment order. Only the first three are used
# for any chart where every pair of colours must be distinguishable at once
# (scatter, pairplot); that three-slot set is validated for all pairs under
# colour-vision deficiency.
CATEGORICAL_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]

# Single hue for one-series charts, and the two poles of the diverging scale.
PRIMARY_COLOUR = "#2a78d6"
NEUTRAL_MIDPOINT = "#e8e8e6"
DIVERGING_POLES = ("#2a78d6", "#eb6834")

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID_COLOUR = "#dcdcd8"

# Publisher classes in ascending order of studio scale. Hobbyist is folded into
# Indie for grouped charts, as argued in section 5.2: a single observation
# cannot support a category of its own.
PUBLISHER_CLASS_ORDER = ["Indie", "AA", "AAA"]

SEQUENTIAL_COLOURMAP = LinearSegmentedColormap.from_list(
    "house_sequential", ["#f5f8fd", PRIMARY_COLOUR]
)
DIVERGING_COLOURMAP = LinearSegmentedColormap.from_list(
    "house_diverging", [DIVERGING_POLES[0], NEUTRAL_MIDPOINT, DIVERGING_POLES[1]]
)


def apply_house_style() -> None:
    """Set the shared matplotlib defaults: thin marks and a recessive grid."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID_COLOUR,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID_COLOUR,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "font.size": 10,
        }
    )


def _label_thousands(value: float, _position: int) -> str:
    """Format an axis tick as a compact currency-free magnitude."""
    for threshold, suffix in [(1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(value) >= threshold:
            return f"{value / threshold:.0f}{suffix}"
    return f"{value:.0f}"


def _class_palette(categories: list[str]) -> dict[str, str]:
    """Map category names to colour slots in fixed order."""
    return dict(zip(categories, CATEGORICAL_COLOURS))


def plot_revenue_distribution(games: pd.DataFrame) -> plt.Figure:
    """Histogram of revenue on the raw and logarithmic scale, side by side."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(games["revenue"], bins=60, color=PRIMARY_COLOUR)
    axes[0].set_title("Revenue is unreadable on a linear scale")
    axes[0].set_xlabel("Revenue (USD)")
    axes[0].set_ylabel("Number of games")
    axes[0].xaxis.set_major_formatter(mpl.ticker.FuncFormatter(_label_thousands))

    axes[1].hist(np.log10(games["revenue"]), bins=60, color=PRIMARY_COLOUR)
    axes[1].set_title("On a log scale its shape becomes readable")
    axes[1].set_xlabel("log$_{10}$ revenue (USD)")
    axes[1].set_ylabel("Number of games")

    figure.suptitle(
        "Distribution of game revenue, 1,500 top-earning Steam titles of 2024",
        fontsize=13,
        fontweight="semibold",
        y=1.04,
    )
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_copies_versus_revenue(games: pd.DataFrame) -> plt.Figure:
    """Scatter of copies sold against revenue, on log axes, split by publisher class."""
    figure, axis = plt.subplots(figsize=(8, 5.5))
    palette = _class_palette(PUBLISHER_CLASS_ORDER)

    for publisher_class in PUBLISHER_CLASS_ORDER:
        subset = games[games["plotClass"] == publisher_class]
        axis.scatter(
            subset["copiesSold"],
            subset["revenue"],
            s=18,
            alpha=0.65,
            color=palette[publisher_class],
            edgecolor="white",
            linewidth=0.4,
            label=f"{publisher_class} (n={len(subset):,})",
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Copies sold (log scale)")
    axis.set_ylabel("Revenue in USD (log scale)")
    axis.set_title("Copies sold against revenue, by publisher class")
    axis.legend(title="Publisher class", loc="upper left")
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_median_revenue_by_class(games: pd.DataFrame) -> plt.Figure:
    """Bar chart of median revenue per publisher class, with direct labels."""
    medians = (
        games.groupby("plotClass", observed=True)["revenue"]
        .median()
        .reindex(PUBLISHER_CLASS_ORDER)
    )

    figure, axis = plt.subplots(figsize=(7, 4.2))
    bars = axis.bar(medians.index, medians.to_numpy(), color=PRIMARY_COLOUR, width=0.55)

    for bar, value in zip(bars, medians.to_numpy()):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"${value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK_SECONDARY,
        )

    axis.set_xlabel("Publisher class")
    axis.set_ylabel("Median revenue (USD)")
    axis.set_title("Median revenue rises tenfold from Indie to AAA")
    axis.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_label_thousands))
    axis.margins(y=0.15)
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_revenue_box_by_class(games: pd.DataFrame) -> plt.Figure:
    """Box plot of log revenue by publisher class."""
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    sns.boxplot(
        data=games,
        x="plotClass",
        y="logRevenue",
        order=PUBLISHER_CLASS_ORDER,
        color=PRIMARY_COLOUR,
        width=0.45,
        fliersize=2.5,
        linewidth=1.1,
        ax=axis,
    )
    axis.set_xlabel("Publisher class")
    axis.set_ylabel("log$_{10}$ revenue (USD)")
    axis.set_title("Revenue spread within each publisher class")
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_playtime_violin_by_price(games: pd.DataFrame) -> plt.Figure:
    """Violin plot of playtime across price bands."""
    figure, axis = plt.subplots(figsize=(9, 4.8))
    sns.violinplot(
        data=games.dropna(subset=["avgPlaytime", "priceBand"]),
        x="priceBand",
        y="avgPlaytime",
        color=PRIMARY_COLOUR,
        cut=0,
        linewidth=1.0,
        ax=axis,
    )
    axis.set_yscale("log")
    axis.set_xlabel("Price band")
    axis.set_ylabel("Average playtime in hours (log scale)")
    axis.set_title("Costlier games are played for longer, with free-to-play the exception")
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_share_comparison_pies(games: pd.DataFrame) -> plt.Figure:
    """Two pies contrasting each class's share of games with its share of revenue."""
    grouped = games.groupby("plotClass", observed=True)
    share_of_games = grouped.size().reindex(PUBLISHER_CLASS_ORDER)
    share_of_revenue = grouped["revenue"].sum().reindex(PUBLISHER_CLASS_ORDER)
    colours = [_class_palette(PUBLISHER_CLASS_ORDER)[name] for name in PUBLISHER_CLASS_ORDER]

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for axis, values, title in [
        (axes[0], share_of_games, "Share of games"),
        (axes[1], share_of_revenue, "Share of revenue"),
    ]:
        axis.pie(
            values.to_numpy(),
            labels=values.index,
            colors=colours,
            autopct="%1.1f%%",
            startangle=90,
            counterclock=False,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
            textprops={"color": INK_PRIMARY, "fontsize": 10},
        )
        axis.set_title(title)
        axis.grid(False)

    figure.suptitle(
        "Indie studios make most of the games and take the smallest share of the money",
        fontsize=13,
        fontweight="semibold",
        y=1.02,
    )
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_numeric_pairplot(games: pd.DataFrame) -> sns.PairGrid:
    """Pairplot of the log-scaled numeric variables, split by publisher class."""
    columns = ["logRevenue", "logCopiesSold", "price", "avgPlaytime", "reviewScore"]
    grid = sns.pairplot(
        games.dropna(subset=columns + ["plotClass"]),
        vars=columns,
        hue="plotClass",
        hue_order=PUBLISHER_CLASS_ORDER,
        palette=CATEGORICAL_COLOURS,
        corner=True,
        plot_kws={"s": 10, "alpha": 0.5, "linewidth": 0},
        diag_kws={"common_norm": False},
        height=1.9,
    )
    grid.figure.suptitle(
        "Pairwise relationships between the numeric variables", y=1.01, fontsize=13,
        fontweight="semibold",
    )
    grid.legend.set_title("Publisher class")
    plt.close(grid.figure)
    return grid


def plot_correlation_heatmap(matrix: pd.DataFrame, title: str) -> plt.Figure:
    """Heatmap of a correlation matrix on a diverging scale centred at zero."""
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap=DIVERGING_COLOURMAP,
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=2,
        linecolor="white",
        cbar_kws={"label": "Correlation coefficient", "shrink": 0.8},
        annot_kws={"fontsize": 8.5},
        ax=axis,
    )
    axis.set_title(title)
    axis.grid(False)
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_association_heatmap(matrix: pd.DataFrame, title: str) -> plt.Figure:
    """Heatmap of Cramér's V on a single-hue sequential scale from 0 to 1."""
    figure, axis = plt.subplots(figsize=(7.6, 6.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap=SEQUENTIAL_COLOURMAP,
        vmin=0,
        vmax=1,
        square=True,
        linewidths=2,
        linecolor="white",
        cbar_kws={"label": "Cramér's V", "shrink": 0.8},
        annot_kws={"fontsize": 8.5},
        ax=axis,
    )
    axis.set_title(title)
    axis.grid(False)
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_release_calendar(games: pd.DataFrame) -> plt.Figure:
    """Releases per ISO week, with the quietest week emphasised."""
    weekly_counts = games.groupby(
        games["releaseDate"].dt.isocalendar().week, observed=True
    ).size()
    quietest_week = weekly_counts.idxmin()

    # Emphasis, not a value ramp: one bar is singled out because the narrative
    # is about that week, and every other bar keeps the single series colour.
    colours = [
        DIVERGING_POLES[1] if week == quietest_week else PRIMARY_COLOUR
        for week in weekly_counts.index
    ]

    figure, axis = plt.subplots(figsize=(10, 4))
    axis.bar(weekly_counts.index, weekly_counts.to_numpy(), color=colours, width=0.75)
    axis.annotate(
        f"Week {quietest_week}: {weekly_counts.min()} releases",
        xy=(quietest_week, weekly_counts.min()),
        xytext=(quietest_week + 2, weekly_counts.max() * 1.1),
        color=INK_SECONDARY,
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": INK_SECONDARY, "linewidth": 1},
    )
    axis.set_ylim(0, weekly_counts.max() * 1.3)
    axis.set_xlabel("ISO week of 2024")
    axis.set_ylabel("Games released")
    axis.set_title("Releases per week, showing a near-total gap in early July")
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_revenue_per_day_trend(games: pd.DataFrame) -> plt.Figure:
    """Median revenue per day of exposure, by release month, on a log axis."""
    monthly = games.groupby(
        games["releaseDate"].dt.to_period("M"), observed=True
    )["revenuePerDay"].median()

    figure, axis = plt.subplots(figsize=(9, 4.2))
    axis.bar(monthly.index.astype(str), monthly.to_numpy(), color=PRIMARY_COLOUR, width=0.6)
    axis.set_yscale("log")
    axis.set_xlabel("Release month")
    axis.set_ylabel("Median revenue per day on sale (USD, log scale)")
    axis.set_title("Normalising by exposure exposes the selection gradient")
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_monthly_revenue_trend(games: pd.DataFrame) -> plt.Figure:
    """Median revenue and median days on sale by release month, as small multiples."""
    monthly = games.groupby(games["releaseDate"].dt.to_period("M"), observed=True).agg(
        median_revenue=("revenue", "median"),
        median_days=("daysOnSale", "median"),
        games=("revenue", "size"),
    )
    months = monthly.index.astype(str)

    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].bar(months, monthly["median_revenue"], color=PRIMARY_COLOUR, width=0.6)
    axes[0].set_ylabel("Median revenue (USD)")
    axes[0].set_title("Median revenue barely moves across release months")
    axes[0].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(_label_thousands))

    axes[1].bar(months, monthly["median_days"], color=DIVERGING_POLES[1], width=0.6)
    axes[1].set_ylabel("Median days on sale")
    axes[1].set_xlabel("Release month")
    axes[1].set_title("...even though time on sale collapses from 236 days to 7")

    figure.suptitle(
        "The exposure effect is absent, which is itself evidence of selection",
        fontsize=13,
        fontweight="semibold",
        y=1.0,
    )
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_review_score_distribution(raw_games: pd.DataFrame) -> plt.Figure:
    """Histogram of the raw review scores, before the placeholders are removed.

    Section 4.3 argues from a frequency table that a stored 0 is a placeholder
    rather than a rating. The histogram shows the same evidence directly: a
    spike at zero separated from the real distribution by an almost empty gap.
    """
    figure, axis = plt.subplots(figsize=(9, 4.4))

    axis.hist(raw_games["reviewScore"], bins=range(0, 102, 2), color=PRIMARY_COLOUR)
    axis.axvspan(1, 30, color=DIVERGING_POLES[1], alpha=0.12)
    axis.annotate(
        "99 games at exactly 0",
        xy=(0, (raw_games["reviewScore"] == 0).sum()),
        xytext=(14, (raw_games["reviewScore"] == 0).sum() * 0.92),
        color=INK_SECONDARY,
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": INK_SECONDARY, "linewidth": 1},
    )
    axis.annotate(
        "only 6 games score 1-30",
        xy=(15, 6),
        xytext=(33, (raw_games["reviewScore"] == 0).sum() * 0.55),
        color=INK_SECONDARY,
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": INK_SECONDARY, "linewidth": 1},
    )

    axis.set_xlabel("Review score (percentage of positive reviews)")
    axis.set_ylabel("Number of games")
    axis.set_title("A spike at zero, separated from the real distribution by a gap")
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_numeric_spread(games: pd.DataFrame) -> plt.Figure:
    """Box plots of every numeric variable, as small multiples.

    Each variable keeps its own panel and its own axis, because their units and
    magnitudes are not comparable. The three heavy-tailed variables are drawn on
    a logarithmic scale, which is the only way their boxes remain visible.
    """
    panels = [
        ("revenue", "Revenue (USD)", True),
        ("copiesSold", "Copies sold", True),
        ("avgPlaytime", "Average playtime (hours)", True),
        ("price", "Price (USD)", False),
        ("reviewScore", "Review score", False),
    ]

    figure, axes = plt.subplots(1, len(panels), figsize=(12, 4.2))
    for axis, (column, label, use_log) in zip(axes, panels):
        axis.boxplot(
            games[column].dropna(),
            widths=0.45,
            patch_artist=True,
            boxprops={"facecolor": PRIMARY_COLOUR, "edgecolor": INK_SECONDARY, "linewidth": 0.9},
            medianprops={"color": "white", "linewidth": 1.4},
            whiskerprops={"color": INK_SECONDARY, "linewidth": 0.9},
            capprops={"color": INK_SECONDARY, "linewidth": 0.9},
            flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": INK_SECONDARY,
                        "markeredgecolor": "none", "alpha": 0.4},
        )
        if use_log:
            axis.set_yscale("log")
        axis.set_title(label, fontsize=10)
        axis.set_xticks([])

    figure.suptitle(
        "Spread and outliers of each numeric variable, on independent axes",
        fontsize=13,
        fontweight="semibold",
        y=1.03,
    )
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_playtime_versus_revenue(games: pd.DataFrame) -> plt.Figure:
    """Scatter of average playtime against revenue, both on logarithmic axes.

    This is the pair where Pearson and Spearman disagree most sharply in section
    6.1, and the plot shows why: the relationship is real but curved, so a
    coefficient that assumes a straight line finds almost nothing.
    """
    plotted = games.dropna(subset=["avgPlaytime", "revenue"])

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(
        plotted["avgPlaytime"],
        plotted["revenue"],
        s=16,
        alpha=0.5,
        color=PRIMARY_COLOUR,
        edgecolor="white",
        linewidth=0.3,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Average playtime in hours (log scale)")
    axis.set_ylabel("Revenue in USD (log scale)")
    axis.set_title("Playtime against revenue: Pearson 0.08, Spearman 0.44")
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_revenue_by_row_order(games: pd.DataFrame, block_starts: list[int]) -> plt.Figure:
    """Scatter of revenue against position in the file.

    Section 7 establishes from a table that the file is four concatenated
    sorted extracts. Plotting revenue against row number makes the four
    descending runs, and the block of top earners stored last, visible at once.
    """
    figure, axis = plt.subplots(figsize=(10, 4.6))
    axis.scatter(
        games.index,
        games["revenue"],
        s=7,
        alpha=0.55,
        color=PRIMARY_COLOUR,
        edgecolor="none",
    )
    axis.set_yscale("log")

    for boundary in block_starts[1:]:
        axis.axvline(boundary, color=DIVERGING_POLES[1], linewidth=1.1)

    labels = ["A", "B", "C", "D"]
    edges = block_starts + [len(games)]
    for name, start, end in zip(labels, edges[:-1], edges[1:]):
        axis.text(
            (start + end) / 2,
            games["revenue"].max() * 1.6,
            name,
            ha="center",
            fontsize=10,
            color=INK_SECONDARY,
        )

    axis.set_ylim(top=games["revenue"].max() * 4)
    axis.set_xlabel("Row position in the file")
    axis.set_ylabel("Revenue in USD (log scale)")
    axis.set_title("Four separately sorted blocks, with the top earners stored last")
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_release_calendar_heatmap(games: pd.DataFrame) -> plt.Figure:
    """Heatmap of release counts by month and weekday.

    Section 9.2 reports the weekday concentration as a single table. Crossing it
    with the month shows whether the convention holds all year or is driven by a
    few unusual weeks.
    """
    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    ]
    counts = pd.crosstab(
        games["releaseDate"].dt.strftime("%Y-%m"),
        games["releaseDate"].dt.day_name(),
    ).reindex(columns=weekday_order)

    figure, axis = plt.subplots(figsize=(8.5, 5))
    sns.heatmap(
        counts,
        annot=True,
        fmt="d",
        cmap=SEQUENTIAL_COLOURMAP,
        linewidths=2,
        linecolor="white",
        cbar_kws={"label": "Games released", "shrink": 0.8},
        annot_kws={"fontsize": 8.5},
        ax=axis,
    )
    axis.set_xlabel("Day of the week")
    axis.set_ylabel("Release month")
    axis.set_title("Release counts by month and weekday")
    axis.grid(False)
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_revenue_qq(games: pd.DataFrame) -> plt.Figure:
    """Normal probability plots of revenue before and after a log transform.

    A Q-Q plot compares the observed quantiles against those of a normal
    distribution: points on the diagonal mean the two agree. It tests directly
    the claim made in section 5.1, that logarithms make revenue tractable
    without making it normal.
    """
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))

    for axis, values, label in [
        (axes[0], games["revenue"], "Revenue, raw scale"),
        (axes[1], np.log10(games["revenue"]), "Revenue, log$_{10}$ scale"),
    ]:
        probplot(values.dropna(), dist="norm", plot=axis)
        axis.get_lines()[0].set(marker="o", markersize=2.5, color=PRIMARY_COLOUR, alpha=0.5)
        axis.get_lines()[1].set(color=DIVERGING_POLES[1], linewidth=1.4)
        axis.set_title(label)
        axis.set_xlabel("Theoretical normal quantiles")
        axis.set_ylabel("Observed quantiles")

    figure.suptitle(
        "Normal probability plots: the log transform straightens most, but not all, of the curve",
        fontsize=13,
        fontweight="semibold",
        y=1.03,
    )
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_revenue_lorenz_curve(games: pd.DataFrame) -> plt.Figure:
    """Lorenz curve of revenue, with the Gini coefficient.

    Section 5 reports the concentration of revenue as a list of percentages.
    The Lorenz curve shows the whole distribution of that concentration at
    once: the further the curve sags below the diagonal, the more unequal the
    share of revenue across games.
    """
    ordered = games["revenue"].sort_values().to_numpy()
    cumulative_revenue = np.concatenate([[0.0], ordered.cumsum() / ordered.sum()])
    cumulative_games = np.linspace(0.0, 1.0, len(cumulative_revenue))

    # Gini is twice the area between the diagonal and the curve.
    gini = 1 - 2 * np.trapezoid(cumulative_revenue, cumulative_games)

    figure, axis = plt.subplots(figsize=(6.4, 6))
    axis.plot([0, 1], [0, 1], color=INK_SECONDARY, linewidth=1.2, label="Perfect equality")
    axis.plot(cumulative_games, cumulative_revenue, color=PRIMARY_COLOUR, linewidth=2.2,
              label=f"Observed (Gini = {gini:.2f})")
    axis.fill_between(cumulative_games, cumulative_revenue, cumulative_games,
                      color=PRIMARY_COLOUR, alpha=0.10)

    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Cumulative share of games, poorest earning first")
    axis.set_ylabel("Cumulative share of revenue")
    axis.set_title("Lorenz curve of revenue across the 1,500 games")
    axis.legend(loc="upper left")
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_copies_revenue_hexbin(games: pd.DataFrame) -> plt.Figure:
    """Hexagonal density of copies sold against revenue.

    The scatter plot of the same two variables draws 1,500 semi-transparent
    points, which hides how much of the data sits in the dense core. Binning
    the plane and colouring by count shows the concentration that overplotting
    conceals, which is the same pair of variables seen a different way.
    """
    figure, axis = plt.subplots(figsize=(8, 5.4))
    mesh = axis.hexbin(
        np.log10(games["copiesSold"]),
        np.log10(games["revenue"]),
        gridsize=32,
        cmap=SEQUENTIAL_COLOURMAP,
        mincnt=1,
        linewidths=0.2,
        edgecolors="white",
    )
    colour_bar = figure.colorbar(mesh, ax=axis, shrink=0.85)
    colour_bar.set_label("Games in bin")

    axis.set_xlabel("log$_{10}$ copies sold")
    axis.set_ylabel("log$_{10}$ revenue (USD)")
    axis.set_title("Where the games actually sit, which the scatter plot obscures")
    axis.grid(False)
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_class_composition_by_price(games: pd.DataFrame) -> plt.Figure:
    """Stacked bars showing the publisher-class mix inside each price band.

    Section 6.2 measures the association between class and price with Cramer's
    V and a contingency table. Normalising each band to 100% shows the shape of
    that association: which kind of studio occupies which part of the price
    range.
    """
    composition = pd.crosstab(games["priceBand"], games["plotClass"], normalize="index") * 100
    composition = composition.reindex(columns=PUBLISHER_CLASS_ORDER)

    figure, axis = plt.subplots(figsize=(9, 4.8))
    running_total = np.zeros(len(composition))
    for publisher_class, colour in zip(PUBLISHER_CLASS_ORDER, CATEGORICAL_COLOURS):
        values = composition[publisher_class].to_numpy()
        axis.bar(
            composition.index.astype(str),
            values,
            bottom=running_total,
            color=colour,
            width=0.62,
            label=publisher_class,
            edgecolor="white",
            linewidth=2,
        )
        running_total += values

    axis.set_ylim(0, 100)
    axis.set_xlabel("Price band")
    axis.set_ylabel("Share of games in the band (%)")
    axis.set_title("Publisher class composition within each price band")
    axis.legend(title="Publisher class", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    figure.tight_layout()
    plt.close(figure)
    return figure


def plot_monthly_stability_lines(games: pd.DataFrame) -> plt.Figure:
    """Line chart of four monthly indicators, each indexed to its January value.

    Section 7 asks whether the analysis changes over time and answers with a
    table of quarterly figures. Indexing each series to 100 at January puts four
    quantities with different units on a single axis without distorting any of
    them, so drift in any one of them would be immediately visible.
    """
    monthly = games.groupby(games["releaseDate"].dt.to_period("M"), observed=True).agg(
        median_revenue=("revenue", "median"),
        median_price=("price", "median"),
        median_review=("reviewScore", "median"),
        indie_share=("plotClass", lambda column: (column == "Indie").mean()),
    )
    indexed = monthly / monthly.iloc[0] * 100
    months = monthly.index.astype(str)

    series_labels = {
        "median_revenue": "Median revenue",
        "median_price": "Median price",
        "median_review": "Median review score",
        "indie_share": "Indie share of releases",
    }

    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    for (column, label), colour in zip(series_labels.items(), CATEGORICAL_COLOURS + ["#eda100"]):
        axis.plot(months, indexed[column], color=colour, marker="o", markersize=4, label=label)

    axis.axhline(100, color=INK_SECONDARY, linewidth=1.0)
    axis.set_xlabel("Release month")
    axis.set_ylabel("Value indexed to January = 100")
    axis.set_title("Quality and composition hold flat; median revenue is noisier")
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    figure.tight_layout()
    plt.close(figure)
    return figure
