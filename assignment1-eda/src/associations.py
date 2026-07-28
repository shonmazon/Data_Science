"""Measures of association between variables.

Correlation is reported by three methods side by side rather than one, because
section 6.1 is about the ways they disagree. For categorical pairs, Cramér's V
is used with the bias correction, since several categories here are small.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

CORRELATION_METHODS = ["pearson", "spearman", "kendall"]

# Price points in this market cluster on a few psychological values, so fixed
# boundaries are more interpretable than quantiles. The lowest band isolates
# free-to-play, which section 4.3 showed is a different business model.
PRICE_BAND_EDGES = [-0.01, 0.0, 9.99, 19.99, 39.99, 100.0]
PRICE_BAND_LABELS = ["Free", "$0.01-9.99", "$10-19.99", "$20-39.99", "$40+"]

REVIEW_BAND_EDGES = [0, 70, 80, 90, 100]
REVIEW_BAND_LABELS = ["<=70", "70-80", "80-90", "90-100"]


def add_analysis_features(
    dataframe: pd.DataFrame, snapshot_date: pd.Timestamp
) -> pd.DataFrame:
    """Derive the columns section 6 needs, including the binned categoricals.

    Binning is required to relate numeric variables to categorical ones. Where
    a variable has meaningful natural boundaries, such as price, fixed edges are
    used; where it does not, quartiles are used instead.

    Args:
        dataframe: The analysis-ready dataset.
        snapshot_date: Date the data was extracted, from section 3.1.

    Returns:
        A copy of the frame with the derived columns appended.
    """
    featured = dataframe.copy()

    featured["daysOnSale"] = (snapshot_date - featured["releaseDate"]).dt.days
    featured["logRevenue"] = np.log10(featured["revenue"])
    featured["logCopiesSold"] = np.log10(featured["copiesSold"])

    # A missing publisher means the game is self-published rather than unknown,
    # as established in section 4.1, so it is compared as an ordinary value.
    featured["selfPublished"] = featured["publishers"].fillna("") == featured[
        "developers"
    ].fillna("")
    featured["isFreeToPlay"] = featured["price"] == 0

    featured["priceBand"] = pd.cut(
        featured["price"], bins=PRICE_BAND_EDGES, labels=PRICE_BAND_LABELS
    )
    featured["reviewBand"] = pd.cut(
        featured["reviewScore"], bins=REVIEW_BAND_EDGES, labels=REVIEW_BAND_LABELS
    )
    featured["revenueQuartile"] = pd.qcut(
        featured["revenue"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"]
    )
    featured["playtimeQuartile"] = pd.qcut(
        featured["avgPlaytime"], 4, labels=["P1 shortest", "P2", "P3", "P4 longest"]
    )
    featured["releaseQuarter"] = "Q" + featured["releaseDate"].dt.quarter.astype(str)

    # Hobbyist holds a single game, which cannot support a category of its own.
    # Section 5.2 argued for folding it into Indie for any grouped analysis,
    # while publisherClass keeps the original four levels.
    featured["plotClass"] = pd.Categorical(
        featured["publisherClass"].astype(str).replace({"Hobbyist": "Indie"}),
        categories=["Indie", "AA", "AAA"],
        ordered=True,
    )

    return featured


def compare_correlation_methods(
    dataframe: pd.DataFrame, pairs: list[tuple[str, str]]
) -> pd.DataFrame:
    """Report all three correlation coefficients for chosen variable pairs.

    Args:
        dataframe: The dataset.
        pairs: Column pairs to correlate.

    Returns:
        One row per pair with the Pearson, Spearman and Kendall coefficients
        and the gap between the first two.
    """
    rows = []
    for first, second in pairs:
        coefficients = {
            method: dataframe[first].corr(dataframe[second], method=method)
            for method in CORRELATION_METHODS
        }
        rows.append(
            {
                "Variable pair": f"{first} vs {second}",
                "Pearson": round(coefficients["pearson"], 3),
                "Spearman": round(coefficients["spearman"], 3),
                "Kendall": round(coefficients["kendall"], 3),
                "Spearman - Pearson": round(
                    coefficients["spearman"] - coefficients["pearson"], 3
                ),
            }
        )
    return pd.DataFrame(rows)


def cramers_v(first: pd.Series, second: pd.Series) -> float:
    """Measure association between two categorical variables, from 0 to 1.

    The bias-corrected form is used because several categories in this dataset
    are small, and the uncorrected statistic is inflated when a contingency
    table has many sparse cells.

    Args:
        first: A categorical series.
        second: A categorical series of the same length.

    Returns:
        The corrected Cramér's V, or NaN when the table is degenerate.
    """
    contingency = pd.crosstab(first, second)
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return np.nan

    chi_squared = chi2_contingency(contingency)[0]
    observations = contingency.to_numpy().sum()
    rows, columns = contingency.shape

    phi_squared = chi_squared / observations
    corrected_phi_squared = max(
        0.0, phi_squared - ((columns - 1) * (rows - 1)) / (observations - 1)
    )
    corrected_rows = rows - ((rows - 1) ** 2) / (observations - 1)
    corrected_columns = columns - ((columns - 1) ** 2) / (observations - 1)

    denominator = min(corrected_columns - 1, corrected_rows - 1)
    return np.sqrt(corrected_phi_squared / denominator) if denominator > 0 else np.nan


def cramers_v_matrix(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Build a square matrix of Cramér's V for every pair of categorical columns.

    Args:
        dataframe: The dataset.
        columns: Categorical columns to cross.

    Returns:
        A symmetric matrix indexed and columned by `columns`.
    """
    matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    for first in columns:
        for second in columns:
            matrix.loc[first, second] = (
                1.0
                if first == second
                else cramers_v(
                    dataframe[first].astype(str), dataframe[second].astype(str)
                )
            )
    return matrix


def summarise_by_category(
    dataframe: pd.DataFrame, category: str, value: str
) -> pd.DataFrame:
    """Describe how a numeric variable behaves within each level of a category.

    Args:
        dataframe: The dataset.
        category: The grouping column.
        value: The numeric column to summarise.

    Returns:
        One row per category level with counts and robust summary statistics.
    """
    grouped = dataframe.groupby(category, observed=True)[value]
    summary = pd.DataFrame(
        {
            "Games": grouped.size(),
            "Median": grouped.median().round(2),
            "Mean": grouped.mean().round(2),
            "Total": grouped.sum().round(0),
        }
    )
    summary["Share of total"] = (
        summary["Total"] / summary["Total"].sum() * 100
    ).round(2)
    return summary
