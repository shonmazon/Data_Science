"""Single-variable summaries for numeric and categorical columns.

The outlier detectors are kept as three separate functions returning boolean
masks rather than one function with a mode argument, because the point of
section 5.1 is to compare what the three methods disagree about.
"""

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

# Scaling constant that puts the median absolute deviation on the same footing
# as a standard deviation for normally distributed data.
MODIFIED_Z_SCALE = 0.6745


def describe_numeric(dataframe: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Produce the full set of location, spread and shape statistics.

    Both a classical and a robust measure is reported for location (mean and
    median) and for spread (standard deviation and MAD), so the gap between
    them can be read directly off the table.

    Args:
        dataframe: The analysis-ready dataset.
        columns: Numeric columns to summarise.

    Returns:
        One row per column.
    """
    rows = []
    for column in columns:
        values = dataframe[column].dropna()
        first_quartile, third_quartile = values.quantile([0.25, 0.75])
        rows.append(
            {
                "Column": column,
                "Count": len(values),
                "Mean": values.mean(),
                "Median": values.median(),
                "Std": values.std(),
                "MAD": median_abs_deviation(values),
                "Min": values.min(),
                "Q1": first_quartile,
                "Q3": third_quartile,
                "Max": values.max(),
                "IQR": third_quartile - first_quartile,
                "Skew": values.skew(),
            }
        )
    return pd.DataFrame(rows).set_index("Column")


def flag_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Flag values outside Tukey's fences at Q1 - k*IQR and Q3 + k*IQR."""
    first_quartile, third_quartile = series.quantile([0.25, 0.75])
    spread = third_quartile - first_quartile
    lower = first_quartile - multiplier * spread
    upper = third_quartile + multiplier * spread
    return (series < lower) | (series > upper)


def flag_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Flag values more than `threshold` standard deviations from the mean."""
    standard_deviation = series.std()
    if standard_deviation == 0:
        return pd.Series(False, index=series.index)
    return ((series - series.mean()).abs() / standard_deviation) > threshold


def flag_outliers_modified_zscore(
    series: pd.Series, threshold: float = 3.5
) -> pd.Series:
    """Flag values far from the median relative to the MAD.

    This is the robust counterpart of the z-score: because it is built on the
    median and the MAD, the outliers themselves cannot inflate the scale
    against which they are judged.
    """
    deviation = median_abs_deviation(series.dropna())
    if deviation == 0:
        return pd.Series(False, index=series.index)
    scores = MODIFIED_Z_SCALE * (series - series.median()) / deviation
    return scores.abs() > threshold


def compare_outlier_methods(
    dataframe: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Count how many values each detector flags in each column.

    Args:
        dataframe: The analysis-ready dataset.
        columns: Numeric columns to test.

    Returns:
        One row per column with the count flagged by each of the three methods.
    """
    rows = []
    for column in columns:
        values = dataframe[column].dropna()
        rows.append(
            {
                "Column": column,
                "Values": len(values),
                "IQR (1.5x)": int(flag_outliers_iqr(values).sum()),
                "Z-score (|z|>3)": int(flag_outliers_zscore(values).sum()),
                "Modified Z (|M|>3.5)": int(flag_outliers_modified_zscore(values).sum()),
            }
        )
    summary = pd.DataFrame(rows)
    for method in ["IQR (1.5x)", "Z-score (|z|>3)", "Modified Z (|M|>3.5)"]:
        summary[f"{method} %"] = (summary[method] / summary["Values"] * 100).round(1)
    return summary.set_index("Column")


def summarise_categorical(series: pd.Series, top_n: int = 10) -> pd.DataFrame:
    """Tabulate the most frequent values with their cumulative coverage.

    Args:
        series: A categorical column, already expanded if it holds lists.
        top_n: How many of the most frequent values to show.

    Returns:
        One row per value, with its count, share and running cumulative share.
    """
    counts = series.value_counts()
    table = pd.DataFrame(
        {
            "Value": counts.index[:top_n],
            "Count": counts.to_numpy()[:top_n],
            "Share": (counts.to_numpy()[:top_n] / counts.sum() * 100).round(2),
        }
    )
    table["Cumulative share"] = table["Share"].cumsum().round(2)
    return table


def values_needed_for_coverage(
    series: pd.Series, targets: tuple[float, ...] = (0.25, 0.50, 0.80, 0.90)
) -> pd.DataFrame:
    """Find the smallest number of categories that together cover a share of the data.

    Args:
        series: A categorical column, already expanded if it holds lists.
        targets: Coverage levels to report, as fractions.

    Returns:
        One row per target, with the number of distinct values required and the
        share of all distinct values that represents.
    """
    counts = series.value_counts()
    cumulative_share = counts.cumsum() / counts.sum()
    distinct_total = len(counts)

    rows = []
    for target in targets:
        needed = int((cumulative_share < target).sum() + 1)
        rows.append(
            {
                "Coverage target": f"{target:.0%}",
                "Categories needed": needed,
                "Share of all categories": f"{needed / distinct_total:.1%}",
            }
        )
    return pd.DataFrame(rows)


def expand_entity_column(dataframe: pd.DataFrame, column: str) -> pd.Series:
    """Flatten a column of delimited company lists into one company per row.

    Args:
        dataframe: The dataset.
        column: Name of the column holding delimited lists.

    Returns:
        A series with one entry per company mention.
    """
    from .data_loading import split_entity_list

    return dataframe[column].apply(split_entity_list).explode().dropna()
