"""Checks for completeness, duplication, suspicious values and cardinality.

Every function here returns a table rather than printing or plotting, so the
notebook decides how results are presented and the same check can be reused in
a later section without side effects.
"""

import pandas as pd


def summarise_missing(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Report how much of each column is absent.

    Args:
        dataframe: The dataset to inspect.

    Returns:
        One row per column with missing values, plus the share of the column
        and of the table as a whole. Columns that are complete are omitted so
        the result stays readable.
    """
    missing_counts = dataframe.isna().sum()
    total_cells = dataframe.size

    summary = pd.DataFrame(
        {
            "Column": missing_counts.index,
            "Missing": missing_counts.to_numpy(),
            "Share of column": [
                f"{count / len(dataframe):.3%}" for count in missing_counts
            ],
            "Share of all cells": [f"{count / total_cells:.4%}" for count in missing_counts],
        }
    )
    return summary[summary["Missing"] > 0].reset_index(drop=True)


def summarise_duplicates(
    dataframe: pd.DataFrame, key_subsets: dict[str, list[str]]
) -> pd.DataFrame:
    """Count duplicates for the whole row and for candidate key combinations.

    A full-row duplicate and a repeated identifier mean very different things,
    so both are reported side by side.

    Args:
        dataframe: The dataset to inspect.
        key_subsets: Mapping of a human-readable label to the columns that
            together should uniquely identify a row.

    Returns:
        One row per check, giving the number of rows that repeat a combination
        already seen earlier in the file.
    """
    checks = [("Entire row (all 11 columns)", list(dataframe.columns))]
    checks.extend(key_subsets.items())

    return pd.DataFrame(
        [
            {
                "Duplicate check": label,
                "Columns compared": len(columns),
                "Repeated rows": int(dataframe.duplicated(subset=columns).sum()),
            }
            for label, columns in checks
        ]
    )


def detect_descending_blocks(series: pd.Series) -> pd.DataFrame:
    """Split a series into maximal runs that never increase.

    If a file was produced by concatenating several sorted extracts, the joins
    show up as the only points where the values step back up.

    Args:
        series: A numeric column, in the file's own row order.

    Returns:
        One row per descending run, with its position, size and value range.
    """
    break_positions = series.index[series.diff() > 0].tolist()
    boundaries = [0, *break_positions, len(series)]

    return pd.DataFrame(
        [
            {
                "Block": chr(ord("A") + block_number),
                "First row": start,
                "Last row": end - 1,
                "Rows": end - start,
                "Highest value": series.iloc[start:end].max(),
                "Lowest value": series.iloc[start:end].min(),
            }
            for block_number, (start, end) in enumerate(
                zip(boundaries[:-1], boundaries[1:])
            )
        ]
    )


def describe_index(
    dataframe: pd.DataFrame, sort_candidates: list[str]
) -> pd.DataFrame:
    """Report what the row index is, and whether the rows are ordered by anything.

    Args:
        dataframe: The dataset, in its original row order.
        sort_candidates: Columns the file might plausibly have been sorted by.

    Returns:
        A two-column table of properties and their observed values.
    """
    index = dataframe.index
    properties = [
        ("Index type", type(index).__name__),
        ("Index range", f"{index.min()} to {index.max()}"),
        ("Index is unique", "Yes" if index.is_unique else "No"),
        ("Index carries meaning", "No — it is positional, assigned on load"),
        ("Index is time-based", "No"),
    ]

    for column in sort_candidates:
        values = dataframe[column]
        if values.is_monotonic_increasing:
            order = "Sorted ascending"
        elif values.is_monotonic_decreasing:
            order = "Sorted descending"
        else:
            order = "Not sorted"
        properties.append((f"Rows ordered by {column}", order))

    return pd.DataFrame(properties, columns=["Property", "Value"])


def summarise_zero_values(
    dataframe: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Count zeros in numeric columns, where zero may be a real value or a stand-in.

    Args:
        dataframe: The dataset to inspect.
        columns: Numeric columns in which a zero is worth questioning.

    Returns:
        One row per column with the count and share of zero values, and the
        smallest non-zero value for comparison.
    """
    return pd.DataFrame(
        [
            {
                "Column": column,
                "Zeros": int((dataframe[column] == 0).sum()),
                "Share of rows": f"{(dataframe[column] == 0).mean():.1%}",
                "Smallest non-zero value": dataframe.loc[
                    dataframe[column] > 0, column
                ].min(),
            }
            for column in columns
        ]
    )


def compare_groups(
    dataframe: pd.DataFrame, mask: pd.Series, columns: list[str], labels: tuple[str, str]
) -> pd.DataFrame:
    """Compare median values between two subsets of the rows.

    Used to test whether a suspicious value behaves like a real measurement or
    like a placeholder: a placeholder group usually differs from the rest in
    ways the stated meaning cannot explain.

    Args:
        dataframe: The dataset to inspect.
        mask: Boolean selector for the group under suspicion.
        columns: Numeric columns to compare.
        labels: Display names for the selected and unselected groups.

    Returns:
        One row per column with the median of each group.
    """
    selected_label, other_label = labels
    return pd.DataFrame(
        [
            {
                "Column": column,
                f"Median — {selected_label}": round(dataframe.loc[mask, column].median(), 2),
                f"Median — {other_label}": round(dataframe.loc[~mask, column].median(), 2),
            }
            for column in columns
        ]
    )


def build_cardinality_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Describe how many distinct values each column holds.

    Args:
        dataframe: The dataset to inspect.

    Returns:
        One row per column with its distinct count, uniqueness ratio, and the
        most frequent value together with the share of rows it covers.
    """
    row_count = len(dataframe)
    rows = []
    for column in dataframe.columns:
        distinct = dataframe[column].nunique(dropna=True)
        value_counts = dataframe[column].value_counts(dropna=True)
        rows.append(
            {
                "Column": column,
                "Distinct values": int(distinct),
                "Uniqueness": f"{distinct / row_count:.1%}",
                "Constant": "Yes" if distinct <= 1 else "No",
                "Most frequent value": value_counts.index[0] if len(value_counts) else None,
                "Its share of rows": f"{value_counts.iloc[0] / row_count:.1%}"
                if len(value_counts)
                else None,
            }
        )
    return pd.DataFrame(rows)
