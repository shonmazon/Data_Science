"""Turn the raw dataset into an analysis-ready frame.

Every change made here follows from a finding in section 4, and each one is
deliberately conservative: values are corrected or marked as unknown, never
invented. No row is dropped, so the analysed table still contains all 1,500
games.
"""

import pandas as pd

from .data_loading import parse_release_date

# Columns where a stored 0 was shown in section 4.3 to mean "no value
# available" rather than the number zero.
PLACEHOLDER_ZERO_COLUMNS = ["reviewScore", "avgPlaytime"]

# Publisher class is an ordinal scale by studio size. The order is not recorded
# anywhere in the file and comes from domain knowledge.
PUBLISHER_CLASS_ORDER = ["Hobbyist", "Indie", "AA", "AAA"]


def apply_quality_fixes(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply the corrections justified by the data quality review.

    The corrections are:

    * `reviewScore` and `avgPlaytime` placeholders of 0 become missing values,
      so they are excluded from statistics instead of dragging them down.
    * `releaseDate` becomes a real timestamp, parsed with an explicit format.
    * `publisherClass` becomes an ordered category.
    * Stray whitespace is stripped from game titles.

    `price` is deliberately left untouched: its zeros are genuine free-to-play
    prices, not placeholders.

    Args:
        dataframe: The raw dataset.

    Returns:
        A new frame with the same rows and columns, ready for analysis.
    """
    cleaned = dataframe.copy()

    for column in PLACEHOLDER_ZERO_COLUMNS:
        cleaned[column] = cleaned[column].mask(cleaned[column] == 0)

    cleaned["releaseDate"] = parse_release_date(dataframe)
    cleaned["publisherClass"] = pd.Categorical(
        cleaned["publisherClass"], categories=PUBLISHER_CLASS_ORDER, ordered=True
    )
    cleaned["name"] = cleaned["name"].str.strip()

    return cleaned


def summarise_cleaning_effect(
    raw: pd.DataFrame, cleaned: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Show how the corrections changed each affected column.

    Args:
        raw: The dataset before cleaning.
        cleaned: The dataset after cleaning.
        columns: Columns to report on.

    Returns:
        One row per column comparing counts and means before and after.
    """
    return pd.DataFrame(
        [
            {
                "Column": column,
                "Values before": int(raw[column].notna().sum()),
                "Values after": int(cleaned[column].notna().sum()),
                "Marked unknown": int(raw[column].notna().sum() - cleaned[column].notna().sum()),
                "Mean before": round(raw[column].mean(), 2),
                "Mean after": round(cleaned[column].mean(), 2),
            }
            for column in columns
        ]
    )
