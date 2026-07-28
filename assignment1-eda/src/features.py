"""Engineered features.

These columns are not in the source file. Each one is built to answer a
question the raw columns cannot, and each is documented with the reasoning that
motivated it, because an engineered feature that nobody can interpret is worse
than no feature at all.
"""

import pandas as pd

from .data_loading import split_entity_list


def add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Derive the analytical features used in section 9.

    The features are:

    * `revenuePerCopy` — the average amount actually received per unit sold.
    * `priceRealisation` — that amount as a fraction of the list price. It
      exposes the combined effect of discounts, regional pricing and any
      platform cut, none of which the file states.
    * `revenuePerDay` — revenue normalised by time on sale, which is the only
      way to compare titles whose exposure windows differ by a factor of fifty.
    * `releaseWeekday` and `isWeekendRelease` — release timing, to test whether
      the industry's scheduling conventions are visible in the data.
    * `publisherReleaseCount` — how many titles the publisher has in this
      dataset, which turns an unusable high-cardinality column into a number.

    Args:
        dataframe: A frame that already has `daysOnSale` from
            `add_analysis_features`.

    Returns:
        A copy of the frame with the engineered columns appended.
    """
    engineered = dataframe.copy()

    engineered["revenuePerCopy"] = engineered["revenue"] / engineered["copiesSold"]
    # Undefined for free-to-play titles, where the list price is zero.
    engineered["priceRealisation"] = (
        engineered["revenuePerCopy"] / engineered["price"].replace(0, pd.NA)
    )
    engineered["revenuePerDay"] = engineered["revenue"] / engineered["daysOnSale"]

    engineered["releaseWeekday"] = engineered["releaseDate"].dt.day_name()
    engineered["isWeekendRelease"] = engineered["releaseDate"].dt.dayofweek >= 5

    # Counting the stored strings would repeat the mistake section 3.2 warned
    # about, since "Aspyr,Crystal Dynamics" is not the same category as "Aspyr".
    # Companies are counted individually, and a title takes the count of the
    # most prolific publisher attached to it.
    publisher_lists = engineered["publishers"].apply(split_entity_list)
    company_counts = publisher_lists.explode().dropna().value_counts()
    engineered["publisherReleaseCount"] = publisher_lists.apply(
        lambda companies: max((company_counts[name] for name in companies), default=1)
    )

    return engineered


def summarise_release_timing(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Compare release volume and outcome across days of the week.

    Args:
        dataframe: A frame with the engineered timing columns.

    Returns:
        One row per weekday, ordered Monday to Sunday.
    """
    weekday_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    ]
    grouped = dataframe.groupby("releaseWeekday", observed=True)

    summary = pd.DataFrame(
        {
            "Releases": grouped.size(),
            "Share of releases": (grouped.size() / len(dataframe) * 100).round(1),
            "Median revenue": grouped["revenue"].median().round(0),
            "Median review score": grouped["reviewScore"].median(),
        }
    )
    return summary.reindex(weekday_order)
