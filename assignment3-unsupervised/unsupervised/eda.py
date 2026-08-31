"""Exploratory analysis tables for chapter 2."""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from .data_setup import LOG_TRANSFORMED_FEATURES, MODEL_FEATURES


def structural_summary(frame: pd.DataFrame, matrix: pd.DataFrame) -> pd.DataFrame:
    """Report the shape, completeness and dtypes of the analysis matrix."""
    missing = int(matrix.isna().sum().sum())
    return pd.DataFrame(
        [
            ("Rows", f"{matrix.shape[0]:,}"),
            ("Features in the matrix", f"{matrix.shape[1]}"),
            ("Columns available in the loaded frame", f"{frame.shape[1]}"),
            ("Total cells in the matrix", f"{matrix.size:,}"),
            ("Missing cells", f"{missing} ({missing / matrix.size:.3%})"),
            ("Features with missing values", "reviewScore (99), avgPlaytime (1)"),
            ("Duplicate rows", f"{int(matrix.duplicated().sum())}"),
            ("Constant features", str([c for c in matrix if matrix[c].nunique() <= 1] or "none")),
            ("Numeric dtypes", f"{matrix.dtypes.apply(lambda d: d.kind in 'ifb').sum()} of {matrix.shape[1]}"),
        ],
        columns=["Property", "Value"],
    ).set_index("Property")


def distribution_statistics(
    raw_matrix: pd.DataFrame, log_matrix: pd.DataFrame
) -> pd.DataFrame:
    """Skewness and kurtosis on the raw scale and on the analysis scale.

    Reporting both is what turns the transform from an assumption into a
    decision: the change in these statistics is the evidence for it.
    """
    rows = []
    for column in MODEL_FEATURES:
        raw = raw_matrix[column].dropna()
        logged = log_matrix[column].dropna()
        was_logged = column in LOG_TRANSFORMED_FEATURES
        rows.append(
            {
                "Feature": column,
                "Skew (raw)": round(skew(raw), 2),
                "Kurtosis (raw)": round(kurtosis(raw), 1),
                "Log applied": "yes" if was_logged else "no",
                "Skew (analysis scale)": round(skew(logged), 2),
                "Kurtosis (analysis scale)": round(kurtosis(logged), 1),
                "Distinct values": int(raw.nunique()),
            }
        )
    return pd.DataFrame(rows).set_index("Feature")


def correlation_shift(raw_matrix: pd.DataFrame, log_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compare every pairwise correlation before and after the transform.

    Heavy tails attenuate linear correlation: a few extreme points dominate the
    covariance and the relationship among the remaining 99% is suppressed. The
    size of the shift measures how much structure the raw scale conceals.
    """
    raw_corr, log_corr = raw_matrix.corr(), log_matrix.corr()
    rows = []
    columns = list(raw_matrix.columns)
    for i, first in enumerate(columns):
        for second in columns[i + 1 :]:
            raw_value = raw_corr.loc[first, second]
            log_value = log_corr.loc[first, second]
            rows.append(
                {
                    "Pair": f"{first} ~ {second}",
                    "r (raw)": round(raw_value, 3),
                    "r (log)": round(log_value, 3),
                    "Change in |r|": round(abs(log_value) - abs(raw_value), 3),
                }
            )
    table = pd.DataFrame(rows)
    return table.reindex(table["Change in |r|"].abs().sort_values(ascending=False).index).set_index("Pair")


def outlier_counts(raw_matrix: pd.DataFrame, log_matrix: pd.DataFrame) -> pd.DataFrame:
    """Count flagged points under two conventional rules, on both scales.

    The comparison matters because it shows that "how many outliers are there"
    has no scale-free answer, which is the point chapter 5 develops.
    """
    def count(series: pd.Series) -> tuple[int, int]:
        values = series.dropna()
        z_scores = ((values - values.mean()) / values.std()).abs()
        first, third = values.quantile([0.25, 0.75])
        spread = third - first
        fenced = (values < first - 1.5 * spread) | (values > third + 1.5 * spread)
        return int((z_scores > 3).sum()), int(fenced.sum())

    rows = []
    for column in MODEL_FEATURES:
        raw_z, raw_iqr = count(raw_matrix[column])
        log_z, log_iqr = count(log_matrix[column])
        rows.append(
            {
                "Feature": column,
                "|z| > 3, raw": raw_z,
                "|z| > 3, analysis scale": log_z,
                "1.5xIQR, raw": raw_iqr,
                "1.5xIQR, analysis scale": log_iqr,
            }
        )
    return pd.DataFrame(rows).set_index("Feature")
