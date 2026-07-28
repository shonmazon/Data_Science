"""Formal hypothesis tests for section 9.

Non-parametric tests are used throughout. Section 5 showed the numeric
variables are heavily skewed with extreme tails, which violates the normality
assumption behind the t-test, so rank-based tests are the appropriate choice.

Because several hypotheses are tested on the same dataset, a Bonferroni
correction is applied: testing enough hypotheses will eventually produce a
significant result by chance alone.
"""

from dataclasses import dataclass

import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr

UNCORRECTED_ALPHA = 0.05


@dataclass(frozen=True)
class TestResult:
    """One hypothesis, the test applied to it, and its outcome."""

    hypothesis: str
    test: str
    statistic: float
    p_value: float
    detail: str


def _test_class_revenue(dataframe: pd.DataFrame) -> TestResult:
    """AAA titles earn more than Indie titles."""
    aaa = dataframe.loc[dataframe["plotClass"] == "AAA", "revenue"]
    indie = dataframe.loc[dataframe["plotClass"] == "Indie", "revenue"]
    statistic, p_value = mannwhitneyu(aaa, indie, alternative="greater")
    return TestResult(
        "AAA titles earn more than Indie titles",
        "Mann-Whitney U (one-sided)",
        statistic,
        p_value,
        f"medians ${aaa.median():,.0f} vs ${indie.median():,.0f}",
    )


def _test_weekend_revenue(dataframe: pd.DataFrame) -> TestResult:
    """Weekend releases earn less than weekday releases."""
    weekend = dataframe.loc[dataframe["isWeekendRelease"], "revenue"]
    weekday = dataframe.loc[~dataframe["isWeekendRelease"], "revenue"]
    statistic, p_value = mannwhitneyu(weekend, weekday, alternative="less")
    return TestResult(
        "Weekend releases earn less than weekday releases",
        "Mann-Whitney U (one-sided)",
        statistic,
        p_value,
        f"n = {len(weekend)} weekend vs {len(weekday)} weekday",
    )


def _test_score_revenue(dataframe: pd.DataFrame) -> TestResult:
    """Review score is associated with revenue."""
    scored = dataframe.dropna(subset=["reviewScore"])
    statistic, p_value = spearmanr(scored["reviewScore"], scored["revenue"])
    return TestResult(
        "Review score is associated with revenue",
        "Spearman rank correlation",
        statistic,
        p_value,
        f"rho = {statistic:.4f} on n = {len(scored):,}",
    )


def _test_free_to_play_scores(dataframe: pd.DataFrame) -> TestResult:
    """Free-to-play titles are rated differently from paid titles."""
    free = dataframe.loc[dataframe["isFreeToPlay"], "reviewScore"].dropna()
    paid = dataframe.loc[~dataframe["isFreeToPlay"], "reviewScore"].dropna()
    statistic, p_value = mannwhitneyu(free, paid, alternative="two-sided")
    return TestResult(
        "Free-to-play titles are rated differently from paid titles",
        "Mann-Whitney U (two-sided)",
        statistic,
        p_value,
        f"medians {free.median():.0f} vs {paid.median():.0f}",
    )


def _test_weekday_independence(dataframe: pd.DataFrame) -> TestResult:
    """Release weekday depends on publisher class."""
    contingency = pd.crosstab(dataframe["releaseWeekday"], dataframe["plotClass"])
    statistic, p_value, degrees_of_freedom, _ = chi2_contingency(contingency)
    return TestResult(
        "Release weekday depends on publisher class",
        "Chi-squared test of independence",
        statistic,
        p_value,
        f"chi2 = {statistic:.1f} on {degrees_of_freedom} degrees of freedom",
    )


def run_hypothesis_tests(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Run every hypothesis test and report them against a corrected threshold.

    Args:
        dataframe: A frame with the engineered features from section 9.

    Returns:
        One row per hypothesis, with the decision taken at the Bonferroni
        corrected significance level.
    """
    tests = [
        _test_class_revenue,
        _test_weekend_revenue,
        _test_score_revenue,
        _test_free_to_play_scores,
        _test_weekday_independence,
    ]
    results = [test(dataframe) for test in tests]
    corrected_alpha = UNCORRECTED_ALPHA / len(results)

    return pd.DataFrame(
        [
            {
                "Hypothesis": result.hypothesis,
                "Test": result.test,
                "p-value": f"{result.p_value:.3g}",
                "Detail": result.detail,
                f"Decision at alpha={corrected_alpha:.3f}": (
                    "Reject null" if result.p_value < corrected_alpha else "Fail to reject null"
                ),
            }
            for result in results
        ]
    )
