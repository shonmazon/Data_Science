"""Cross-validation helpers shared by the regression and classification chapters."""

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from scipy.stats import kurtosis, skew
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data_setup import build_pipeline

RANDOM_STATE = 42


def compare_fold_counts(
    features: pd.DataFrame, target: pd.Series, fold_counts: tuple[int, ...] = (3, 5, 10, 20)
) -> pd.DataFrame:
    """Measure how the choice of k affects bias, variance and runtime.

    The assignment requires the choice of k to be justified in terms of dataset
    size, computational complexity and the bias-variance trade-off. This runs
    the baseline model at several values of k so that the justification rests on
    measurements from this dataset rather than on a rule of thumb.

    Args:
        features: The model input matrix.
        target: The regression target.
        fold_counts: Values of k to evaluate.

    Returns:
        One row per k, reporting fold sizes, mean score, the spread of scores
        across folds, and elapsed time.
    """
    rows = []
    for k in fold_counts:
        started = time.perf_counter()
        scores = cross_val_score(
            build_pipeline(LinearRegression()),
            features,
            target,
            cv=KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE),
            scoring="r2",
        )
        rows.append(
            {
                "k": k,
                "Training rows per fold": len(features) - len(features) // k,
                "Test rows per fold": len(features) // k,
                "Mean R2": round(scores.mean(), 4),
                "SD of R2 across folds": round(scores.std(), 4),
                "Runtime (s)": round(time.perf_counter() - started, 2),
            }
        )
    return pd.DataFrame(rows).set_index("k")


def regression_metrics(target: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Compute the metric set used throughout the regression chapters.

    Errors are reported in log units, since that is the modelling scale, and
    also as a multiplicative factor on the original dollar scale, which is what
    a log error actually means in practice.

    Args:
        target: Observed values of log10(revenue).
        predicted: Predicted values on the same scale.

    Returns:
        A mapping of metric name to value.
    """
    residuals = target - predicted
    mean_squared = mean_squared_error(target, predicted)
    return {
        "MAE (log10)": mean_absolute_error(target, predicted),
        "MSE (log10)": mean_squared,
        "RMSE (log10)": np.sqrt(mean_squared),
        "R2": r2_score(target, predicted),
        "Median error factor": 10 ** np.median(np.abs(residuals)),
    }


def residual_statistics(residuals: pd.Series) -> pd.DataFrame:
    """Report the statistical properties of an error distribution.

    Args:
        residuals: Observed minus predicted values.

    Returns:
        A single-column table of named statistics.
    """
    statistics = {
        "Count": len(residuals),
        "Mean residual": residuals.mean(),
        "Median residual": residuals.median(),
        "MAE": residuals.abs().mean(),
        "Standard deviation": residuals.std(),
        "Skewness": skew(residuals),
        "Excess kurtosis": kurtosis(residuals),
    }
    return pd.DataFrame({"Value": pd.Series(statistics)}).round(4)


def compare_feature_sets(
    frame: pd.DataFrame,
    target: pd.Series,
    numeric_variants: dict[str, list[str]],
    cv,
) -> pd.DataFrame:
    """Score the same estimator under different numeric feature sets.

    Used in section 1.3 to decide empirically whether a candidate predictor
    should be withheld, by holding the model, the protocol and every other
    column fixed and varying only the column in question.

    Args:
        frame: The modelling frame.
        target: The regression target.
        numeric_variants: Label mapped to the numeric columns for that variant.
        cv: A cross-validation splitter.

    Returns:
        One row per variant with the standard regression metrics and the
        shape of its residual distribution.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_predict

    from .data_setup import BOOLEAN_FEATURES, CATEGORICAL_FEATURES

    rows = []
    for label, numeric in numeric_variants.items():
        features = frame[numeric + CATEGORICAL_FEATURES + BOOLEAN_FEATURES]
        predicted = cross_val_predict(
            build_pipeline(LinearRegression(), numeric), features, target, cv=cv
        )
        residuals = target - predicted
        metrics = regression_metrics(target, predicted)
        rows.append(
            {
                "Feature set": label,
                "R2": round(metrics["R2"], 4),
                "MAE (log10)": round(metrics["MAE (log10)"], 4),
                "RMSE (log10)": round(metrics["RMSE (log10)"], 4),
                "Median error factor": round(metrics["Median error factor"], 2),
                "Residual SD": round(residuals.std(), 4),
                "Excess kurtosis": round(kurtosis(residuals), 2),
            }
        )
    return pd.DataFrame(rows).set_index("Feature set")
