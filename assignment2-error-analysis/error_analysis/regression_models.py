"""The regression models compared in chapter 3, and the machinery to score them.

Hyperparameters are set here rather than in the notebook so that the exact
configuration behind every reported number is recorded in one place.
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.tree import DecisionTreeRegressor

from .data_setup import build_pipeline
from .validation import RANDOM_STATE, regression_metrics

# The unbounded tree is included deliberately. It is not a candidate model; it
# is the evidence for the bias-variance argument in section 3.5, and it shows
# what the tuned depth in the next entry is protecting against.
TREE_MAX_DEPTH = 3

REGRESSION_MODELS = {
    "Linear Regression": LinearRegression(),
    "Decision Tree (unbounded)": DecisionTreeRegressor(random_state=RANDOM_STATE),
    f"Decision Tree (max_depth={TREE_MAX_DEPTH})": DecisionTreeRegressor(
        max_depth=TREE_MAX_DEPTH, random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
    ),
}


def describe_hyperparameters() -> pd.DataFrame:
    """List the configuration of every model, as the assignment requires.

    Returns:
        One row per model with its non-default settings spelled out.
    """
    descriptions = {
        "Linear Regression": "All defaults. Ordinary least squares, no regularisation.",
        "Decision Tree (unbounded)": "All defaults, so the depth is unlimited. Retained as evidence, not as a candidate.",
        f"Decision Tree (max_depth={TREE_MAX_DEPTH})": f"max_depth={TREE_MAX_DEPTH}, chosen by the sweep in 3.2. Other settings default.",
        "Random Forest": "n_estimators=300, min_samples_leaf=2. Other settings default.",
    }
    return pd.DataFrame(
        [{"Model": name, "Configuration": text} for name, text in descriptions.items()]
    )


def sweep_tree_depth(
    features: pd.DataFrame, target: pd.Series, cv, depths=(2, 3, 4, 5, 6, 8, 10, 15, None)
) -> pd.DataFrame:
    """Score a decision tree at several depths, reporting train and test alike.

    The gap between the two is the quantity of interest: it is what the
    bias-variance discussion needs in order to be evidence rather than
    assertion.

    Args:
        features: Model input matrix.
        target: Regression target.
        cv: Cross-validation splitter.
        depths: Values of max_depth to try, in increasing order of model
            complexity; None means unlimited and is therefore placed last.

    Returns:
        One row per depth.
    """
    rows = []
    for depth in depths:
        scores = cross_validate(
            build_pipeline(DecisionTreeRegressor(max_depth=depth, random_state=RANDOM_STATE)),
            features,
            target,
            cv=cv,
            scoring="r2",
            return_train_score=True,
        )
        rows.append(
            {
                "max_depth": "unbounded" if depth is None else depth,
                "Train R2": round(scores["train_score"].mean(), 3),
                "Test R2": round(scores["test_score"].mean(), 3),
                "Gap": round(scores["train_score"].mean() - scores["test_score"].mean(), 3),
            }
        )
    return pd.DataFrame(rows).set_index("max_depth")


def run_model_comparison(
    features: pd.DataFrame, target: pd.Series, cv
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    """Score every model on an identical feature set and resampling scheme.

    Args:
        features: Model input matrix, shared by all models.
        target: Regression target.
        cv: Cross-validation splitter, shared by all models.

    Returns:
        A metrics table, the out-of-fold predictions per model, and the
        per-fold R2 scores used to show stability.
    """
    metric_rows, predictions, fold_scores = [], {}, {}

    for name, estimator in REGRESSION_MODELS.items():
        pipeline = build_pipeline(estimator)

        started = time.perf_counter()
        predicted = cross_val_predict(pipeline, features, target, cv=cv)
        elapsed = time.perf_counter() - started

        scores = cross_validate(
            pipeline, features, target, cv=cv, scoring="r2", return_train_score=True
        )
        metrics = regression_metrics(target, predicted)

        predictions[name] = predicted
        fold_scores[name] = scores["test_score"]
        metric_rows.append(
            {
                "Model": name,
                "MAE": round(metrics["MAE (log10)"], 4),
                "MSE": round(metrics["MSE (log10)"], 4),
                "RMSE": round(metrics["RMSE (log10)"], 4),
                "R2": round(metrics["R2"], 4),
                "Median error factor": round(metrics["Median error factor"], 2),
                "Train R2": round(scores["train_score"].mean(), 3),
                "Overfit gap": round(scores["train_score"].mean() - scores["test_score"].mean(), 3),
                "R2 SD across folds": round(scores["test_score"].std(), 3),
                "Fit time (s)": round(elapsed, 2),
            }
        )

    return (
        pd.DataFrame(metric_rows).set_index("Model"),
        predictions,
        pd.DataFrame(fold_scores),
    )


def compare_explanations(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Put the linear coefficients beside the forest's feature importances.

    The two are not on the same scale and are not directly comparable in
    magnitude; the point of the table is which features each model relies on,
    and where the two disagree.

    Args:
        features: Model input matrix.
        target: Regression target.

    Returns:
        One row per encoded feature.
    """
    linear = build_pipeline(LinearRegression()).fit(features, target)
    forest = build_pipeline(REGRESSION_MODELS["Random Forest"]).fit(features, target)

    encoded_names = [
        name.split("__", 1)[1]
        for name in linear.named_steps["preprocess"].get_feature_names_out()
    ]
    return pd.DataFrame(
        {
            "Linear coefficient": linear.named_steps["estimator"].coef_,
            "Random Forest importance": forest.named_steps["estimator"].feature_importances_,
        },
        index=encoded_names,
    ).sort_values("Random Forest importance", ascending=False)


def measure_noise_sensitivity(
    features: pd.DataFrame, target: pd.Series, cv, noise_levels=(0.0, 0.25, 0.5)
) -> pd.DataFrame:
    """Re-score the models after adding Gaussian noise to the target.

    Sensitivity to noise is one of the aspects the assignment asks the
    discussion to address, and injecting a known amount of it is a more direct
    test than reasoning about the models' properties.

    Args:
        features: Model input matrix.
        target: Regression target.
        cv: Cross-validation splitter.
        noise_levels: Standard deviations of the noise added, in log units.

    Returns:
        Mean R2 per model at each noise level.
    """
    from sklearn.model_selection import cross_val_score

    generator = np.random.default_rng(RANDOM_STATE)
    candidates = {
        name: model
        for name, model in REGRESSION_MODELS.items()
        if name != "Decision Tree (unbounded)"
    }

    rows = []
    for noise_sd in noise_levels:
        noisy_target = target + generator.normal(0, noise_sd, len(target))
        row = {"Noise SD (log units)": noise_sd}
        for name, model in candidates.items():
            row[name] = round(
                cross_val_score(
                    build_pipeline(model), features, noisy_target, cv=cv, scoring="r2"
                ).mean(),
                3,
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("Noise SD (log units)")
