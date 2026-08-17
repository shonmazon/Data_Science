"""Classifiers and error-analysis machinery for chapter 4.

Two models are compared. The assignment does not require more than one; a second
is included so that the discussion in 4.7 has something to compare against, and
because a linear and a tree-based classifier fail in different ways.
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_predict

from .data_setup import build_pipeline
from .validation import RANDOM_STATE

DEFAULT_THRESHOLD = 0.5
SWEEP_THRESHOLDS = np.round(np.arange(0.1, 0.95, 0.1), 1)
BETA_VALUES = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0)

CLASSIFICATION_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
    ),
}


def out_of_fold_probabilities(
    features: pd.DataFrame, target: pd.Series, cv
) -> dict[str, np.ndarray]:
    """Predicted probability of the positive class, from models that never saw the row.

    Every threshold, confusion matrix and ROC point in chapter 4 is built from
    these, so none of them reflects performance on training data.
    """
    return {
        name: cross_val_predict(
            build_pipeline(model), features, target, cv=cv, method="predict_proba"
        )[:, 1]
        for name, model in CLASSIFICATION_MODELS.items()
    }


def summarise_classifiers(
    target: pd.Series, probabilities: dict[str, np.ndarray],
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Headline metrics for each classifier at a single decision threshold."""
    rows = []
    for name, probability in probabilities.items():
        predicted = (probability >= threshold).astype(int)
        true_negative, false_positive, false_negative, true_positive = confusion_matrix(
            target, predicted, labels=[0, 1]
        ).ravel()
        rows.append(
            {
                "Model": name,
                "TP": true_positive,
                "FP": false_positive,
                "FN": false_negative,
                "TN": true_negative,
                "Accuracy": round(accuracy_score(target, predicted), 3),
                "Balanced accuracy": round(balanced_accuracy_score(target, predicted), 3),
                "Precision": round(precision_score(target, predicted, zero_division=0), 3),
                "Recall": round(recall_score(target, predicted, zero_division=0), 3),
                "F1": round(f1_score(target, predicted, zero_division=0), 3),
                "MCC": round(matthews_corrcoef(target, predicted), 3),
                "ROC-AUC": round(roc_auc_score(target, probability), 4),
            }
        )
    return pd.DataFrame(rows).set_index("Model")


def threshold_sweep(target: pd.Series, probability: np.ndarray) -> pd.DataFrame:
    """Evaluate every threshold from 0.1 to 0.9, as the assignment specifies."""
    rows = []
    for threshold in SWEEP_THRESHOLDS:
        predicted = (probability >= threshold).astype(int)
        true_negative, false_positive, false_negative, true_positive = confusion_matrix(
            target, predicted, labels=[0, 1]
        ).ravel()
        rows.append(
            {
                "Threshold": threshold,
                "Predicted positive": int(predicted.sum()),
                "TP": true_positive,
                "FP": false_positive,
                "FN": false_negative,
                "TN": true_negative,
                "Precision": round(precision_score(target, predicted, zero_division=0), 3),
                "Recall": round(recall_score(target, predicted, zero_division=0), 3),
                "F1": round(f1_score(target, predicted, zero_division=0), 3),
                "MCC": round(matthews_corrcoef(target, predicted), 3),
            }
        )
    return pd.DataFrame(rows).set_index("Threshold")


def fbeta_grid(target: pd.Series, probability: np.ndarray) -> pd.DataFrame:
    """F-beta at every combination of threshold and beta.

    Plotting F-beta at one fixed threshold shows only how the score changes.
    Computing it across thresholds as well shows the operationally useful fact:
    which threshold one should choose given how much recall is worth.
    """
    return pd.DataFrame(
        {
            f"beta={beta:g}": [
                round(
                    fbeta_score(
                        target, (probability >= threshold).astype(int),
                        beta=beta, zero_division=0,
                    ),
                    3,
                )
                for threshold in SWEEP_THRESHOLDS
            ]
            for beta in BETA_VALUES
        },
        index=SWEEP_THRESHOLDS,
    ).rename_axis("Threshold")


def confidence_of_prediction(probability: np.ndarray, threshold: float) -> np.ndarray:
    """How strongly the model backs whichever class it chose."""
    predicted = probability >= threshold
    return np.where(predicted, probability, 1 - probability)


def summarise_confidence(
    target: pd.Series, probability: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> pd.DataFrame:
    """Compare the model's confidence when it is right against when it is wrong."""
    predicted = (probability >= threshold).astype(int)
    frame = pd.DataFrame(
        {
            "confidence": confidence_of_prediction(probability, threshold),
            "Outcome": np.where(predicted == target, "Correct", "Incorrect"),
        }
    )
    return frame.groupby("Outcome")["confidence"].describe().round(3)


def high_confidence_errors(
    target: pd.Series, probability: np.ndarray, threshold: float = DEFAULT_THRESHOLD,
    confidence_levels=(0.8, 0.9, 0.95),
) -> pd.DataFrame:
    """Count the errors the model made while being most sure of itself."""
    predicted = (probability >= threshold).astype(int)
    confidence = confidence_of_prediction(probability, threshold)
    wrong = predicted != target

    rows = []
    for level in confidence_levels:
        confident = confidence >= level
        rows.append(
            {
                "Confidence at least": level,
                "Predictions": int(confident.sum()),
                "Of which wrong": int((wrong & confident).sum()),
                "False negatives": int((wrong & confident & (target == 1)).sum()),
                "False positives": int((wrong & confident & (target == 0)).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("Confidence at least")


def compare_features_by_outcome(
    frame: pd.DataFrame, features: list[str], correct: pd.Series
) -> pd.DataFrame:
    """Test whether misclassified games differ from correctly classified ones.

    A Mann-Whitney test is used rather than a t-test because chapter 2 showed
    these features to be heavily skewed.
    """
    rows = []
    for feature in features:
        subset = frame[[feature]].assign(correct=correct).dropna()
        right = subset.loc[subset["correct"], feature]
        wrong = subset.loc[~subset["correct"], feature]
        rows.append(
            {
                "Feature": feature,
                "Median when correct": round(right.median(), 2),
                "Median when wrong": round(wrong.median(), 2),
                "Mann-Whitney p": f"{mannwhitneyu(right, wrong).pvalue:.2g}",
            }
        )
    return pd.DataFrame(rows).set_index("Feature")


def error_rate_by_revenue_decile(
    frame: pd.DataFrame, target: pd.Series, correct: pd.Series
) -> pd.DataFrame:
    """Locate the failures against the underlying continuous quantity.

    The class label is a threshold on revenue, so grouping by revenue decile
    shows whether errors are spread across the range or concentrated near the
    boundary that defines the label.
    """
    decile = pd.qcut(frame["revenue"], 10, labels=False) + 1
    grouped = pd.DataFrame(
        {"decile": decile, "target": target, "correct": correct}
    ).groupby("decile")
    return pd.DataFrame(
        {
            "Games": grouped.size(),
            "Actual standouts": grouped["target"].sum(),
            "Errors": grouped["correct"].apply(lambda column: (~column).sum()),
            "Error rate": grouped["correct"].apply(lambda column: round(1 - column.mean(), 3)),
        }
    )
