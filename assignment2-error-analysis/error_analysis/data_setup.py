"""Modelling frame, feature definitions and cross-validation pipelines.

The cleaning performed in Assignment 1 is reused rather than reimplemented, so
the placeholder handling and the derived features are identical across the two
notebooks. Only the parts specific to modelling are defined here.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Assignment 1's helper package lives in a sibling project directory. Adding it
# to the import path here keeps this module self-contained, so it behaves the
# same whether it is imported from the notebook or from a script.
ASSIGNMENT1_DIR = Path(__file__).resolve().parents[2] / "assignment1-eda"
if str(ASSIGNMENT1_DIR) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENT1_DIR))

from src.associations import add_analysis_features  # noqa: E402
from src.cleaning import apply_quality_fixes  # noqa: E402
from src.data_loading import load_raw_dataset  # noqa: E402
from src.features import add_engineered_features  # noqa: E402

# Extraction date established in Assignment 1, section 3.1.
SNAPSHOT_DATE = pd.Timestamp("2024-09-09")

# Features available to every model. The same set is used for regression and
# for classification, and for all model families, so that differences in
# performance can be attributed to the models rather than to their inputs.
NUMERIC_FEATURES = [
    "price",
    "reviewScore",
    "avgPlaytime",
    "daysOnSale",
    "publisherReleaseCount",
    "releaseMonth",
]
CATEGORICAL_FEATURES = ["plotClass", "releaseWeekday"]
BOOLEAN_FEATURES = ["selfPublished", "isFreeToPlay"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

# Columns withheld from every model. Each is either the target itself or a
# quantity computed from it; section 1.3 demonstrates why copiesSold belongs
# on this list rather than in the feature set.
WITHHELD_COLUMNS = [
    "revenue",
    "logRevenue",
    "copiesSold",
    "logCopiesSold",
    "revenuePerCopy",
    "revenuePerDay",
    "priceRealisation",
    "revenueQuartile",
    "isStandout",
]

REGRESSION_TARGET = "logRevenue"
CLASSIFICATION_TARGET = "isStandout"
STANDOUT_QUANTILE = 0.75


def build_modelling_frame() -> pd.DataFrame:
    """Load the dataset and derive every column the modelling chapters need.

    Returns:
        The 1,500 games with the Assignment 1 cleaning applied, the derived
        features added, and both targets attached.
    """
    frame = add_engineered_features(
        add_analysis_features(apply_quality_fixes(load_raw_dataset()), SNAPSHOT_DATE)
    )

    frame["logRevenue"] = np.log10(frame["revenue"])
    frame["logCopiesSold"] = np.log10(frame["copiesSold"])
    frame["releaseMonth"] = frame["releaseDate"].dt.month

    standout_threshold = frame["revenue"].quantile(STANDOUT_QUANTILE)
    frame["isStandout"] = (frame["revenue"] >= standout_threshold).astype(int)

    return frame


def build_preprocessor(numeric_features: list[str] | None = None) -> ColumnTransformer:
    """Assemble the preprocessing applied inside every cross-validation fold.

    Numeric columns are median-imputed and standardised; categorical columns
    are one-hot encoded with the first level dropped; booleans pass through.
    Placing all of it inside a transformer means it is fitted on the training
    portion of each fold only, so no information crosses from the held-out
    fold into the transformation applied to it.

    Args:
        numeric_features: Numeric columns to use. Defaults to the standard set,
            and is overridden in section 1.3 to test an alternative.

    Returns:
        An unfitted column transformer.
    """
    numeric = NUMERIC_FEATURES if numeric_features is None else numeric_features
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("boolean", "passthrough", BOOLEAN_FEATURES),
        ]
    )


def build_pipeline(estimator, numeric_features: list[str] | None = None) -> Pipeline:
    """Wrap an estimator behind the shared preprocessing.

    Args:
        estimator: Any scikit-learn regressor or classifier.
        numeric_features: Optional override of the numeric column list.

    Returns:
        A pipeline that can be passed directly to a cross-validation helper.
    """
    return Pipeline(
        [("preprocess", build_preprocessor(numeric_features)), ("estimator", estimator)]
    )


def describe_feature_roles() -> pd.DataFrame:
    """Tabulate every feature with its role and the reason it is included.

    Returns:
        One row per feature used by the models.
    """
    reasons = {
        "price": "Listed price; the strongest legitimate predictor of revenue",
        "reviewScore": "Percentage of positive reviews, after placeholder removal",
        "avgPlaytime": "Average hours played, a proxy for engagement",
        "daysOnSale": "Exposure at the snapshot date",
        "publisherReleaseCount": "How many titles the publisher has in this dataset",
        "releaseMonth": "Calendar month of release",
        "plotClass": "Studio scale: Indie, AA or AAA",
        "releaseWeekday": "Day of the week the title launched",
        "selfPublished": "Whether publisher and developer are the same company",
        "isFreeToPlay": "Whether the listed price is zero",
    }
    roles = (
        [("numeric", name) for name in NUMERIC_FEATURES]
        + [("categorical", name) for name in CATEGORICAL_FEATURES]
        + [("boolean", name) for name in BOOLEAN_FEATURES]
    )
    return pd.DataFrame(
        [{"Feature": name, "Type": role, "Why it is included": reasons[name]}
         for role, name in roles]
    )
