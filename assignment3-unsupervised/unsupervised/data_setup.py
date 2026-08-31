"""Dataset loading, the analysis matrix, and its provenance record.

The cleaning and the derived features are imported from Assignment 1 rather
than reimplemented, so the three notebooks treat the data identically. What is
defined here is specific to this assignment: which columns enter the numeric
matrix used for PCA, clustering and anomaly detection, which are held back as
interpretation-only annotations, and why.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Assignment 1's helper package lives in a sibling project directory. Adding it
# here keeps this module self-contained.
ASSIGNMENT1_DIR = Path(__file__).resolve().parents[2] / "assignment1-eda"
if str(ASSIGNMENT1_DIR) not in sys.path:
    sys.path.insert(0, str(ASSIGNMENT1_DIR))

from src.associations import add_analysis_features  # noqa: E402
from src.cleaning import apply_quality_fixes  # noqa: E402
from src.data_loading import RAW_DATA_PATH, load_raw_dataset  # noqa: E402
from src.features import add_engineered_features  # noqa: E402

# Extraction date established in Assignment 1 section 3.1, corroborated by the
# date stated in the source dataset's title.
SNAPSHOT_DATE = pd.Timestamp("2024-09-09")

DATASET_URL = (
    "https://www.kaggle.com/datasets/alicemtopcu/"
    "top-1500-games-on-steam-by-revenue-09-09-2024"
)

# --- the analysis matrix -------------------------------------------------
# Five columns present in the source file, and three derived in Assignment 1
# and reused unchanged in Assignment 2. No feature was created for this
# assignment: the eight below already satisfy the minimum of eight features.
ORIGINAL_FEATURES = ["copiesSold", "price", "revenue", "avgPlaytime", "reviewScore"]
PRIOR_ENGINEERED_FEATURES = ["daysOnSale", "publisherReleaseCount", "selfPublished"]
NEW_FEATURES: list[str] = []

MODEL_FEATURES = ORIGINAL_FEATURES + PRIOR_ENGINEERED_FEATURES + NEW_FEATURES

# Heavy-tailed columns. Section 2 justifies the transform from the measured
# skewness; it is applied before scaling for every distance-based method.
LOG_TRANSFORMED_FEATURES = [
    "copiesSold",
    "revenue",
    "avgPlaytime",
    "publisherReleaseCount",
]

# Kept out of the numeric matrix and used only to interpret and validate the
# structures the unsupervised methods find.
ANNOTATION_VARIABLES = ["name", "steamId", "plotClass", "priceBand", "isFreeToPlay"]

MINIMUM_ROWS = 1000
MINIMUM_FEATURES = 8


def load_analysis_frame() -> pd.DataFrame:
    """Load the dataset with Assignment 1's cleaning and derived columns applied.

    Returns:
        All 1,500 games, carrying the eight modelling features, the annotation
        variables, and the intermediate columns the notebook reports on.
    """
    frame = add_engineered_features(
        add_analysis_features(apply_quality_fixes(load_raw_dataset()), SNAPSHOT_DATE)
    )
    frame["selfPublished"] = frame["selfPublished"].astype(int)
    return frame


def describe_feature_provenance() -> pd.DataFrame:
    """Record where every modelling feature came from and what it means.

    Units marked "inferred" are not documented by the source; they are the only
    reading consistent with the observed values, and the distinction is kept
    visible because an undocumented unit is a limitation of the data.
    """
    catalogue = [
        ("copiesSold", "Original column", "Estimated lifetime units sold at the snapshot date",
         "count"),
        ("price", "Original column", "Listed store price; zero denotes a free-to-play title",
         "USD (currency not stated by the source; inferred)"),
        ("revenue", "Original column", "Estimated lifetime gross revenue at the snapshot date",
         "USD (currency and gross/net basis not stated; inferred)"),
        ("avgPlaytime", "Original column", "Average time played per owner",
         "hours (unit not stated; inferred from the observed range)"),
        ("reviewScore", "Original column",
         "Share of user reviews that are positive; 0 is a placeholder, not a score",
         "percent, 0-100 (definition not stated; inferred)"),
        ("daysOnSale", "Engineered in Assignment 1",
         "Days between release and the 2024-09-09 snapshot, i.e. exposure", "days"),
        ("publisherReleaseCount", "Engineered in Assignment 1",
         "Titles in this dataset by the most prolific publisher credited on the game",
         "count"),
        ("selfPublished", "Engineered in Assignment 1",
         "1 when the publisher and developer strings match, 0 otherwise", "binary"),
    ]
    return pd.DataFrame(
        catalogue, columns=["Feature", "Provenance", "Meaning", "Unit"]
    ).set_index("Feature")


def verify_selection_requirements(frame: pd.DataFrame) -> pd.DataFrame:
    """Check the dataset against the five selection criteria in section 1.

    Args:
        frame: The loaded dataset.

    Returns:
        One row per criterion with the observed value and a pass marker.
    """
    raw = load_raw_dataset()
    identifiers = ["name", "steamId"]
    analysable = [column for column in raw.columns if column not in identifiers]
    numeric_raw = ["copiesSold", "price", "revenue", "avgPlaytime", "reviewScore"]

    checks = [
        ("At least 1,000 rows", f">= {MINIMUM_ROWS:,}", f"{len(raw):,}", len(raw) >= MINIMUM_ROWS),
        ("At least 8 features", f">= {MINIMUM_FEATURES}",
         f"{len(analysable)} in the source file, excluding 2 identifiers; "
         f"{len(MODEL_FEATURES)} used for modelling",
         len(analysable) >= MINIMUM_FEATURES),
        ("Mostly numerical variables", "majority numeric",
         f"{len(numeric_raw)} of {len(analysable)} source features are numeric; "
         f"the modelling matrix is {len(MODEL_FEATURES)}/{len(MODEL_FEATURES)} numeric",
         len(numeric_raw) > len(analysable) / 2),
        ("Real-world dataset", "yes",
         "Commercially released Steam titles, published on Kaggle", True),
        ("No pre-labeled anomaly column", "none present",
         "No column encodes an outlier, anomaly or class label", True),
    ]
    return pd.DataFrame(
        [
            {"Requirement": name, "Required": need, "Observed": seen,
             "Status": "PASS" if ok else "FAIL"}
            for name, need, seen, ok in checks
        ]
    ).set_index("Requirement")


def summarise_excluded_candidates() -> pd.DataFrame:
    """Record the columns considered for the matrix and rejected, with reasons.

    Documenting the rejections matters as much as the inclusions: two of these
    would have injected structure that the later analysis would then have
    "discovered".
    """
    exclusions = [
        ("publisherTier", "Would have been new for this assignment",
         "Ordinal encoding of plotClass. Excluded on evidence: a Z-score on it flags "
         "exactly the 52 AAA titles, turning a studio label into a statistical anomaly; "
         "it raises the components needed for 90% of variance from 6 to 7; and it lowers "
         "the k=3 silhouette from 0.195 to 0.181. Retained instead as an annotation."),
        ("releaseMonth", "Would have been new for this assignment",
         "Correlates -0.99 with daysOnSale because both derive from releaseDate. "
         "Including both would create redundancy that PCA would then report as a finding."),
        ("revenuePerCopy, revenuePerDay, priceRealisation", "Engineered in Assignment 1",
         "Algebraic functions of features already in the matrix, so they would manufacture "
         "collinearity rather than add information."),
        ("name, steamId", "Original columns",
         "Identifiers with 1,500 distinct values; carry no distributional signal."),
        ("publishers, developers", "Original columns",
         "Free-text with 1,131 and 1,406 distinct values; not numeric and not encodable "
         "at this cardinality. Their information enters through selfPublished and "
         "publisherReleaseCount."),
        ("plotClass, priceBand, isFreeToPlay", "Engineered in Assignment 1",
         "Categorical. Held back as annotations used to interpret and validate the "
         "clusters and anomalies rather than to define distances."),
    ]
    return pd.DataFrame(
        exclusions, columns=["Column", "Would have been", "Reason for exclusion"]
    ).set_index("Column")


def describe_source_file() -> pd.DataFrame:
    """Physical facts about the file, and what the source does and does not state."""
    size = RAW_DATA_PATH.stat().st_size
    return pd.DataFrame(
        [
            ("File", RAW_DATA_PATH.name),
            ("Size", f"{size:,} bytes ({size / 1024:.1f} KB)"),
            ("Distributed via", f"Kaggle, user 'alicemtopcu'"),
            ("Dataset URL", DATASET_URL),
            ("Extraction date", "2024-09-09, stated in the dataset title"),
            ("Why it was collected", "Not specified by the source"),
            ("Who originally collected it", "Not specified by the source"),
            ("Licence", "Not recorded in the downloaded file"),
            ("Methodology note", "None supplied with the file"),
        ],
        columns=["Property", "Value"],
    ).set_index("Property")


def build_matrix(frame: pd.DataFrame, log_transform: bool = True) -> pd.DataFrame:
    """Return the numeric matrix, optionally with the heavy tails compressed.

    Args:
        frame: Output of `load_analysis_frame`.
        log_transform: Whether to apply log10 to the heavy-tailed features.

    Returns:
        A frame of the eight modelling features, unscaled. Scaling belongs to
        the pipeline of whichever method consumes it.
    """
    matrix = frame[MODEL_FEATURES].copy()
    if log_transform:
        for column in LOG_TRANSFORMED_FEATURES:
            matrix[column] = np.log10(matrix[column])
    return matrix
