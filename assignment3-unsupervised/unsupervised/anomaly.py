"""The three anomaly detection methods and their diagnostics.

Each method is applied to the same eight-dimensional standardised matrix and
each flags the same number of points, so that chapter 6 compares the methods
rather than the sizes of their outputs.
"""

import time

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, shapiro
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from .data_setup import MODEL_FEATURES

RANDOM_STATE = 42

# Every method flags this share, chosen so the comparison in chapter 6 is not
# confounded by different output sizes. Section 5.2 shows what the choice costs.
CONTAMINATION = 0.05
LOF_NEIGHBOURS = 20
ISOLATION_TREES = 300
ZSCORE_THRESHOLD = 3.0


def normality_diagnostics(matrix: np.ndarray) -> pd.DataFrame:
    """Test the assumption the Z-score method rests on, feature by feature.

    The Shapiro-Wilk statistic is computed on a 1,000-row sample because the
    test becomes hypersensitive at larger n. The final two columns matter more
    than the p-values: a feature whose maximum standardised deviation is below
    the threshold cannot flag anything at all.
    """
    frame = pd.DataFrame(matrix, columns=MODEL_FEATURES)
    rows = []
    for column in MODEL_FEATURES:
        values = frame[column]
        statistic, p_value = shapiro(values.sample(1000, random_state=0))
        standardised = (values - values.mean()) / values.std()
        rows.append(
            {
                "Feature": column,
                "Shapiro-Wilk W": round(statistic, 3),
                "p-value": f"{p_value:.1e}",
                "Normal at 1%": "yes" if p_value > 0.01 else "no",
                "Max |z| observed": round(standardised.abs().max(), 2),
                f"Games with |z| > {ZSCORE_THRESHOLD:g}": int(
                    (standardised.abs() > ZSCORE_THRESHOLD).sum()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("Feature")


def zscore_featurewise(matrix: np.ndarray, threshold: float = ZSCORE_THRESHOLD) -> pd.Series:
    """Flag a row if any single feature exceeds the threshold in absolute z."""
    frame = pd.DataFrame(matrix, columns=MODEL_FEATURES)
    standardised = frame.apply(lambda column: (column - column.mean()) / column.std())
    return (standardised.abs() > threshold).any(axis=1)


def zscore_threshold_sweep(matrix: np.ndarray, thresholds=(2.5, 3.0, 3.5, 4.0)) -> pd.DataFrame:
    """How many rows the feature-wise rule flags at each threshold.

    The expected count under a true Gaussian is reported alongside, because the
    difference between the two is the multiple-comparisons problem made visible:
    testing eight features per row inflates the chance that at least one exceeds
    any fixed cut.
    """
    n_rows, n_features = matrix.shape
    rows = []
    for threshold in thresholds:
        per_feature = 2 * (1 - norm.cdf(threshold))
        expected = n_rows * (1 - (1 - per_feature) ** n_features)
        rows.append(
            {
                "Threshold": threshold,
                "Games flagged": int(zscore_featurewise(matrix, threshold).sum()),
                "Expected if truly Gaussian": round(expected, 1),
            }
        )
    return pd.DataFrame(rows).set_index("Threshold")


def mahalanobis_distance(matrix: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance of every row from the centre.

    This is the global counterpart of the feature-wise rule. It measures
    distance in a space stretched by the inverse covariance, so it accounts for
    the correlations between features that the feature-wise rule ignores.
    """
    centre = matrix.mean(axis=0)
    precision = np.linalg.pinv(np.cov(matrix, rowvar=False))
    centred = matrix - centre
    return np.einsum("ij,jk,ik->i", centred, precision, centred)


def zscore_global(matrix: np.ndarray, contamination: float = CONTAMINATION) -> pd.Series:
    """Flag the most distant rows by Mahalanobis distance."""
    distances = mahalanobis_distance(matrix)
    cutoff = np.sort(distances)[-int(round(contamination * len(matrix)))]
    return pd.Series(distances >= cutoff)


def compare_zscore_views(matrix: np.ndarray, contamination: float = CONTAMINATION) -> pd.DataFrame:
    """Set the feature-wise and global versions of the Z-score idea side by side."""
    featurewise = zscore_featurewise(matrix).to_numpy()
    global_flags = zscore_global(matrix, contamination).to_numpy()
    distances = mahalanobis_distance(matrix)
    return pd.DataFrame(
        [
            ("Feature-wise, any |z| > 3", int(featurewise.sum())),
            (f"Global, Mahalanobis top {contamination:.0%}", int(global_flags.sum())),
            ("Flagged by both", int((featurewise & global_flags).sum())),
            ("Flagged by only one", int((featurewise ^ global_flags).sum())),
            ("Jaccard agreement", round((featurewise & global_flags).sum()
                                        / (featurewise | global_flags).sum(), 3)),
            ("Chi-squared 99% critical value (df=8)", round(float(chi2.ppf(0.99, matrix.shape[1])), 2)),
            ("Games beyond that critical value", int((distances > chi2.ppf(0.99, matrix.shape[1])).sum())),
        ],
        columns=["Quantity", "Value"],
    ).set_index("Quantity")


def isolation_forest_sweep(matrix: np.ndarray, contaminations=(0.01, 0.02, 0.05, 0.10, 0.15)) -> pd.DataFrame:
    """Flag counts and score cut-offs across the contamination parameter."""
    rows, flag_sets = [], {}
    for contamination in contaminations:
        started = time.perf_counter()
        model = IsolationForest(
            n_estimators=ISOLATION_TREES, contamination=contamination,
            random_state=RANDOM_STATE, n_jobs=-1,
        ).fit(matrix)
        flags = model.predict(matrix) == -1
        flag_sets[contamination] = flags
        rows.append(
            {
                "Contamination": contamination,
                "Games flagged": int(flags.sum()),
                "Score cut-off": round(float(np.quantile(-model.score_samples(matrix), 1 - contamination)), 3),
                "Seconds": round(time.perf_counter() - started, 3),
            }
        )
    table = pd.DataFrame(rows).set_index("Contamination")
    nested = all(
        not (flag_sets[a] & ~flag_sets[b]).any()
        for a, b in zip(contaminations[:-1], contaminations[1:])
    )
    table.attrs["nested"] = nested
    return table


def isolation_forest_scores(matrix: np.ndarray, contamination: float = CONTAMINATION):
    """Fit the forest once and return its anomaly scores and flags."""
    model = IsolationForest(
        n_estimators=ISOLATION_TREES, contamination=contamination,
        random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(matrix)
    return -model.score_samples(matrix), pd.Series(model.predict(matrix) == -1)


def isolation_forest_seed_stability(matrix: np.ndarray, seeds=(1, 2, 3, 4)) -> pd.DataFrame:
    """How much the flagged set changes when only the random seed changes.

    The forest is built from random partitions, so its output is not
    deterministic. Comparing seeds separates what the method found from what
    this particular set of random trees found.
    """
    reference = set(np.flatnonzero(
        IsolationForest(n_estimators=ISOLATION_TREES, contamination=CONTAMINATION,
                        random_state=RANDOM_STATE, n_jobs=-1).fit_predict(matrix) == -1
    ))
    rows = []
    for seed in seeds:
        other = set(np.flatnonzero(
            IsolationForest(n_estimators=ISOLATION_TREES, contamination=CONTAMINATION,
                            random_state=seed, n_jobs=-1).fit_predict(matrix) == -1
        ))
        rows.append({"Seed": seed,
                     "Jaccard with seed 42": round(len(reference & other) / len(reference | other), 3)})
    return pd.DataFrame(rows).set_index("Seed")


def lof_sweep(matrix: np.ndarray, neighbour_counts=(5, 10, 20, 35, 50, 100)) -> tuple[pd.DataFrame, dict]:
    """Flag counts and agreement across the neighbourhood size k."""
    rows, flag_sets = [], {}
    for k in neighbour_counts:
        started = time.perf_counter()
        model = LocalOutlierFactor(n_neighbors=k, contamination=CONTAMINATION)
        flags = model.fit_predict(matrix) == -1
        scores = -model.negative_outlier_factor_
        flag_sets[k] = flags
        rows.append(
            {
                "n_neighbors": k,
                "Games flagged": int(flags.sum()),
                "Max LOF score": round(float(scores.max()), 2),
                "Median LOF score": round(float(np.median(scores)), 3),
                "Seconds": round(time.perf_counter() - started, 3),
            }
        )
    table = pd.DataFrame(rows).set_index("n_neighbors")
    reference = flag_sets[LOF_NEIGHBOURS]
    table[f"Jaccard with k={LOF_NEIGHBOURS}"] = [
        round((reference & flag_sets[k]).sum() / (reference | flag_sets[k]).sum(), 3)
        for k in neighbour_counts
    ]
    return table, flag_sets


def lof_scores(matrix: np.ndarray, n_neighbors: int = LOF_NEIGHBOURS):
    """Fit LOF once and return its scores and flags."""
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=CONTAMINATION)
    flags = model.fit_predict(matrix) == -1
    return -model.negative_outlier_factor_, pd.Series(flags)


def collect_flags(matrix: np.ndarray) -> pd.DataFrame:
    """Run all three methods at a common contamination and return their flags."""
    _, isolation = isolation_forest_scores(matrix)
    _, local = lof_scores(matrix)
    return pd.DataFrame(
        {
            "Z-score (Mahalanobis)": zscore_global(matrix).to_numpy(),
            "Isolation Forest": isolation.to_numpy(),
            "LOF": local.to_numpy(),
        }
    )


def describe_hyperparameters() -> pd.DataFrame:
    """Every setting used in chapter 5, with the class that consumes it.

    The assignment requires methodological choices to be stated. Recording the
    scikit-learn class alongside each configuration means the notebook itself
    documents exactly what was run, without the reader having to open a module.
    """
    from sklearn.ensemble import IsolationForest as _IsolationForest
    from sklearn.impute import SimpleImputer as _SimpleImputer
    from sklearn.neighbors import LocalOutlierFactor as _LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler as _StandardScaler

    rows = [
        ("Preprocessing: imputation", _SimpleImputer.__name__,
         "strategy='median'",
         "Fills the 100 placeholder-derived missing values; median because "
         "section 2.2 showed the features are skewed."),
        ("Preprocessing: scaling", _StandardScaler.__name__,
         "default (zero mean, unit variance)",
         "Required by every distance-based method here; section 6.3 shows LOF "
         "loses 80% of its flags without it."),
        ("Z-score, feature-wise", "numpy / pandas (no estimator)",
         f"threshold |z| > {ZSCORE_THRESHOLD:g}",
         "Conventional 3-sigma cut; section 5.1 shows the Gaussian basis for it "
         "does not hold for any feature."),
        ("Z-score, global", "numpy.linalg.pinv (Mahalanobis)",
         f"top {CONTAMINATION:.0%} by squared distance",
         "Covariance-aware alternative to the feature-wise rule; cut at the same "
         "share as the other two methods."),
        ("Isolation Forest", _IsolationForest.__name__,
         f"n_estimators={ISOLATION_TREES}, contamination={CONTAMINATION}, "
         f"random_state={RANDOM_STATE}, n_jobs=-1",
         "300 trees for a stable path-length average; contamination is a cut on "
         "the ranking, not a discovery (section 5.2)."),
        ("Local Outlier Factor", _LocalOutlierFactor.__name__,
         f"n_neighbors={LOF_NEIGHBOURS}, contamination={CONTAMINATION}",
         "k=20 sits inside the usual 10-50 range; section 5.3 measures how much "
         "the flagged set moves with it."),
    ]
    return pd.DataFrame(
        rows, columns=["Step", "scikit-learn class", "Configuration", "Why this value"]
    ).set_index("Step")
