"""Experiments comparing the three anomaly detection methods.

Each property the assignment asks about — scaling, dimensionality, runtime,
robustness — is measured by an experiment rather than argued from the methods'
definitions.
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor

from .anomaly import (
    CONTAMINATION,
    ISOLATION_TREES,
    LOF_NEIGHBOURS,
    RANDOM_STATE,
    isolation_forest_scores,
    lof_scores,
    mahalanobis_distance,
    zscore_global,
)
from .data_setup import build_matrix

METHOD_NAMES = ["Z-score (Mahalanobis)", "Isolation Forest", "LOF"]


def flag_all(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Run the three methods at the shared contamination and return boolean flags."""
    return {
        "Z-score (Mahalanobis)": zscore_global(matrix).to_numpy(),
        "Isolation Forest": isolation_forest_scores(matrix)[1].to_numpy(),
        "LOF": lof_scores(matrix)[1].to_numpy(),
    }


def jaccard(first: np.ndarray, second: np.ndarray) -> float:
    """Overlap of two flag sets as a share of their union."""
    union = (first | second).sum()
    return float((first & second).sum() / union) if union else float("nan")


def agreement_table(flags: dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise Jaccard agreement between the methods' flag sets."""
    names = list(flags)
    table = pd.DataFrame(index=names, columns=names, dtype=float)
    for first in names:
        for second in names:
            table.loc[first, second] = 1.0 if first == second else jaccard(flags[first], flags[second])
    return table.round(3)


def consensus_summary(flags: dict[str, np.ndarray]) -> pd.DataFrame:
    """How many games each combination of methods agrees on."""
    counts = np.sum(list(flags.values()), axis=0)
    return pd.DataFrame(
        [
            ("Flagged by all three", int((counts == 3).sum())),
            ("Flagged by exactly two", int((counts == 2).sum())),
            ("Flagged by exactly one", int((counts == 1).sum())),
            ("Flagged by at least one", int((counts >= 1).sum())),
            ("Flagged by none", int((counts == 0).sum())),
        ],
        columns=["Outcome", "Games"],
    ).set_index("Outcome"), counts


def scaling_sensitivity(frame: pd.DataFrame, reference: np.ndarray) -> pd.DataFrame:
    """Compare flags under three preprocessing choices.

    Two distinct questions are separated here. Standardisation is a *linear*
    rescaling of each column; the log transform is a *non-linear* change of
    shape. A method can be invariant to the first and highly sensitive to the
    second, and the table shows exactly that.
    """
    from .dimensionality import prepare_matrix

    impute = SimpleImputer(strategy="median")
    unscaled = impute.fit_transform(build_matrix(frame, log_transform=True))
    raw_scaled = prepare_matrix(frame, log_transform=False)

    reference_flags = flag_all(reference)
    variants = {
        "Unscaled (log applied, no standardisation)": flag_all(unscaled),
        "Raw scale (standardised, no log)": flag_all(raw_scaled),
    }
    return pd.DataFrame(
        {
            label: {method: round(jaccard(reference_flags[method], variant[method]), 3)
                    for method in METHOD_NAMES}
            for label, variant in variants.items()
        }
    ).rename_axis("Method")


def dimensionality_sensitivity(
    matrix: np.ndarray, extra_dimensions=(0, 4, 12, 42, 92)
) -> pd.DataFrame:
    """Re-run each method after appending pure-noise dimensions.

    The added columns are independent Gaussian noise and contain no information,
    so any change in the flagged set is the method losing its grip on the
    original eight dimensions rather than discovering anything.
    """
    generator = np.random.default_rng(0)
    reference = flag_all(matrix)
    rows = []
    for extra in extra_dimensions:
        padded = (
            np.hstack([matrix, generator.normal(size=(len(matrix), extra))])
            if extra else matrix
        )
        current = flag_all(padded)
        rows.append(
            {"Noise dimensions added": extra, "Total dimensions": matrix.shape[1] + extra,
             **{method: round(jaccard(reference[method], current[method]), 3)
                for method in METHOD_NAMES}}
        )
    return pd.DataFrame(rows).set_index("Noise dimensions added")


def noise_robustness(matrix: np.ndarray, noise_levels=(0.0, 0.1, 0.25, 0.5)) -> pd.DataFrame:
    """Perturb every value slightly and measure how much the flag set survives."""
    generator = np.random.default_rng(1)
    reference = flag_all(matrix)
    rows = []
    for level in noise_levels:
        perturbed = matrix + generator.normal(0, level, matrix.shape)
        current = flag_all(perturbed)
        rows.append(
            {"Noise SD (standard deviations)": level,
             **{method: round(jaccard(reference[method], current[method]), 3)
                for method in METHOD_NAMES}}
        )
    return pd.DataFrame(rows).set_index("Noise SD (standard deviations)")


def runtime_scaling(matrix: np.ndarray, sizes=(1500, 6000, 24000, 96000)) -> pd.DataFrame:
    """Time each method on progressively larger replicated datasets."""
    generator = np.random.default_rng(2)
    rows = []
    for size in sizes:
        repeats = max(1, size // len(matrix))
        enlarged = np.vstack(
            [matrix + generator.normal(0, 0.01, matrix.shape) for _ in range(repeats)]
        )[:size]

        started = time.perf_counter()
        mahalanobis_distance(enlarged)
        mahalanobis_time = time.perf_counter() - started

        started = time.perf_counter()
        IsolationForest(n_estimators=ISOLATION_TREES, contamination=CONTAMINATION,
                        random_state=RANDOM_STATE, n_jobs=-1).fit_predict(enlarged)
        forest_time = time.perf_counter() - started

        started = time.perf_counter()
        LocalOutlierFactor(n_neighbors=LOF_NEIGHBOURS, contamination=CONTAMINATION).fit_predict(enlarged)
        lof_time = time.perf_counter() - started

        rows.append({"Rows": size,
                     "Z-score (Mahalanobis)": round(mahalanobis_time, 4),
                     "Isolation Forest": round(forest_time, 4),
                     "LOF": round(lof_time, 4)})
    return pd.DataFrame(rows).set_index("Rows")


def method_specific_examples(
    frame: pd.DataFrame, flags: dict[str, np.ndarray], columns: list[str], n: int = 4
) -> dict[str, pd.DataFrame]:
    """Games flagged by exactly one method, which is where the methods differ."""
    examples = {}
    for method in METHOD_NAMES:
        others = [flags[other] for other in METHOD_NAMES if other != method]
        unique = flags[method] & ~others[0] & ~others[1]
        examples[method] = frame.loc[unique, columns].head(n)
    return examples
