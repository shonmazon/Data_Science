"""PCA and the diagnostics chapter 3 needs.

The preprocessing is kept in a pipeline so that imputation and standardisation
are described once and applied identically wherever a scaled matrix is required.
"""

import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import trustworthiness
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_setup import MODEL_FEATURES, build_matrix

RANDOM_STATE = 42


def build_preprocessor() -> Pipeline:
    """Median imputation followed by standardisation.

    Standardisation is not optional here. PCA maximises variance, and the eight
    features are measured in dollars, hours, days, counts and a binary flag; on
    their native scales the component structure would simply report which column
    has the largest units.
    """
    return Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )


def prepare_matrix(frame: pd.DataFrame, log_transform: bool = True) -> np.ndarray:
    """Return the imputed, standardised matrix used by every method downstream."""
    return build_preprocessor().fit_transform(build_matrix(frame, log_transform))


def variance_table(matrix: np.ndarray) -> pd.DataFrame:
    """Explained variance ratio and its cumulative sum, per component."""
    model = PCA().fit(matrix)
    ratios = model.explained_variance_ratio_
    return pd.DataFrame(
        {
            "Eigenvalue": model.explained_variance_.round(3),
            "Explained variance": ratios.round(4),
            "Cumulative": ratios.cumsum().round(4),
        },
        index=[f"PC{i}" for i in range(1, len(ratios) + 1)],
    ).rename_axis("Component")


def components_needed(matrix: np.ndarray, targets=(0.80, 0.90, 0.95)) -> pd.DataFrame:
    """How many components reach each variance target, plus the Kaiser count."""
    model = PCA().fit(matrix)
    cumulative = model.explained_variance_ratio_.cumsum()
    rows = [
        {"Criterion": f"{target:.0%} of variance retained",
         "Components": int(np.searchsorted(cumulative, target) + 1)}
        for target in targets
    ]
    rows.append(
        {"Criterion": "Kaiser rule (eigenvalue > 1)",
         "Components": int((model.explained_variance_ > 1).sum())}
    )
    return pd.DataFrame(rows).set_index("Criterion")


def loading_table(matrix: np.ndarray, n_components: int = 4) -> pd.DataFrame:
    """Loadings of the leading components on the original features."""
    model = PCA(n_components=n_components).fit(matrix)
    return pd.DataFrame(
        model.components_.T,
        index=MODEL_FEATURES,
        columns=[f"PC{i}" for i in range(1, n_components + 1)],
    ).round(3)


def reconstruction_error(matrix: np.ndarray, component_counts=(2, 3, 4, 5, 6)) -> pd.DataFrame:
    """Per-feature error when the data is rebuilt from k components.

    A single "information lost" percentage hides which variables were lost. The
    per-feature root mean squared error, in standard-deviation units because the
    matrix is standardised, says exactly what a projection discards.
    """
    rows = []
    for k in component_counts:
        model = PCA(n_components=k).fit(matrix)
        rebuilt = model.inverse_transform(model.transform(matrix))
        errors = np.sqrt(((matrix - rebuilt) ** 2).mean(axis=0))
        rows.append(
            {"Components": k,
             "Variance kept": f"{model.explained_variance_ratio_.sum():.1%}",
             **{feature: round(error, 2) for feature, error in zip(MODEL_FEATURES, errors)}}
        )
    return pd.DataFrame(rows).set_index("Components")


def neighbourhood_preservation(
    matrix: np.ndarray, dimensions=(2, 3, 5), n_neighbors: int = 15,
    sample_size: int = 600,
) -> pd.DataFrame:
    """How faithfully a projection preserves each point's local neighbourhood.

    Trustworthiness asks what fraction of the k nearest neighbours in the
    projection were also near in the original space. It measures local structure,
    which explained variance, a purely global quantity, says nothing about.
    """
    generator = np.random.default_rng(0)
    sample = generator.choice(len(matrix), min(sample_size, len(matrix)), replace=False)
    rows = []
    for dimension in dimensions:
        projected = PCA(n_components=dimension, random_state=RANDOM_STATE).fit_transform(matrix)
        rows.append(
            {"Projection": f"{dimension}D",
             f"Trustworthiness (k={n_neighbors})":
                 round(trustworthiness(matrix[sample], projected[sample], n_neighbors=n_neighbors), 3)}
        )
    return pd.DataFrame(rows).set_index("Projection")


def bootstrap_stability(
    matrix: np.ndarray, n_components: int = 3, n_resamples: int = 200
) -> pd.DataFrame:
    """Test whether the component axes survive resampling of the rows.

    Each resample refits PCA and the new axis is compared with the full-sample
    axis by absolute cosine similarity, which is 1 for an identical direction and
    0 for a perpendicular one. The sign is discarded because a component's
    orientation is arbitrary.
    """
    generator = np.random.default_rng(RANDOM_STATE)
    reference = PCA(n_components=n_components).fit(matrix).components_

    similarities = {index: [] for index in range(n_components)}
    for _ in range(n_resamples):
        rows = generator.choice(len(matrix), len(matrix), replace=True)
        resampled = PCA(n_components=n_components).fit(matrix[rows]).components_
        for index in range(n_components):
            similarities[index].append(abs(float(np.dot(resampled[index], reference[index]))))

    return pd.DataFrame(
        [
            {"Component": f"PC{index + 1}",
             "Mean |cosine|": round(float(np.mean(similarities[index])), 3),
             "5th percentile": round(float(np.percentile(similarities[index], 5)), 3),
             "Resamples below 0.9": int((np.array(similarities[index]) < 0.9).sum())}
            for index in range(n_components)
        ]
    ).set_index("Component"), similarities


def timing_table(matrix: np.ndarray, repeats: int = 20) -> pd.DataFrame:
    """Fit time for PCA at several output dimensionalities."""
    rows = []
    for n_components in (2, 3, len(MODEL_FEATURES)):
        started = time.perf_counter()
        for _ in range(repeats):
            PCA(n_components=n_components).fit(matrix)
        rows.append(
            {"Components": n_components,
             f"Seconds for {repeats} fits": round(time.perf_counter() - started, 4)}
        )
    return pd.DataFrame(rows).set_index("Components")
