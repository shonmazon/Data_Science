"""Two small experiments supporting the critical reflection in chapter 7.

Both address claims the assignment asks to be discussed, and both are measured
rather than asserted: distance concentration in high dimensions, and the
distortion introduced by projecting to two dimensions.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA


def _nearest_and_farthest(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance from each point to its nearest and farthest other point."""
    distances = squareform(pdist(matrix))
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    np.fill_diagonal(distances, -np.inf)
    farthest = distances.max(axis=1)
    return nearest, farthest


def distance_concentration(
    dimensions=(2, 5, 8, 20, 50, 100, 500), n_points: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Measure how nearest and farthest distances converge as dimension grows.

    Independent Gaussian points are used rather than the dataset, so that the
    only thing changing is dimensionality. If the nearest neighbour is almost as
    far away as the farthest, no method built on distance can distinguish them.
    """
    generator = np.random.default_rng(seed)
    rows = []
    for dimension in dimensions:
        points = generator.normal(size=(n_points, dimension))
        distances = pdist(points)
        nearest, farthest = _nearest_and_farthest(points)
        rows.append(
            {
                "Dimension": dimension,
                "Mean pairwise distance": round(float(distances.mean()), 2),
                "Relative contrast (max-min)/min": round(
                    float((distances.max() - distances.min()) / distances.min()), 2
                ),
                "Mean nearest / farthest": round(float(np.mean(nearest / farthest)), 3),
            }
        )
    return pd.DataFrame(rows).set_index("Dimension")


def observed_contrast(matrix: np.ndarray) -> float:
    """The same nearest-to-farthest ratio, measured on the real dataset."""
    nearest, farthest = _nearest_and_farthest(matrix)
    return float(np.mean(nearest / farthest))


def projection_distortion(
    matrix: np.ndarray, names: pd.Series, n_examples: int = 4, percentile: float = 90
) -> tuple[pd.DataFrame, float, np.ndarray, np.ndarray]:
    """Find pairs that appear adjacent in 2D but are distant in the full space.

    These pairs are the concrete form of the trustworthiness figure reported in
    section 3.4: they are what a reader of the projection would wrongly conclude
    is a close relationship.
    """
    projected = PCA(n_components=2, random_state=42).fit_transform(matrix)
    projected_distances = squareform(pdist(projected))
    full_distances = squareform(pdist(matrix))
    upper = full_distances[np.triu_indices_from(full_distances, 1)]
    threshold = np.percentile(upper, percentile)

    np.fill_diagonal(projected_distances, np.inf)
    order = np.unravel_index(np.argsort(projected_distances, axis=None), projected_distances.shape)

    rows, seen = [], set()
    for first, second in zip(*order):
        if first >= second or (first, second) in seen:
            continue
        seen.add((first, second))
        if full_distances[first, second] > threshold:
            rows.append(
                {
                    "Game A": names.iloc[first][:34],
                    "Game B": names.iloc[second][:34],
                    "Distance in 2D": round(float(projected_distances[first, second]), 4),
                    "Distance in 8D": round(float(full_distances[first, second]), 2),
                    "Percentile of that 8D distance": round(
                        float((upper < full_distances[first, second]).mean() * 100), 1
                    ),
                }
            )
        if len(rows) >= n_examples:
            break

    return pd.DataFrame(rows), float(np.median(upper)), projected_distances, full_distances
