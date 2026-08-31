"""Clustering of the samples and of the feature space, with evaluation.

Every parameter is swept rather than assumed. The point of the chapter is to
establish whether structure exists, and choosing parameters that produce an
attractive partition would answer that question by construction.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from .data_setup import MODEL_FEATURES

RANDOM_STATE = 42

# Chosen from the sweeps in sections 4.1 to 4.3 and recorded here so that every
# reported result has its configuration in one place.
KMEANS_K = 3
DBSCAN_EPS = 1.8
DBSCAN_MIN_SAMPLES = 16
HIERARCHICAL_K = 3
HIERARCHICAL_LINKAGE = "ward"


def sweep_kmeans(matrix: np.ndarray, k_values=range(2, 11)) -> pd.DataFrame:
    """Score K-Means across a range of k on four internal criteria."""
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(matrix)
        rows.append(
            {
                "k": k,
                "Inertia": round(model.inertia_, 0),
                "Silhouette": round(silhouette_score(matrix, model.labels_), 3),
                "Calinski-Harabasz": round(calinski_harabasz_score(matrix, model.labels_), 1),
                "Davies-Bouldin": round(davies_bouldin_score(matrix, model.labels_), 3),
                "Smallest cluster": int(np.bincount(model.labels_).min()),
            }
        )
    return pd.DataFrame(rows).set_index("k")


def k_distance_curve(matrix: np.ndarray, n_neighbors: int = DBSCAN_MIN_SAMPLES) -> np.ndarray:
    """Sorted distance to the k-th nearest neighbour, used to choose DBSCAN's eps.

    The conventional heuristic reads eps off the knee of this curve: below the
    knee most points have a neighbour that close, above it they do not.
    """
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors).fit(matrix).kneighbors(matrix)
    return np.sort(distances[:, -1])


def sweep_dbscan(
    matrix: np.ndarray, eps_values=(1.5, 1.8, 2.0, 2.2, 2.5, 3.0),
    min_samples_values=(8, 16, 24),
) -> pd.DataFrame:
    """Score DBSCAN across the two parameters that define it."""
    rows = []
    for eps in eps_values:
        for min_samples in min_samples_values:
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(matrix)
            clustered = labels != -1
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            score = (
                silhouette_score(matrix[clustered], labels[clustered])
                if n_clusters > 1 and clustered.sum() > n_clusters
                else None
            )
            rows.append(
                {
                    "eps": eps,
                    "min_samples": min_samples,
                    "Clusters": n_clusters,
                    "Noise %": f"{(~clustered).mean():.1%}",
                    "Largest cluster %": (
                        f"{np.bincount(labels[clustered]).max() / len(matrix):.1%}"
                        if n_clusters > 0 else "-"
                    ),
                    "Silhouette (excl. noise)": round(score, 3) if score is not None else None,
                }
            )
    return pd.DataFrame(rows)


def compare_linkages(matrix: np.ndarray, k: int = HIERARCHICAL_K) -> pd.DataFrame:
    """Compare linkage rules by cophenetic correlation and resulting partition.

    Cophenetic correlation measures how faithfully the dendrogram's implied
    distances reproduce the real pairwise distances. A low value means the tree
    is imposing a hierarchy the data does not have.
    """
    condensed = pdist(matrix)
    rows = []
    for method in ("ward", "complete", "average", "single"):
        tree = linkage(matrix, method=method) if method == "ward" else linkage(condensed, method=method)
        labels = fcluster(tree, k, criterion="maxclust")
        rows.append(
            {
                "Linkage": method,
                "Cophenetic r": round(float(cophenet(tree, condensed)[0]), 3),
                f"Silhouette at k={k}": round(silhouette_score(matrix, labels), 3)
                if len(set(labels)) > 1 else None,
                "Cluster sizes": str(np.bincount(labels)[1:].tolist()),
            }
        )
    return pd.DataFrame(rows).set_index("Linkage")


def fit_all(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Fit the three chosen configurations and return their labels."""
    tree = linkage(matrix, method=HIERARCHICAL_LINKAGE)
    return {
        f"K-Means (k={KMEANS_K})": KMeans(
            n_clusters=KMEANS_K, n_init=20, random_state=RANDOM_STATE
        ).fit_predict(matrix),
        f"DBSCAN (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})": DBSCAN(
            eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES
        ).fit_predict(matrix),
        f"Ward hierarchical (k={HIERARCHICAL_K})": fcluster(
            tree, HIERARCHICAL_K, criterion="maxclust"
        ) - 1,
    }


def agreement_matrix(labellings: dict[str, np.ndarray]) -> pd.DataFrame:
    """Adjusted Rand Index between every pair of labellings.

    The ARI is 1 for identical partitions and 0 for the agreement expected by
    chance, so it is comparable across methods that produce different numbers
    of clusters.
    """
    names = list(labellings)
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    for first in names:
        for second in names:
            matrix.loc[first, second] = (
                1.0 if first == second
                else adjusted_rand_score(labellings[first], labellings[second])
            )
    return matrix.round(3)


def variance_ratio(matrix: np.ndarray, labels: np.ndarray) -> float:
    """Mean within-cluster variance divided by the global variance.

    This is the metric the assignment names. Noise points are excluded, since
    DBSCAN's -1 label is not a cluster.
    """
    clustered = labels != -1
    global_variance = matrix[clustered].var(axis=0).mean()
    within = [
        matrix[clustered][labels[clustered] == cluster].var(axis=0).mean()
        for cluster in np.unique(labels[clustered])
    ]
    return float(np.mean(within) / global_variance)


def evaluate_partitions(
    matrix: np.ndarray, labellings: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Score every partition on the named metric and on three alternatives."""
    rows = []
    for name, labels in labellings.items():
        clustered = labels != -1
        distinct = len(set(labels[clustered]))
        rows.append(
            {
                "Partition": name,
                "Clusters": distinct,
                "Variance ratio": round(variance_ratio(matrix, labels), 3),
                "Silhouette": round(silhouette_score(matrix[clustered], labels[clustered]), 3)
                if distinct > 1 else None,
                "Calinski-Harabasz": round(
                    calinski_harabasz_score(matrix[clustered], labels[clustered]), 1
                ) if distinct > 1 else None,
                "Davies-Bouldin": round(
                    davies_bouldin_score(matrix[clustered], labels[clustered]), 3
                ) if distinct > 1 else None,
            }
        )
    return pd.DataFrame(rows).set_index("Partition")


def profile_clusters(
    matrix: np.ndarray, labels: np.ndarray, frame: pd.DataFrame
) -> pd.DataFrame:
    """Describe each cluster by its feature means and by held-out annotations."""
    scaled = pd.DataFrame(matrix, columns=MODEL_FEATURES).assign(cluster=labels)
    profile = scaled.groupby("cluster").mean().round(2)
    profile.insert(0, "Games", scaled.groupby("cluster").size())
    annotations = pd.DataFrame(
        {
            "Indie share": frame.assign(cluster=labels).groupby("cluster")["plotClass"]
            .apply(lambda column: (column.astype(str) == "Indie").mean()).round(3),
            "Median revenue": frame.assign(cluster=labels).groupby("cluster")["revenue"]
            .median().round(0),
        }
    )
    return profile.join(annotations)


def feature_distance_matrix(matrix: np.ndarray) -> pd.DataFrame:
    """Distance between features, defined as one minus absolute correlation.

    Clustering the transpose needs a notion of distance between columns.
    Correlation distance is the natural choice: two features are close when they
    carry the same information, regardless of the sign of the relationship.
    """
    correlation = pd.DataFrame(matrix, columns=MODEL_FEATURES).corr()
    distance = pd.DataFrame(
        1 - correlation.abs().to_numpy().copy(), index=MODEL_FEATURES, columns=MODEL_FEATURES
    )
    for feature in MODEL_FEATURES:
        distance.loc[feature, feature] = 0.0
    return distance


def cluster_features(distance: pd.DataFrame, k_values=(2, 3, 4)) -> tuple[pd.DataFrame, np.ndarray, float]:
    """Hierarchically cluster the feature space and report the groupings."""
    condensed = squareform(distance.to_numpy(), checks=False)
    tree = linkage(condensed, method="average")
    cophenetic = float(cophenet(tree, condensed)[0])

    rows = []
    for k in k_values:
        labels = fcluster(tree, k, criterion="maxclust")
        rows.append(
            {
                "Groups": k,
                "Composition": " | ".join(
                    ", ".join(np.array(MODEL_FEATURES)[labels == group])
                    for group in sorted(set(labels))
                ),
            }
        )
    return pd.DataFrame(rows).set_index("Groups"), tree, cophenetic


def feature_isolation(distance: pd.DataFrame) -> pd.DataFrame:
    """Rank features by how far they sit from every other feature."""
    return pd.DataFrame(
        {
            "Closest other feature": {
                feature: distance.loc[feature].drop(feature).idxmin()
                for feature in distance.index
            },
            "Distance to it": {
                feature: round(distance.loc[feature].drop(feature).min(), 2)
                for feature in distance.index
            },
            "Mean distance to all others": {
                feature: round(distance.loc[feature].drop(feature).mean(), 3)
                for feature in distance.index
            },
        }
    ).sort_values("Mean distance to all others", ascending=False)
