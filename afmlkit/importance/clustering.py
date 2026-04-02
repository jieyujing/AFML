"""
Feature clustering module for AFML-style importance analysis.

Implements hierarchical feature clustering based on correlation-distance matrices,
following the methodology in *Advances in Financial Machine Learning* (López de Prado, 2018).

Key functions:
- ``get_feature_distance_matrix``: Compute distance matrix D = sqrt(0.5 * (1 - ρ))
- ``hierarchical_clustering``: Agglomerative clustering via scipy linkage
- ``cluster_features``: End-to-end clustering with automatic or manual k selection
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score


def get_feature_distance_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the correlation-based distance matrix for features.

    Uses the metric-space transformation: D = sqrt(0.5 * (1 - ρ)),
    where ρ is the Pearson correlation coefficient.  This ensures the
    distance satisfies the triangle inequality.

    :param X: Feature matrix (samples × features).
    :returns: Symmetric distance matrix (features × features),
              with zero diagonal and non-negative entries.

    References
    ----------
    López de Prado, M. "Advances in Financial Machine Learning." Ch. 4.
    """
    # Remove constant features (they cause NaN in correlation)
    const_cols = X.columns[X.nunique() <= 1]
    if len(const_cols) > 0:
        X = X.drop(columns=const_cols)

    corr = X.corr()
    # Fill NaN correlations (from constant features removed above) with 0
    corr = corr.fillna(0.0)
    # Clip to [-1, 1] to handle numerical precision issues
    corr = corr.clip(-1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))
    return dist


def hierarchical_clustering(
    dist: pd.DataFrame,
    method: str = "ward",
) -> np.ndarray:
    """
    Perform agglomerative hierarchical clustering on a distance matrix.

    :param dist: Symmetric distance matrix (features × features),
                 as returned by :func:`get_feature_distance_matrix`.
    :param method: Linkage method passed to ``scipy.cluster.hierarchy.linkage``.
                   Defaults to ``'ward'``.
    :returns: Linkage matrix (scipy format).

    See Also
    --------
    scipy.cluster.hierarchy.linkage
    """
    # Convert full distance matrix to condensed form for scipy
    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, method=method)
    return link


def _find_optimal_k(
    dist: pd.DataFrame,
    link: np.ndarray,
    k_range: tuple[int, int] | None = None,
) -> int:
    """
    Find the optimal number of clusters via Silhouette Score maximization.

    :param dist: Distance matrix (features × features).
    :param link: Linkage matrix from :func:`hierarchical_clustering`.
    :param k_range: (min_k, max_k) search range.  Defaults to (2, min(n_features-1, 20)).
    :returns: Optimal number of clusters.
    """
    n = dist.shape[0]
    if k_range is None:
        k_min = 2
        k_max = min(n - 1, 20)
    else:
        k_min, k_max = k_range

    if k_min < 2:
        k_min = 2
    if k_max >= n:
        k_max = n - 1
    if k_min > k_max:
        return k_min

    best_k = k_min
    best_score = -1.0
    dist_arr = dist.values

    for k in range(k_min, k_max + 1):
        labels = fcluster(link, t=k, criterion="maxclust")
        # Need at least 2 distinct clusters for silhouette
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(dist_arr, labels, metric="precomputed")
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def cluster_features(
    X: pd.DataFrame,
    n_clusters: int | None = None,
    method: str = "ward",
    k_range: tuple[int, int] | None = None,
) -> dict[int, list[str]]:
    """
    End-to-end feature clustering pipeline.

    If ``n_clusters`` is provided, cut the dendrogram at that level.
    Otherwise, automatically determine the optimal k via Silhouette Score.

    :param X: Feature matrix (samples × features).
    :param n_clusters: Number of clusters.  If ``None``, auto-select via Silhouette.
    :param method: Linkage method (default ``'ward'``).
    :param k_range: Search range for automatic k selection.
    :returns: Dictionary mapping cluster_id → list of feature names.

    Examples
    --------
    >>> clusters = cluster_features(X_features)
    >>> for cid, feats in clusters.items():
    ...     print(f"Cluster {cid}: {feats}")
    """
    dist = get_feature_distance_matrix(X)
    link = hierarchical_clustering(dist, method=method)

    if n_clusters is None:
        n_clusters = _find_optimal_k(dist, link, k_range=k_range)
        print(f"[Feature Clustering] Auto-selected k={n_clusters} (Silhouette)")

    labels = fcluster(link, t=n_clusters, criterion="maxclust")
    feature_names = X.columns.tolist()

    clusters: dict[int, list[str]] = {}
    for feat, label in zip(feature_names, labels):
        clusters.setdefault(int(label), []).append(feat)

    print(f"[Feature Clustering] {len(clusters)} clusters from {len(feature_names)} features")
    for cid in sorted(clusters):
        print(f"  Cluster {cid}: {clusters[cid]}")

    return clusters
