"""
Clustered Mean Decrease Accuracy (MDA) for feature importance analysis.

Implements the Clustered MDA methodology from *Advances in Financial Machine
Learning* (López de Prado, 2018, Ch. 8).  Key improvements over vanilla MDA:

- Feature perturbation at the **cluster level** to avoid substitution effects.
- Uses **Log-loss** (with sample weights) instead of accuracy as the default
  scoring metric, capturing probability calibration quality.
- Integrates with :class:`~afmlkit.validation.purged_cv.PurgedKFold` for
  leak-free cross-validation.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

from afmlkit.validation.purged_cv import PurgedKFold


# ======================================================================
# Scoring wrapper
# ======================================================================

def _score_log_loss(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: np.ndarray | None = None,
) -> float:
    """
    Compute negative log-loss (higher is better, consistent with sklearn convention).

    Uses ``model.predict_proba`` and supports ``sample_weight``.

    :param model: Fitted classifier with ``predict_proba`` method.
    :param X: Feature matrix.
    :param y: True labels.
    :param sample_weight: Per-sample weights.
    :returns: Negative log-loss (higher = better).
    """
    proba = model.predict_proba(X)
    labels = model.classes_
    ll = log_loss(y, proba, labels=labels, sample_weight=sample_weight)
    return -ll  # negate so that higher = better


# ======================================================================
# Clustered MDA
# ======================================================================

def clustered_mda(
    X: pd.DataFrame,
    y: pd.Series,
    clusters: dict[int, list[str]],
    t1: pd.Series,
    sample_weight: pd.Series | np.ndarray | None = None,
    classifier: BaseEstimator | None = None,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
    n_repeats: int = 1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute Clustered Mean Decrease Accuracy using Log-loss and PurgedKFold.

    For each cluster of features, the algorithm:

    1. Trains a classifier on un-permuted data (baseline).
    2. Shuffles **all features in the cluster together** on the OOS fold.
    3. Computes the **drop in negative log-loss** relative to baseline.
    4. Aggregates across all folds and repeats.

    :param X: Feature matrix (samples × features).
    :param y: Label series (same index as X).
    :param clusters: Dictionary ``{cluster_id: [feature_name, ...]}``.
    :param t1: Event end-time series for Purged CV (same index as X).
    :param sample_weight: Per-sample weights for weighted log-loss.
    :param classifier: sklearn classifier.  Defaults to ``RandomForestClassifier``.
    :param n_splits: Number of CV folds (default 5).
    :param embargo_pct: Embargo fraction (default 0.01).
    :param n_repeats: Number of permutation repeats per cluster per fold.
    :param random_state: Random seed for reproducibility.
    :returns: DataFrame with columns ``['cluster_id', 'features', 'mean_importance',
              'std_importance']``, sorted by importance descending.

    References
    ----------
    López de Prado, M. "Advances in Financial Machine Learning." Ch. 8.
    """
    if classifier is None:
        classifier = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            max_depth=5,
            random_state=random_state,
            n_jobs=-1,
        )

    rng = np.random.RandomState(random_state)

    cv = PurgedKFold(n_splits=n_splits, t1=t1, embargo_pct=embargo_pct)

    # Accumulate importance scores per cluster
    importance_scores: dict[int, list[float]] = defaultdict(list)

    fold_idx = 0
    for train_idx, test_idx in cv.split(X, y):
        fold_idx += 1
        print(f"\n[Clustered MDA] === Fold {fold_idx}/{n_splits} ===")

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        sw_train = (
            sample_weight.iloc[train_idx]
            if sample_weight is not None
            else None
        )
        sw_test = (
            sample_weight.iloc[test_idx]
            if sample_weight is not None
            else None
        )

        # --- Fit baseline model ---
        model = clone(classifier)
        if sw_train is not None:
            model.fit(X_train, y_train, sample_weight=sw_train)
        else:
            model.fit(X_train, y_train)

        baseline_score = _score_log_loss(model, X_test, y_test, sw_test)
        print(f"  Baseline neg-log-loss: {baseline_score:.6f}")

        # --- Permute each cluster and measure drop ---
        for cid, feat_names in clusters.items():
            # Filter to features actually present in X
            valid_feats = [f for f in feat_names if f in X_test.columns]
            if not valid_feats:
                continue

            drops = []
            for rep in range(n_repeats):
                X_test_perm = X_test.copy()
                # Shuffle all features in the cluster TOGETHER
                # (preserve intra-cluster correlation structure during shuffle)
                perm_idx = rng.permutation(X_test_perm.shape[0])
                X_test_perm[valid_feats] = X_test_perm[valid_feats].values[perm_idx]

                perm_score = _score_log_loss(model, X_test_perm, y_test, sw_test)
                drop = baseline_score - perm_score  # positive = feature was useful
                drops.append(drop)

            mean_drop = float(np.mean(drops))
            importance_scores[cid].append(mean_drop)
            print(
                f"  Cluster {cid} ({valid_feats}): "
                f"drop={mean_drop:.6f}"
            )

    # --- Aggregate across folds ---
    results = []
    for cid in sorted(importance_scores):
        scores = importance_scores[cid]
        results.append(
            {
                "cluster_id": cid,
                "features": clusters[cid],
                "mean_importance": float(np.mean(scores)),
                "std_importance": float(np.std(scores)),
            }
        )

    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values("mean_importance", ascending=False).reset_index(
        drop=True
    )

    print("\n[Clustered MDA] Final Importance Ranking:")
    for _, row in df_result.iterrows():
        print(
            f"  Cluster {row['cluster_id']}: "
            f"importance={row['mean_importance']:.6f} ± {row['std_importance']:.6f}  "
            f"features={row['features']}"
        )

    return df_result
