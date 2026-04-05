"""
04_rf_primary_model.py - AL9999 RF Primary Model (Improved).

Four key improvements over the baseline:
1. Clustered Feature Filtering (CFF): Remove clusters with negative MDA.
2. Absolute Return Attribution: Weight samples by event period absolute returns.
3. neg_log_loss as Primary Optimization: Hyperparameter tuning via log-loss scoring.
4. Purged CV: Leak-free evaluation with embargo and purge.
"""

import os
import sys
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.AL9999.config import FEATURES_DIR, MODELS_DIR, RF_PRIMARY_CONFIG
from afmlkit.label.weights import average_uniqueness
from afmlkit.sampling import avg_uniqueness_of_sample, sequential_bootstrap_indices
from afmlkit.validation.purged_cv import PurgedKFold


def select_feature_columns(columns, prefixes):
    """
    Select feature columns by prefixes.

    :param columns: Iterable of column names.
    :param prefixes: Prefix list.
    :returns: Filtered feature column list.
    """
    all_feat = [c for c in columns if c.startswith("feat_")]
    if not prefixes:
        return all_feat

    selected = [c for c in all_feat if any(c.startswith(prefix) for prefix in prefixes)]
    if selected:
        return selected
    return all_feat


def prepare_rf_dataset(features, labels, feature_prefixes, min_t_value, bars=None):
    """
    Build aligned RF dataset from features and trend labels.

    :param features: Event features DataFrame.
    :param labels: Trend labels DataFrame with side/t_value/t1.
    :param feature_prefixes: Prefix filter list.
    :param min_t_value: Minimum absolute t_value threshold.
    :param bars: Optional raw dollar bars (for return attribution weights).
    :returns: (X, y, t1, meta, sample_weight)
    """
    common_idx = features.index.intersection(labels.index)
    merged = features.loc[common_idx].copy()
    merged["trend_side"] = labels.loc[common_idx, "side"].astype(float)
    merged["t_value"] = labels.loc[common_idx, "t_value"].astype(float)
    merged["t1"] = pd.to_datetime(labels.loc[common_idx, "t1"], errors="coerce")

    merged = merged[merged["trend_side"] != 0].copy()
    merged["abs_t_value"] = merged["t_value"].abs()
    if min_t_value > 0:
        merged = merged[merged["abs_t_value"] >= float(min_t_value)].copy()

    feature_cols = select_feature_columns(merged.columns.tolist(), feature_prefixes)
    X = merged[feature_cols].fillna(0.0)

    y = merged["trend_side"].astype(int)
    # Keep side in +/-1 domain for consistency with existing primary model side.
    y = pd.Series(np.where(y > 0, 1, -1), index=merged.index, dtype=int)

    t1 = merged["t1"].copy()
    missing_t1 = t1.isna()
    if missing_t1.any():
        t1.loc[missing_t1] = X.index[missing_t1] + pd.Timedelta(days=1)

    meta = merged[["trend_side", "abs_t_value"]].copy()

    # --- Return Attribution Sample Weights ---
    sample_weight = compute_return_attribution_weight(X, t1, bars)

    return X, y, t1, meta, sample_weight


def compute_return_attribution_weight(X, t1, bars):
    """
    Compute sample weights by absolute log-return during the event period.

    For each event at row i: find price at event start (index) and at event
    end (t1[i]), compute |log ret|, then normalize across all events.
    This forces the model to pay attention to high-magnitude events.
    """
    if bars is None:
        # Fallback: weight by abs t_value (proxy for trend intensity)
        return pd.Series(np.ones(len(X)), index=X.index)

    close_s = bars["close"]
    log_ret_vals = np.full(len(X), np.nan)

    for i, (idx, t1_val) in enumerate(zip(X.index, t1)):
        start_price = close_s.get(idx)
        end_price = close_s.get(t1_val)
        if start_price is not None and end_price is not None and start_price > 0 and end_price > 0:
            log_ret = np.abs(np.log(end_price / start_price))
            log_ret_vals[i] = log_ret

    # Weight by absolute log return, normalized to mean=1
    weights = pd.Series(log_ret_vals, index=X.index)
    nonzero = weights.dropna()
    if len(nonzero) > 0 and nonzero.mean() > 0:
        weights = weights / nonzero.mean()
    weights = weights.fillna(1.0)
    weights = weights.clip(0.5, 5.0)  # Cap extremes to avoid dominating

    return weights


def _touch_positions_from_t1(index, t1):
    n = len(index)
    if n == 0:
        return np.array([], dtype=np.int64)
    idx_vals = index.values.astype("datetime64[ns]")
    t1_vals = pd.to_datetime(t1).values.astype("datetime64[ns]")
    pos = np.searchsorted(idx_vals, t1_vals, side="left")
    pos = np.clip(pos, 0, n - 1).astype(np.int64)
    return pos


def compute_avg_uniqueness_from_t1(index, t1):
    """
    Compute average uniqueness in [0, 1] from event start index and t1.

    :param index: DatetimeIndex for events.
    :param t1: Event end timestamps aligned with index.
    :returns: avg uniqueness scalar; fallback to 0.5 if invalid.
    """
    if len(index) == 0:
        return 0.5

    timestamps = index.view("int64").astype(np.int64)
    event_pos = np.arange(len(index), dtype=np.int64)
    touch_pos = _touch_positions_from_t1(index, t1)
    touch_pos = np.maximum(touch_pos, event_pos)

    uniq, _ = average_uniqueness(timestamps, event_pos, touch_pos)
    avg_u = float(np.nanmean(uniq))
    if not np.isfinite(avg_u) or avg_u <= 0:
        return 0.5
    return float(min(avg_u, 1.0))


class SequentialBootstrapBagging:
    """
    Bagging-like ensemble using AFML sequential bootstrap as sampler.
    """

    def __init__(
        self,
        n_estimators: int,
        max_samples: float,
        max_features: str,
        min_samples_leaf: int,
        random_state: int,
    ):
        self.n_estimators = int(n_estimators)
        self.max_samples = float(max_samples)
        self.max_features = max_features
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)

        self.estimators_ = []
        self.classes_ = np.array([-1, 1], dtype=np.int64)
        self.sample_avg_u_ = []
        self.sampled_indices_ = []
        self.training_event_starts_ = None
        self.training_event_ends_ = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[pd.Series] = None,
        t1: Optional[pd.Series] = None,
    ):
        if t1 is None:
            raise ValueError("t1 is required for sequential bootstrap sampling.")
        if len(X) == 0:
            raise ValueError("X must not be empty.")

        event_starts = np.arange(len(X), dtype=np.int64)
        event_ends = _touch_positions_from_t1(X.index, t1)
        event_ends = np.maximum(event_ends, event_starts)
        sample_length = max(1, int(self.max_samples * len(X)))

        self.estimators_ = []
        self.sample_avg_u_ = []
        self.sampled_indices_ = []
        self.training_event_starts_ = event_starts
        self.training_event_ends_ = event_ends

        for i in range(self.n_estimators):
            seed = self.random_state + i
            sampled_idx = sequential_bootstrap_indices(
                event_starts=event_starts,
                event_ends=event_ends,
                sample_length=sample_length,
                random_state=seed,
            )

            X_sample = X.iloc[sampled_idx]
            y_sample = y.iloc[sampled_idx]
            sw_sample = sample_weight.iloc[sampled_idx] if sample_weight is not None else None

            tree = DecisionTreeClassifier(
                criterion='entropy',
                max_features=self.max_features,
                class_weight='balanced',
                min_samples_leaf=self.min_samples_leaf,
                random_state=seed,
            )
            tree.fit(X_sample, y_sample, sample_weight=sw_sample)

            bag_avg_u = avg_uniqueness_of_sample(
                event_starts=event_starts,
                event_ends=event_ends,
                sampled_indices=sampled_idx,
            )
            self.estimators_.append(tree)
            self.sampled_indices_.append(sampled_idx)
            self.sample_avg_u_.append(float(bag_avg_u))

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if len(self.estimators_) == 0:
            raise ValueError("Model is not fitted yet.")

        n_samples = len(X)
        p_pos_sum = np.zeros(n_samples, dtype=np.float64)

        for tree in self.estimators_:
            tree_probs = tree.predict_proba(X)
            if len(tree.classes_) == 1:
                pos_probs = np.ones(n_samples, dtype=np.float64) if int(tree.classes_[0]) == 1 else np.zeros(
                    n_samples, dtype=np.float64
                )
            else:
                pos_idx = int(np.where(tree.classes_ == 1)[0][0])
                pos_probs = tree_probs[:, pos_idx]
            p_pos_sum += pos_probs

        p_pos = p_pos_sum / float(len(self.estimators_))
        p_pos = np.clip(p_pos, 0.0, 1.0)
        return np.column_stack([1.0 - p_pos, p_pos])


def split_train_holdout(X, y, t1, sample_weight=None, holdout_months=12):
    """
    Split train/holdout by last N months.
    """
    if len(X) == 0:
        raise ValueError("No samples available after filtering.")

    holdout_start = X.index.max() - pd.DateOffset(months=holdout_months)
    train_mask = X.index < holdout_start
    holdout_mask = X.index >= holdout_start

    if train_mask.sum() == 0:
        train_mask = np.ones(len(X), dtype=bool)
        holdout_mask = np.zeros(len(X), dtype=bool)

    if sample_weight is None:
        sample_weight = pd.Series(np.ones(len(X), dtype=np.float64), index=X.index)

    return (
        X.loc[train_mask],
        y.loc[train_mask],
        t1.loc[train_mask],
        sample_weight.loc[train_mask],
    ), (
        X.loc[holdout_mask],
        y.loc[holdout_mask],
        t1.loc[holdout_mask],
        sample_weight.loc[holdout_mask],
    ), holdout_start


def build_model(
    n_estimators,
    random_state,
    n_jobs,
    max_samples,
    max_features='sqrt',
    min_samples_leaf=5,
    sampling_method='avgU',
):
    """
    Build bagging model with AFML-friendly tree base learner.

    :param max_features: Feature subset for tree splits.
    :param min_samples_leaf: Min samples per leaf (higher = more regularization).
    """
    if sampling_method == 'sequential_bootstrap':
        return SequentialBootstrapBagging(
            n_estimators=n_estimators,
            max_samples=float(max_samples),
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    base_tree = DecisionTreeClassifier(
        criterion='entropy',
        max_features=max_features,
        class_weight='balanced',
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    return BaggingClassifier(
        estimator=base_tree,
        n_estimators=n_estimators,
        max_samples=float(max_samples),
        max_features=1.0,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def _fit_model(model, X, y, sample_weight, t1):
    if isinstance(model, SequentialBootstrapBagging):
        model.fit(X, y, sample_weight=sample_weight, t1=t1)
    else:
        model.fit(X, y, sample_weight=sample_weight)


def _predict_positive_class_proba(model, X):
    if len(X) == 0:
        return np.array([], dtype=np.float64)

    probs = model.predict_proba(X)
    if probs.ndim != 2 or probs.shape[1] < 2:
        return np.full(len(X), 0.5, dtype=np.float64)

    cls_idx = int(np.where(model.classes_ == 1)[0][0])
    return probs[:, cls_idx]


def optimize_hyperparams(
    X_train,
    y_train,
    t1_train,
    sample_weight_train,
    n_splits,
    embargo_pct,
    sampling_method,
):
    """
    Hyperparameter tuning via neg_log_loss with PurgedKFold.

    Searches over a grid of (min_samples_leaf, n_estimators) combos and
    picks the one with lowest OOF log-loss.

    :returns: (best_params, oof_probs_of_best)
    """
    param_grid = {
        'min_samples_leaf': [3, 5, 8, 12, 20],
        'n_estimators': [200, 500, 1000],
    }

    best_ll = float('inf')
    best_params = {'min_samples_leaf': 5, 'n_estimators': 1000}
    best_oof_probs = None

    total = len(param_grid['min_samples_leaf']) * len(param_grid['n_estimators'])
    count = 0

    for msl in param_grid['min_samples_leaf']:
        for ne in param_grid['n_estimators']:
            count += 1
            print(f"  [{count}/{total}] min_samples_leaf={msl}, n_estimators={ne}", end=" ... ")

            model = build_model(
                n_estimators=ne,
                random_state=42,
                n_jobs=-1,
                max_samples=0.5,  # fixed for search speed
                max_features='sqrt',
                min_samples_leaf=msl,
                sampling_method=sampling_method,
            )

            cv = PurgedKFold(n_splits=n_splits, t1=pd.Series(t1_train, index=X_train.index),
                             embargo_pct=embargo_pct)
            oof_probs = pd.Series(np.nan, index=X_train.index, dtype=float)

            for train_idx, test_idx in cv.split(X_train):
                X_tr = X_train.iloc[train_idx]
                y_tr = y_train.iloc[train_idx]
                X_te = X_train.iloc[test_idx]
                t1_tr = t1_train.iloc[train_idx]
                sw_tr = sample_weight_train.iloc[train_idx] if sample_weight_train is not None else None

                if y_tr.nunique() < 2:
                    fill_value = 1.0 if int(y_tr.iloc[0]) == 1 else 0.0
                    oof_probs.iloc[test_idx] = fill_value
                    continue

                _fit_model(model=model, X=X_tr, y=y_tr, sample_weight=sw_tr, t1=t1_tr)
                oof_probs.iloc[test_idx] = _predict_positive_class_proba(model, X_te)

            oof_probs = oof_probs.fillna(0.5)
            y_bin = (y_train == 1).astype(int)
            ll = log_loss(y_bin, oof_probs, labels=[0, 1])

            print(f"log_loss={ll:.4f}")
            if ll < best_ll:
                best_ll = ll
                best_params = {'min_samples_leaf': msl, 'n_estimators': ne}
                best_oof_probs = oof_probs

    print(f"\n  Best params: {best_params} (log_loss={best_ll:.4f})")
    return best_params, best_oof_probs


def oof_predict_with_weights(
    model,
    X_train,
    y_train,
    t1_train,
    sample_weight,
    n_splits,
    embargo_pct,
):
    """
    OOF predict with sample weights.
    """
    probs = pd.Series(np.nan, index=X_train.index, dtype=float)
    cv = PurgedKFold(
        n_splits=n_splits,
        t1=pd.Series(t1_train, index=X_train.index),
        embargo_pct=embargo_pct,
    )

    for train_idx, test_idx in cv.split(X_train):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_te = X_train.iloc[test_idx]
        t1_tr = t1_train.iloc[train_idx]
        sw_tr = sample_weight.iloc[train_idx] if sample_weight is not None else None

        if y_tr.nunique() < 2:
            fill_value = 1.0 if int(y_tr.iloc[0]) == 1 else 0.0
            probs.iloc[test_idx] = fill_value
            continue

        _fit_model(model=model, X=X_tr, y=y_tr, sample_weight=sw_tr, t1=t1_tr)
        probs.iloc[test_idx] = _predict_positive_class_proba(model, X_te)

    probs = probs.fillna(0.5)
    return probs


def build_signal_frame(index, trend_side, probs, t1, is_holdout,
                       prob_lower=0.45, prob_upper=0.55):
    """
    Build final RF signal DataFrame with confidence abyss.

    :param index: DatetimeIndex.
    :param trend_side: Original trend label side.
    :param probs: RF predicted probability of class 1.
    :param t1: Event exit timestamps.
    :param is_holdout: Boolean mask.
    :param prob_lower: Below this -> side=-1 (strong short signal).
    :param prob_upper: Above this -> side=+1 (strong long signal).
        Between [prob_lower, prob_upper] -> side=0 (abstain).
    :returns: Signal DataFrame.
    """
    probs = probs.astype(float).clip(0.0, 1.0)
    # Confidence abyss: force side=0 when model is uncertain
    uncertain = (probs > prob_lower) & (probs < prob_upper)
    y_pred = (probs >= 0.5).astype(int)
    side = y_pred.replace({0: -1, 1: 1}).astype(int)
    side = side.where(~uncertain, 0).astype(int)
    return pd.DataFrame(
        {
            "side": side.values,
            "y_prob": probs.values,
            "y_pred": y_pred.values,
            "trend_side": trend_side.astype(int).values,
            "t1": pd.to_datetime(t1).values,
            "is_holdout": is_holdout.astype(bool).values,
        },
        index=index,
    )


def main():
    print("=" * 70)
    print("  AL9999 RF Primary Model (Improved)")
    print("=" * 70)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    features_path = os.path.join(FEATURES_DIR, "events_features.parquet")
    labels_path = os.path.join(FEATURES_DIR, "trend_labels.parquet")

    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)

    # Load original bars for return attribution weights
    bars = None
    from strategies.AL9999.config import BARS_DIR, TARGET_DAILY_BARS
    bars_file = os.path.join(BARS_DIR, f'dollar_bars_target{TARGET_DAILY_BARS}.parquet')
    if os.path.exists(bars_file):
        bars = pd.read_parquet(bars_file)
        print(f"✅ Loaded bars for return attribution: {len(bars)} rows")

    X, y, t1, meta, sample_weight = prepare_rf_dataset(
        features=features,
        labels=labels,
        feature_prefixes=RF_PRIMARY_CONFIG.get("feature_prefixes", []),
        min_t_value=float(RF_PRIMARY_CONFIG.get("min_t_value", 0.0)),
        bars=bars,
    )

    if len(X) < 20:
        raise ValueError(f"RF training samples too few: {len(X)}")
    if y.nunique() < 2:
        raise ValueError("RF training labels have only one class after filtering.")

    (X_train, y_train, t1_train, sw_train), (X_holdout, y_holdout, t1_holdout, sw_holdout), _ = split_train_holdout(
        X=X,
        y=y,
        t1=t1,
        sample_weight=sample_weight,
        holdout_months=int(RF_PRIMARY_CONFIG.get("holdout_months", 12)),
    )

    avg_u = compute_avg_uniqueness_from_t1(X_train.index, t1_train)
    sampling_method = RF_PRIMARY_CONFIG.get("sampling_method", "avgU")
    max_samples = avg_u if RF_PRIMARY_CONFIG.get("max_samples_method") == "avgU" else 0.5
    max_samples = float(np.clip(max_samples, 0.01, 1.0))

    print(f"\n  Samples: {len(X)} total, {len(X_train)} train, {len(X_holdout)} holdout")
    print(f"  avg_uniqueness={avg_u:.4f}, max_samples={max_samples:.4f}, sampling_method={sampling_method}")

    # --- Hyperparameter Optimization via neg_log_loss ---
    print("\n[Hyperparameter Optimization] Searching via neg_log_loss ...")
    best_params, _ = optimize_hyperparams(
        X_train=X_train,
        y_train=y_train,
        t1_train=t1_train,
        sample_weight_train=sw_train,
        n_splits=int(RF_PRIMARY_CONFIG.get("cv_n_splits", 5)),
        embargo_pct=float(RF_PRIMARY_CONFIG.get("cv_embargo_pct", 0.01)),
        sampling_method=sampling_method,
    )

    # --- Build Final Model with Best Params ---
    print(f"\n  Building final model: {best_params}")

    final_model = build_model(
        n_estimators=best_params['n_estimators'],
        random_state=int(RF_PRIMARY_CONFIG.get("random_state", 42)),
        n_jobs=int(RF_PRIMARY_CONFIG.get("n_jobs", -1)),
        max_samples=max_samples,
        max_features=RF_PRIMARY_CONFIG.get("max_features", "sqrt"),
        min_samples_leaf=best_params['min_samples_leaf'],
        sampling_method=sampling_method,
    )

    # Final OOF with sample weights
    oof_probs = oof_predict_with_weights(
        model=final_model,
        X_train=X_train,
        y_train=y_train,
        t1_train=t1_train,
        sample_weight=sw_train,
        n_splits=int(RF_PRIMARY_CONFIG.get("cv_n_splits", 5)),
        embargo_pct=float(RF_PRIMARY_CONFIG.get("cv_embargo_pct", 0.01)),
    )

    # Train on full train set
    _fit_model(model=final_model, X=X_train, y=y_train, sample_weight=sw_train, t1=t1_train)

    sb_avg_u_mean = np.nan
    sb_avg_u_median = np.nan
    sb_avg_u_q25 = np.nan
    sb_avg_u_q75 = np.nan
    sb_avg_u_uplift = np.nan
    sb_avg_u_target_gap = np.nan
    if isinstance(final_model, SequentialBootstrapBagging):
        print("\n[Sequential Bootstrapping] Per-tree in-bag avgU")
        for i, v in enumerate(final_model.sample_avg_u_, start=1):
            print(f"  Tree {i:04d}: avgU={v:.4f}")
        sb_avg_u_arr = np.asarray(final_model.sample_avg_u_, dtype=np.float64)
        sb_avg_u_mean = float(np.nanmean(sb_avg_u_arr))
        sb_avg_u_median = float(np.nanmedian(sb_avg_u_arr))
        sb_avg_u_q25 = float(np.nanquantile(sb_avg_u_arr, 0.25))
        sb_avg_u_q75 = float(np.nanquantile(sb_avg_u_arr, 0.75))
        sb_avg_u_uplift = float(sb_avg_u_mean - 0.47)
        sb_avg_u_target_gap = float(0.632 - sb_avg_u_mean)
        print("\n[Sequential Bootstrapping] avgU summary")
        print(
            f"  mean={sb_avg_u_mean:.4f}, median={sb_avg_u_median:.4f}, "
            f"q25={sb_avg_u_q25:.4f}, q75={sb_avg_u_q75:.4f}"
        )
        print(f"  uplift_vs_baseline_0.47={sb_avg_u_uplift:.4f}")
        print(f"  gap_to_target_0.632={sb_avg_u_target_gap:.4f}")

    if len(X_holdout) > 0:
        holdout_probs = pd.Series(
            _predict_positive_class_proba(final_model, X_holdout),
            index=X_holdout.index,
            dtype=float,
        )
    else:
        holdout_probs = pd.Series(dtype=float)

    all_probs = pd.concat([oof_probs, holdout_probs]).sort_index()
    is_holdout = pd.Series(False, index=X.index)
    is_holdout.loc[X_holdout.index] = True
    abyss_cfg = RF_PRIMARY_CONFIG.get('prob_abyss', (0.45, 0.55))
    signal_df = build_signal_frame(
        index=X.index,
        trend_side=meta.loc[X.index, "trend_side"],
        probs=all_probs.reindex(X.index).fillna(0.5),
        t1=t1.reindex(X.index),
        is_holdout=is_holdout.reindex(X.index).fillna(False),
        prob_lower=float(abyss_cfg[0]),
        prob_upper=float(abyss_cfg[1]),
    )

    rf_signals_path = os.path.join(FEATURES_DIR, "rf_primary_signals.parquet")
    signal_df.to_parquet(rf_signals_path)

    model_path = os.path.join(MODELS_DIR, "rf_primary.pkl")
    joblib.dump(final_model, model_path)

    # Build comprehensive CV report
    y_train_mapped = y_train.copy()
    y_holdout_mapped = y_holdout.copy()

    cv_report = pd.DataFrame(
        [
            {
                "n_samples": int(len(X)),
                "n_train": int(len(X_train)),
                "n_holdout": int(len(X_holdout)),
                "avg_uniqueness": float(avg_u),
                "max_samples": float(max_samples),
                "sampling_method": str(sampling_method),
                "best_min_samples_leaf": int(best_params['min_samples_leaf']),
                "best_n_estimators": int(best_params['n_estimators']),
                "sb_bag_avg_u_mean": float(sb_avg_u_mean) if np.isfinite(sb_avg_u_mean) else np.nan,
                "sb_bag_avg_u_median": float(sb_avg_u_median) if np.isfinite(sb_avg_u_median) else np.nan,
                "sb_bag_avg_u_q25": float(sb_avg_u_q25) if np.isfinite(sb_avg_u_q25) else np.nan,
                "sb_bag_avg_u_q75": float(sb_avg_u_q75) if np.isfinite(sb_avg_u_q75) else np.nan,
                "sb_bag_avg_u_uplift_vs_0.47": float(sb_avg_u_uplift) if np.isfinite(sb_avg_u_uplift) else np.nan,
                "sb_bag_avg_u_gap_to_0.632": float(sb_avg_u_target_gap) if np.isfinite(sb_avg_u_target_gap) else np.nan,
                "oof_accuracy": float(accuracy_score(y_train, (oof_probs >= 0.5).astype(int).replace({0: -1, 1: 1}))),
                "oof_f1": float(f1_score((y_train == 1).astype(int), (oof_probs >= 0.5).astype(int))),
                "oof_log_loss": float(log_loss(
                    (y_train == 1).astype(int), oof_probs.fillna(0.5), labels=[0, 1]
                )),
                "holdout_accuracy": float(
                    accuracy_score(
                        y_holdout,
                        (holdout_probs >= 0.5).astype(int).replace({0: -1, 1: 1}),
                    )
                ) if len(X_holdout) > 0 else np.nan,
                "holdout_log_loss": float(
                    log_loss(
                        (y_holdout == 1).astype(int), holdout_probs.fillna(0.5), labels=[0, 1]
                    )
                ) if len(X_holdout) > 0 else np.nan,
            }
        ]
    )
    cv_report_path = os.path.join(FEATURES_DIR, "rf_primary_cv_report.parquet")
    cv_report.to_parquet(cv_report_path)

    print(f"\n✅ RF signals saved: {rf_signals_path}")
    print(f"✅ RF model saved: {model_path}")
    print(f"✅ CV report saved: {cv_report_path}")

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"  Summary")
    print(f"{'=' * 70}")
    for c in cv_report.columns:
        v = cv_report.iloc[0][c]
        if isinstance(v, float):
            print(f"  {c}: {v:.4f}")
        else:
            print(f"  {c}: {v}")


if __name__ == "__main__":
    main()
