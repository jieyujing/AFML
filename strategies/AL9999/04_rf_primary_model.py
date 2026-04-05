"""
04_rf_primary_model.py - AL9999 RF Primary Model.

Train RF-style primary direction model from trend scanning labels and
engineered event features, then export side/probability signals.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.AL9999.config import FEATURES_DIR, MODELS_DIR, RF_PRIMARY_CONFIG
from afmlkit.label.weights import average_uniqueness
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


def prepare_rf_dataset(features, labels, feature_prefixes, min_t_value):
    """
    Build aligned RF dataset from features and trend labels.

    :param features: Event features DataFrame.
    :param labels: Trend labels DataFrame with side/t_value/t1.
    :param feature_prefixes: Prefix filter list.
    :param min_t_value: Minimum absolute t_value threshold.
    :returns: (X, y, t1, meta)
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
    return X, y, t1, meta


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


def split_train_holdout(X, y, t1, holdout_months):
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

    return (
        X.loc[train_mask],
        y.loc[train_mask],
        t1.loc[train_mask],
    ), (
        X.loc[holdout_mask],
        y.loc[holdout_mask],
        t1.loc[holdout_mask],
    ), holdout_start


def build_model(n_estimators, random_state, n_jobs, max_samples):
    """
    Build bagging model with AFML-friendly tree base learner.
    """
    base_tree = DecisionTreeClassifier(
        criterion="entropy",
        max_features=1,
        min_samples_leaf=5,
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


def build_signal_frame(index, trend_side, probs, t1, is_holdout):
    """
    Build final RF signal DataFrame.
    """
    probs = probs.astype(float).clip(0.0, 1.0)
    y_pred = (probs >= 0.5).astype(int)
    side = y_pred.replace({0: -1, 1: 1}).astype(int)
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


def _oof_predict(model, X_train, y_train, t1_train, n_splits, embargo_pct):
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

        if y_tr.nunique() < 2:
            fill_value = 1.0 if int(y_tr.iloc[0]) == 1 else 0.0
            probs.iloc[test_idx] = fill_value
            continue

        model.fit(X_tr, y_tr)
        cls_idx = int(np.where(model.classes_ == 1)[0][0])
        probs.iloc[test_idx] = model.predict_proba(X_te)[:, cls_idx]

    probs = probs.fillna(0.5)
    return probs


def main():
    print("=" * 70)
    print("  AL9999 RF Primary Model")
    print("=" * 70)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    features_path = os.path.join(FEATURES_DIR, "events_features.parquet")
    labels_path = os.path.join(FEATURES_DIR, "trend_labels.parquet")

    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)

    X, y, t1, meta = prepare_rf_dataset(
        features=features,
        labels=labels,
        feature_prefixes=RF_PRIMARY_CONFIG.get("feature_prefixes", []),
        min_t_value=float(RF_PRIMARY_CONFIG.get("min_t_value", 0.0)),
    )

    if len(X) < 20:
        raise ValueError(f"RF training samples too few: {len(X)}")
    if y.nunique() < 2:
        raise ValueError("RF training labels have only one class after filtering.")

    (X_train, y_train, t1_train), (X_holdout, y_holdout, t1_holdout), _ = split_train_holdout(
        X=X,
        y=y,
        t1=t1,
        holdout_months=int(RF_PRIMARY_CONFIG.get("holdout_months", 12)),
    )

    avg_u = compute_avg_uniqueness_from_t1(X_train.index, t1_train)
    max_samples = avg_u if RF_PRIMARY_CONFIG.get("max_samples_method") == "avgU" else 0.5
    max_samples = float(np.clip(max_samples, 0.01, 1.0))

    model = build_model(
        n_estimators=int(RF_PRIMARY_CONFIG.get("n_estimators", 1000)),
        random_state=int(RF_PRIMARY_CONFIG.get("random_state", 42)),
        n_jobs=int(RF_PRIMARY_CONFIG.get("n_jobs", -1)),
        max_samples=max_samples,
    )

    oof_probs = _oof_predict(
        model=model,
        X_train=X_train,
        y_train=y_train,
        t1_train=t1_train,
        n_splits=int(RF_PRIMARY_CONFIG.get("cv_n_splits", 5)),
        embargo_pct=float(RF_PRIMARY_CONFIG.get("cv_embargo_pct", 0.01)),
    )

    model.fit(X_train, y_train)
    if len(X_holdout) > 0:
        cls_idx = int(np.where(model.classes_ == 1)[0][0])
        holdout_probs = pd.Series(
            model.predict_proba(X_holdout)[:, cls_idx],
            index=X_holdout.index,
            dtype=float,
        )
    else:
        holdout_probs = pd.Series(dtype=float)

    all_probs = pd.concat([oof_probs, holdout_probs]).sort_index()
    is_holdout = pd.Series(False, index=X.index)
    is_holdout.loc[X_holdout.index] = True
    signal_df = build_signal_frame(
        index=X.index,
        trend_side=meta.loc[X.index, "trend_side"],
        probs=all_probs.reindex(X.index).fillna(0.5),
        t1=t1.reindex(X.index),
        is_holdout=is_holdout.reindex(X.index).fillna(False),
    )

    rf_signals_path = os.path.join(FEATURES_DIR, "rf_primary_signals.parquet")
    signal_df.to_parquet(rf_signals_path)

    model_path = os.path.join(MODELS_DIR, "rf_primary.pkl")
    joblib.dump(model, model_path)

    cv_report = pd.DataFrame(
        [
            {
                "n_samples": int(len(X)),
                "n_train": int(len(X_train)),
                "n_holdout": int(len(X_holdout)),
                "avg_uniqueness": float(avg_u),
                "max_samples": float(max_samples),
                "oof_accuracy": float(accuracy_score(y_train, (oof_probs >= 0.5).astype(int).replace({0: -1, 1: 1}))),
                "oof_f1": float(f1_score((y_train == 1).astype(int), (oof_probs >= 0.5).astype(int))),
                "holdout_accuracy": float(
                    accuracy_score(
                        y_holdout,
                        (holdout_probs >= 0.5).astype(int).replace({0: -1, 1: 1}),
                    )
                ) if len(X_holdout) > 0 else np.nan,
            }
        ]
    )
    cv_report_path = os.path.join(FEATURES_DIR, "rf_primary_cv_report.parquet")
    cv_report.to_parquet(cv_report_path)

    print(f"✅ RF 信号已保存: {rf_signals_path}")
    print(f"✅ RF 模型已保存: {model_path}")
    print(f"✅ CV 报告已保存: {cv_report_path}")


if __name__ == "__main__":
    main()
