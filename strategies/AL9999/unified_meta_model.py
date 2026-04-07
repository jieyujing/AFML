"""
unified_meta_model.py - Unified meta model training + inference.

A single LightGBM classifier trained on an event-candidate panel where each row = (event_time, candidate).

Key features:
- Purging/embargo walk-forward validation prevents label horizon overlap leakage
- LGBMClassifier with sigmoid calibration
- Event-level deduplication at inference
"""
import os
import sys
import numpy as np
import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# FIX #7: purge_by_label_overlap — clarify logic
def purge_by_label_overlap(
    train_df: pd.DataFrame,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Remove training samples whose label_end_time falls inside the validation time window.

    The training set must not contain samples whose outcome is determined by
    information that overlaps with the validation period (information leakage).

    Rule: keep only samples where label_end_time <= val_start.
    Samples resolved before validation begins are safe to train on.

    This is a conservative (aggressive) purge. A less aggressive version
    could use: label_end_time < val_start (strictly before) or
    label_end_time < val_end with additional embargo.
    """
    if 'label_end_time' not in train_df.columns:
        return train_df

    before_val = train_df['label_end_time'] <= val_start
    is_nan = train_df['label_end_time'].isna()

    n_removed = len(train_df) - (before_val | is_nan).sum()
    if n_removed > 0:
        print(f"    Purging: removed {n_removed} rows overlapping validation [{val_start} ~ {val_end}]")

    return train_df[before_val | is_nan].copy()


# FIX #5 from user feedback: embargo_after rewritten around val_start.
# In expanding window walk-forward, train is ALWAYS before val.
# The correct embargo target is val_start — remove train samples near val_start
# to create a temporal gap that prevents leakage from train tail into val.
# val_end is NOT used because in expanding window there is no train data after val_end
# within the same fold.
def embargo_after(
    train_df: pd.DataFrame,
    val_start: pd.Timestamp,
    embargo_delta: pd.Timedelta,
) -> pd.DataFrame:
    """
    Remove training samples whose event_time is within embargo_delta of val_start.

    Creates a temporal gap between training and validation sets to prevent
    information leakage from the training tail "bleeding" into validation.

    In expanding window: train = [earliest .. val_start - embargo], val = [val_start .. val_end]

    This replaces the incorrect previous version that used val_end as reference.
    """
    if 'event_time' not in train_df.columns:
        return train_df

    # Embargo boundary: samples must be at least embargo_delta BEFORE val_start
    embargo_boundary = val_start - embargo_delta

    before_embargo = train_df['event_time'] < embargo_boundary

    n_removed = len(train_df) - before_embargo.sum()
    if n_removed > 0:
        print(f"    Embargo: removed {n_removed} rows within {embargo_delta} of validation start ({val_start})")

    return train_df[before_embargo].copy()


# NOTE: this is time-uniform, NOT event-count-uniform — see docstring
def build_time_fraction_walk_forward_folds(
    event_times: pd.Series,
    n_folds: int = 6,
    val_horizon_days: int = 90,
    min_train_events: int = 50,
) -> list:
    """
    Build walk-forward expanding window folds using time-fraction boundaries.

    NOTE (FIX #7 from user feedback): this is time-uniform, NOT event-count-uniform.
    If event density varies significantly over time, fold sample sizes will vary.
    This is acceptable as a first pass; for stricter balance, consider event-count
    quantile-based boundaries in a future iteration.

    :param event_times: Series of event timestamps (sorted)
    :param n_folds: Number of validation windows
    :param val_horizon_days: Duration of each validation window in days
    :param min_train_events: Minimum number of training events per fold
    :returns: List of (train_idx, val_idx, fold_id, val_start, val_end, train_end) tuples
    """
    # FIX #blocker1: convert to pd.Timestamp to ensure .date() method is available
    # event_times.unique() returns numpy.datetime64 which has no .date() method
    sorted_times = pd.to_datetime(event_times.sort_values().unique())
    n_total = len(sorted_times)

    val_duration = pd.Timedelta(days=val_horizon_days)

    # Determine fold boundaries by time fractions (not event-count fractions)
    # Fold i: train = [earliest .. time_at_frac_i), val = [time_at_frac_i .. time_at_frac_i + val_duration)
    # This gives time-uniform folds; sample counts may vary
    fold_boundaries = []
    for fold_id in range(n_folds + 1):
        frac = fold_id / n_folds
        time_at_frac = sorted_times[0] + (sorted_times[-1] - sorted_times[0]) * frac
        fold_boundaries.append(time_at_frac)

    folds = []
    for fold_id in range(n_folds):
        train_end = fold_boundaries[fold_id]
        val_start = fold_boundaries[fold_id + 1]
        val_end = val_start + val_duration

        if val_end > sorted_times[-1]:
            continue  # skip if validation extends beyond data

        train_mask = event_times < train_end
        val_mask = (event_times >= val_start) & (event_times < val_end)

        n_train_rows = train_mask.sum()
        n_val_rows = val_mask.sum()

        n_train_events = event_times[train_mask].nunique()
        n_val_events = event_times[val_mask].nunique()

        if n_train_events < min_train_events or n_val_events < 10:
            continue  # skip folds with too few unique events

        train_idx = event_times[train_mask].index
        val_idx = event_times[val_mask].index

        folds.append((train_idx, val_idx, fold_id, val_start, val_end, train_end))
        # FIX #blocker1: pd.Timestamp ensures .date() method is available
        print(f"  Fold {fold_id}: train_rows={n_train_rows}, train_events={n_train_events}, "
              f"val_rows={n_val_rows}, val_events={n_val_events} | "
              f"val=[{pd.Timestamp(val_start).date()} ~ {pd.Timestamp(val_end).date()}]")

    return folds


if __name__ == "__main__":
    # Quick test
    print("Testing purging and embargo functions...")
    train = pd.DataFrame({
        'event_time': pd.to_datetime(['2020-01-01', '2020-01-10', '2020-01-20', '2020-02-05', '2020-03-01']),
        'label_end_time': pd.to_datetime(['2020-01-05', '2020-01-25', '2020-02-15', '2020-03-01', '2020-04-01']),
        'value': [1, 2, 3, 4, 5],
    })
    val_start = pd.Timestamp('2020-02-01')
    val_end = pd.Timestamp('2020-03-01')

    purged = purge_by_label_overlap(train, val_start, val_end)
    print(f"Purged: {len(purged)} rows (expected 2)")
    assert len(purged) == 2

    embargoed = embargo_after(train, val_start, pd.Timedelta(days=20))
    print(f"Embargoed: {len(embargoed)} rows (expected 2)")
    assert len(embargoed) == 2

    print("All tests passed!")


# ============================================================
# Training Section
# ============================================================

import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, brier_score_loss, confusion_matrix,
)
import joblib
import json
import openpyxl

META_MODEL_DIR = "strategies/AL9999/output/meta_model"

FEATURE_COLS = [
    # Candidate identity
    'candidate_id', 'cusum_rate', 'fast_window', 'slow_window', 'vertical_bars',
    # Event-level
    'shock_size', 'ret_1', 'vol_jump', 'ewm_vol', 'vol_percentile',
    'time_since_prev_event_h',
    # Trend-state
    'ma_gap_pct', 'fast_slope', 'slow_slope', 'price_to_slow',
    # Regime
    'session_bucket', 'dow',
    # Ensemble
    'n_fired', 'n_long', 'n_short', 'conflict_flag',
    'agreement_ratio', 'is_majority_side',
]

CATEGORICAL_FEATURES = ['candidate_id', 'session_bucket', 'dow']


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in FEATURE_COLS if c in df.columns]


def compute_primary_baselines(val_df: pd.DataFrame) -> dict:
    """Compute three baseline precisions for comparing with meta model.

    Three baselines for comparing meta model precision:

    1. row_precision: fraction of meta_y=1 across all rows (naive row-level baseline)

    2. event_precision (OPTIMISTIC): for each event, check if ANY candidate succeeded.
    This is optimistic because it treats an event as "correct" if at least one of
    its candidate rows happened to be correct — even if that candidate was not the
    one the meta model would have selected.

    3. event_majority_precision (REALISTIC): for each event, check if the
    majority-side candidate succeeded. Better comparison for per_event_max_p deduplication
    because it simulates "select by majority side" as a baseline strategy.

    Use event_majority_precision as the primary comparison baseline for precision_lift.
    """
    y_row = val_df['meta_y']
    row_prec = y_row.mean()

    # OPTIMISTIC: event correct if any candidate succeeded
    ev_prec = val_df.groupby('event_time')['meta_y'].max().mean()

    # REALISTIC: event correct if majority-side candidate succeeded
    def majority_side_success(g):
        majority = 1 if (g['primary_side'] == 1).sum() > (g['primary_side'] == -1).sum() else -1
        rows = g.loc[g['primary_side'] == majority]
        return rows['meta_y'].mean() if len(rows) > 0 else 0

    ev_majority_prec = val_df.groupby('event_time').apply(majority_side_success).mean()

    return {
        'row_precision': row_prec,
        'event_precision': ev_prec,        # optimistic
        'event_majority_precision': ev_majority_prec,  # realistic
    }


def fit_lgbm_walk_forward(
    df: pd.DataFrame,
    n_folds: int = 6,
    val_horizon_days: int = 90,
    embargo_delta: pd.Timedelta = pd.Timedelta(days=30),
) -> tuple:
    """
    Walk-forward CV with purging + embargo.
    Uses LGBMClassifier (sklearn-compatible) so CalibratedClassifierCV works natively.

    :returns: (oof_df, fold_reports, final_model_for_inference)
    """
    from strategies.AL9999.config import META_MODEL_CONFIG

    df = df.sort_values('event_time').reset_index(drop=True)
    event_times = df['event_time']
    feature_cols = get_feature_columns(df)
    cat_features = [c for c in CATEGORICAL_FEATURES if c in feature_cols]

    folds = build_time_fraction_walk_forward_folds(
        event_times, n_folds=n_folds,
        val_horizon_days=val_horizon_days,
        min_train_events=META_MODEL_CONFIG['min_train_events'],
    )

    oof_rows = []
    fold_reports = []
    # FIX #blocker2: initialize before loop to avoid UnboundLocalError when folds is empty
    final_model_for_inference = None

    for fold_i, (train_idx, val_idx, fold_id, val_start, val_end, train_end) in enumerate(folds):
        print(f"\n  === Fold {fold_id} ===")
        print(f"    Train: {str(train_end)[:10]} ({len(train_idx)} rows) | Val: {str(val_start)[:10]} ~ {str(val_end)[:10]}")

        train_df = df.loc[train_idx].copy()
        val_df = df.loc[val_idx].copy()

        # Purging + embargo
        train_df = purge_by_label_overlap(train_df, val_start, val_end)
        train_df = embargo_after(train_df, val_start, embargo_delta)

        X_tr = train_df[feature_cols].fillna(0).astype(np.float32)
        y_tr = train_df['meta_y'].values.astype(int)
        X_va = val_df[feature_cols].fillna(0).astype(np.float32)
        y_va = val_df['meta_y'].values.astype(int)

        print(f"    After purge/embargo: train={len(X_tr)} rows | val={len(X_va)} rows")

        # Sample weight: 1/n_fired to reduce intra-event double counting
        w_tr = (1.0 / train_df['n_fired'].values).astype(np.float32)

        # Primary baselines
        baselines = compute_primary_baselines(val_df)

        # LGBMClassifier (sklearn-compatible, works with CalibratedClassifierCV)
        # FIX #8 from user feedback: categorical_feature must be in fit(), not __init__()
        model = LGBMClassifier(
            **META_MODEL_CONFIG['lgbm_params'],
            n_estimators=2000,
        )

        # CalibratedClassifierCV: FIX #3 - skip calibration when too small
        # cv=1 is not recommended; instead skip calibration entirely when training set < 200
        # FIX #4: ensemble=False by default — single estimator is easier to save/load/explain
        calibration_method = META_MODEL_CONFIG.get('calibration_method', 'sigmoid')
        n_train = len(X_tr)
        if n_train >= 300:
            cal_cv = 3
            cal_model = CalibratedClassifierCV(
                estimator=model,
                method=calibration_method,
                cv=cal_cv,
                ensemble=False,  # v1 default: single estimator is easier to save/load/explain
            )
        elif n_train >= 200:
            cal_cv = 2
            cal_model = CalibratedClassifierCV(
                estimator=model,
                method=calibration_method,
                cv=cal_cv,
                ensemble=False,
            )
        else:
            # Too small for calibration — skip and use raw model
            print(f"    Warning: train set ({n_train}) < 200, skipping calibration")
            cal_model = None

        # FIX #8: pass categorical_feature via set_params(), not fit() kwargs
        # LightGBM requires column indices (not names) for categorical_feature
        cat_indices = [list(X_tr.columns).index(c) for c in cat_features if c in X_tr.columns]
        model.set_params(categorical_feature=cat_indices)

        if cal_model is not None:
            cal_model.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
            )
        else:
            # Fit raw model directly (no calibration)
            model.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
            )
            cal_model = model  # use raw model when calibration skipped

        # Predict on validation (works for both calibrated and raw model)
        p_va = cal_model.predict_proba(X_va)[:, 1]

        # Add probabilities back to val_df
        val_df = val_df.copy().reset_index(drop=True)
        val_df['p_meta'] = p_va

        # Row-level metrics
        row_auc = roc_auc_score(y_va, p_va) if len(np.unique(y_va)) > 1 else np.nan
        row_ap = average_precision_score(y_va, p_va)
        row_brier = brier_score_loss(y_va, p_va)
        row_precision = precision_score(y_va, (p_va >= 0.5).astype(int), zero_division=0)

        # Event-level metrics: per_event_max_p deduplication
        idx_best = val_df.groupby('event_time')['p_meta'].idxmax()
        deduped = val_df.loc[idx_best]

        y_dedup = deduped['meta_y'].values
        p_dedup = deduped['p_meta'].values
        ev_precision = precision_score(y_dedup, (p_dedup >= 0.5).astype(int), zero_division=0)
        ev_recall = recall_score(y_dedup, (p_dedup >= 0.5).astype(int), zero_division=0)
        ev_auc = roc_auc_score(y_dedup, p_dedup) if len(np.unique(y_dedup)) > 1 else np.nan

        print(f"    Row: AUC={row_auc:.4f} AP={row_ap:.4f} | Event: Prec={ev_precision:.4f} Rec={ev_recall:.4f}")

        # Per-fold baselines
        baselines_dedup = compute_primary_baselines(val_df)

        fold_reports.append({
            'fold_id': fold_id,
            'train_end': str(train_end)[:10],
            'val_start': str(val_start)[:10],
            'val_end': str(val_end)[:10],
            'n_train_rows': len(X_tr),
            'n_val_rows': len(X_va),
            'n_val_events': val_df['event_time'].nunique(),
            'base_rate': y_va.mean(),
            # Primary baselines (for comparison with meta)
            'primary_row_precision': baselines['row_precision'],
            'primary_event_precision': baselines['event_precision'],
            'primary_event_majority_precision': baselines['event_majority_precision'],
            # Meta metrics
            'meta_row_precision': row_precision,
            'meta_row_auc': row_auc,
            'meta_row_ap': row_ap,
            'meta_row_brier': row_brier,
            'meta_event_precision': ev_precision,
            'meta_event_recall': ev_recall,
            'meta_event_auc': ev_auc,
            'precision_lift_vs_row': ev_precision / baselines['row_precision'] if baselines['row_precision'] > 0 else np.nan,
            'precision_lift_vs_event': ev_precision / baselines['event_majority_precision'] if baselines['event_majority_precision'] > 0 else np.nan,
            'best_iteration': cal_model.best_iteration_ if hasattr(cal_model, 'best_iteration_') else 0,
        })

        # Save OOF
        oof_row = val_df[['event_time', 'event_id', 'combo_id', 'candidate_id',
                           'primary_side', 'meta_y', 'p_meta']].copy()
        oof_row['fold_id'] = fold_id
        oof_rows.append(oof_row)

        # FIX #blocker2: update after each successful fold training
        final_model_for_inference = cal_model

    # FIX #blocker2: check if any fold was successfully trained
    if final_model_for_inference is None or len(oof_rows) == 0:
        raise ValueError(
            "No valid walk-forward folds were produced. "
            "Reduce min_train_events / val_horizon_days / embargo_days, or inspect event density."
        )

    oof_df = pd.concat(oof_rows, ignore_index=True)
    return oof_df, fold_reports, final_model_for_inference


def main():
    print("=" * 70)
    print("  AL9999 Unified Meta Model Training")
    print("=" * 70)

    os.makedirs(META_MODEL_DIR, exist_ok=True)

    # Step 1: Load training table
    print("\n[Step 1] Loading meta training table...")
    df = pd.read_parquet(os.path.join(META_MODEL_DIR, "meta_training_table.parquet"))
    print(f"  Loaded {len(df)} rows, {df['event_time'].nunique()} unique events")

    # Step 2: Walk-forward CV
    print("\n[Step 2] Walk-forward CV with purging/embargo...")
    from strategies.AL9999.config import META_MODEL_CONFIG
    embargo_delta = pd.Timedelta(days=META_MODEL_CONFIG['embargo_days'])
    oof_df, fold_reports, final_model_for_inference = fit_lgbm_walk_forward(
        df,
        n_folds=META_MODEL_CONFIG['n_folds'],
        val_horizon_days=META_MODEL_CONFIG['val_horizon_days'],
        embargo_delta=embargo_delta,
    )

    # Step 3: Save OOF
    print("\n[Step 3] Saving OOF predictions...")
    oof_df.to_parquet(os.path.join(META_MODEL_DIR, "meta_oof_signals.parquet"), index=False)
    print(f"  OOF saved: {len(oof_df)} rows")

    # Step 4: Save fold reports (parquet)
    reports_df = pd.DataFrame(fold_reports)
    reports_df.to_parquet(os.path.join(META_MODEL_DIR, "meta_fold_reports.parquet"), index=False)

    # Step 5: Generate xlsx OOS report
    print("\n[Step 5] Generating OOS report (xlsx)...")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "OOS Report"

    headers = [
        'Fold', 'Train End', 'Val Start', 'Val End',
        'Train Rows', 'Val Rows', 'Val Events',
        'Base Rate',
        # event_precision = optimistic (any candidate wins), event_majority_precision = realistic
        'Primary Event Prec (Optimistic)', 'Primary Event Prec (Realistic)', 'Meta Event Prec', 'Precision Lift',
        'Filtered Recall', 'Event AUC',
        'Row AUC', 'Row AP', 'Row Brier',
    ]
    ws.append(headers)
    for r in fold_reports:
        ws.append([
            r['fold_id'], r['train_end'], r['val_start'], r['val_end'],
            r['n_train_rows'], r['n_val_rows'], r['n_val_events'],
            f"{r['base_rate']:.4f}",
            f"{r['primary_event_precision']:.4f}",  # optimistic
            f"{r['primary_event_majority_precision']:.4f}",  # realistic
            f"{r['meta_event_precision']:.4f}",
            f"{r['precision_lift_vs_event']:.4f}",  # vs realistic
            f"{r['meta_event_recall']:.4f}",
            f"{r['meta_event_auc']:.4f}",
            f"{r['meta_row_auc']:.4f}",
            f"{r['meta_row_ap']:.4f}",
            f"{r['meta_row_brier']:.4f}",
        ])

    wb.save(os.path.join(META_MODEL_DIR, "meta_oos_report.xlsx"))
    print(f"  OOS report saved: meta_oos_report.xlsx")

    # Step 6: Save feature importance
    feature_cols = get_feature_columns(df)
    cat_features = [c for c in CATEGORICAL_FEATURES if c in feature_cols]
    from strategies.AL9999.config import META_MODEL_CONFIG
    final_model = LGBMClassifier(**META_MODEL_CONFIG['lgbm_params'], n_estimators=500)
    # Use column indices for categorical features
    cat_indices = [feature_cols.index(c) for c in cat_features if c in feature_cols]
    final_model.set_params(categorical_feature=cat_indices)
    final_model.fit(df[feature_cols].fillna(0).astype(np.float32), df['meta_y'])
    imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': final_model.feature_importances_,
    }).sort_values('importance', ascending=False)
    imp.to_csv(os.path.join(META_MODEL_DIR, "meta_feature_importance.csv"), index=False)

    # Step 7: Save splits config for reproducibility
    splits_config = {
        'n_folds': META_MODEL_CONFIG['n_folds'],
        'val_horizon_days': META_MODEL_CONFIG['val_horizon_days'],
        'embargo_days': META_MODEL_CONFIG['embargo_days'],
        'min_train_events': META_MODEL_CONFIG['min_train_events'],
        'feature_cols': feature_cols,
    }
    with open(os.path.join(META_MODEL_DIR, "meta_splits.json"), 'w') as f:
        json.dump(splits_config, f, indent=2)

    # Step 8: Save final model (from last fold) for inference
    print("\n[Step 8] Saving final model...")
    joblib.dump(final_model_for_inference, os.path.join(META_MODEL_DIR, "meta_calibrator_v1.pkl"))
    print(f"  Saved calibrator: meta_calibrator_v1.pkl")

    print("\n" + "=" * 70)
    print("  Unified Meta Model Training Complete")
    print("=" * 70)
    print(f"\nOOS Summary:")
    for r in fold_reports:
        print(f"  Fold {r['fold_id']}: Meta Prec={r['meta_event_precision']:.4f} vs Primary(realistic)={r['primary_event_majority_precision']:.4f} | Lift(vs realistic)={r['precision_lift_vs_event']:.4f}")


if __name__ == "__main__":
    # If run directly, execute main training
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        main()
    else:
        # Default: run quick tests
        print("Testing purging and embargo functions...")
        train = pd.DataFrame({
            'event_time': pd.to_datetime(['2020-01-01', '2020-01-10', '2020-01-20', '2020-02-05', '2020-03-01']),
            'label_end_time': pd.to_datetime(['2020-01-05', '2020-01-25', '2020-02-15', '2020-03-01', '2020-04-01']),
            'value': [1, 2, 3, 4, 5],
        })
        val_start = pd.Timestamp('2020-02-01')
        val_end = pd.Timestamp('2020-03-01')

        purged = purge_by_label_overlap(train, val_start, val_end)
        print(f"Purged: {len(purged)} rows (expected 2)")
        assert len(purged) == 2

        embargoed = embargo_after(train, val_start, pd.Timedelta(days=20))
        print(f"Embargoed: {len(embargoed)} rows (expected 2)")
        assert len(embargoed) == 2

        print("All tests passed!")