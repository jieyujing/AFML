"""
07b_meta_model_feature_selection.py - MDA 正贡献簇特征裁剪重训对比

对比两个方案：
1) full: 使用全部 feat_ 特征
2) mda_positive: 仅使用 meta_mda_importance.parquet 中 mean_importance > 0 的簇内特征

输出：
- output/features/07b_mda_pruned_comparison.parquet
- output/features/07b_mda_pruned_comparison.csv
- output/models/meta_model_full.pkl
- output/models/meta_model_mda_positive.pkl
- output/models/meta_oof_signals_full.parquet
- output/models/meta_oof_signals_mda_positive.parquet
- output/models/meta_holdout_signals_full.parquet
- output/models/meta_holdout_signals_mda_positive.parquet
"""

import argparse
import ast
import os
import sys
from typing import Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.AL9999.backtest_utils import calculate_performance, load_dollar_bars, rolling_backtest
from afmlkit.validation.purged_cv import PurgedKFold
from strategies.AL9999.config import FEATURES_DIR, META_MODEL_CONFIG

ANNUALIZATION_FACTOR = 1500
N_TRIALS = 214
COMMISSION_RATE = 0.000023
SLIPPAGE_POINTS = 0.5


def _ensure_list(value) -> List[str]:
    """Normalize parquet object column to list[str]."""
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, tuple):
        return [str(x) for x in value]
    if isinstance(value, np.ndarray):
        return [str(x) for x in value.tolist()]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    return [str(x) for x in parsed]
            except (SyntaxError, ValueError):
                pass
        return [text]
    return []


def split_train_holdout(
    X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series, holdout_months: int
):
    """Split train/holdout by time."""
    holdout_start = X.index.max() - pd.DateOffset(months=holdout_months)
    train_mask = X.index < holdout_start
    holdout_mask = X.index >= holdout_start

    X_train = X[train_mask]
    y_train = y[train_mask]
    w_train = sample_weight[train_mask]

    X_holdout = X[holdout_mask]
    y_holdout = y[holdout_mask]
    w_holdout = sample_weight[holdout_mask]

    return (X_train, y_train, w_train), (X_holdout, y_holdout, w_holdout), holdout_start


def load_data():
    """Load features, labels, t1 and aligned TBM frame."""
    features = pd.read_parquet(os.path.join(FEATURES_DIR, "events_features.parquet"))
    labels = pd.read_parquet(os.path.join(FEATURES_DIR, "meta_labels.parquet"))
    tbm = pd.read_parquet(os.path.join(FEATURES_DIR, "tbm_results.parquet"))

    idx = features.index.intersection(labels.index).intersection(tbm.index)
    X = features.loc[idx].copy()
    y = labels.loc[idx, "bin"].copy()
    sample_weight = labels.loc[idx, "sample_weight"].copy().fillna(labels["sample_weight"].mean())
    t1 = pd.to_datetime(tbm.loc[idx, "exit_ts"], errors="coerce")
    if t1.isna().any():
        t1 = t1.fillna(pd.Series(X.index, index=X.index) + pd.Timedelta(days=1))

    feat_cols = [c for c in X.columns if c.startswith("feat_")]
    X = X[feat_cols].fillna(0.0)

    return X, y, sample_weight, t1, tbm.loc[idx].copy()


def load_positive_features(all_features: Sequence[str]) -> List[str]:
    """Get features from positive MDA clusters."""
    path = os.path.join(FEATURES_DIR, "meta_mda_importance.parquet")
    mda = pd.read_parquet(path)
    pos = mda[mda["mean_importance"] > 0].copy()

    keep = set()
    for value in pos["features"].tolist():
        for feat in _ensure_list(value):
            keep.add(feat)

    ordered = [f for f in all_features if f in keep]
    if len(ordered) == 0:
        raise ValueError("MDA 正贡献簇未筛到任何可用特征，请检查 meta_mda_importance.parquet。")
    return ordered


def build_model() -> BaggingClassifier:
    """Create AFML-style meta model."""
    base_tree = DecisionTreeClassifier(
        criterion="entropy",
        max_features=1,
        max_depth=5,
        class_weight="balanced",
        random_state=42,
    )
    return BaggingClassifier(
        estimator=base_tree,
        n_estimators=META_MODEL_CONFIG.get("n_estimators", 1000),
        max_samples=0.5,
        max_features=1.0,
        n_jobs=-1,
        random_state=42,
    )


def calculate_psr(returns: pd.Series, benchmark_sr: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio."""
    returns = returns.dropna()
    n = len(returns)
    if n < 5:
        return 0.0
    std = returns.std()
    if std == 0:
        return 0.0
    sr = returns.mean() / std
    skew = returns.skew()
    kurt = returns.kurtosis() + 3
    sigma_sr = np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1))
    return float(norm.cdf((sr - benchmark_sr) / sigma_sr))


def calculate_dsr(returns: pd.Series, n_trials: int = N_TRIALS) -> float:
    """Deflated Sharpe Ratio."""
    returns = returns.dropna()
    if len(returns) < 5:
        return 0.0
    sr_std = 0.5
    gamma = 0.5772
    exp_max_sr_annual = sr_std * (
        (1 - gamma) * norm.ppf(1 - 1 / n_trials)
        + gamma * norm.ppf(1 - 1 / (n_trials * np.exp(-1)))
    )
    benchmark_sr_period = exp_max_sr_annual / np.sqrt(ANNUALIZATION_FACTOR)
    return calculate_psr(returns, benchmark_sr_period)


def evaluate_backtest_by_probs(
    tbm: pd.DataFrame, probs: pd.Series, oos_start: pd.Timestamp, threshold: float
) -> Dict[str, float]:
    """Evaluate Combined strategy with the real single-position backtest."""
    eval_df = tbm.loc[tbm.index.intersection(probs.index)].copy()
    eval_df["meta_prob"] = probs.loc[eval_df.index]
    eval_df["meta_pred"] = (eval_df["meta_prob"] >= threshold).astype(int)

    bars = load_dollar_bars()
    combined_trades = rolling_backtest(eval_df, bars, use_meta_filter=True)
    combined_full_perf = calculate_performance(combined_trades)

    combined_full_trades = combined_full_perf.get("trades_df", pd.DataFrame())
    combined_oos_trades = combined_full_trades[combined_full_trades["exit_time"] >= oos_start].copy()
    combined_oos_perf = calculate_performance(combined_oos_trades)

    return {
        "combined_full_n": combined_full_perf.get("n_trades", 0),
        "combined_full_sharpe": combined_full_perf.get("sharpe", 0.0),
        "combined_full_dsr": calculate_dsr(combined_full_trades["net_ret"]) if len(combined_full_trades) > 0 else 0.0,
        "combined_oos_n": combined_oos_perf.get("n_trades", 0),
        "combined_oos_sharpe": combined_oos_perf.get("sharpe", 0.0),
        "combined_oos_dsr": calculate_dsr(combined_oos_trades["net_ret"]) if len(combined_oos_trades) > 0 else 0.0,
        "combined_oos_total_pnl": combined_oos_perf.get("total_pnl", 0.0),
    }


def train_scheme(
    scheme: str,
    feature_cols: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    w_train: pd.Series,
    t1_train: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    threshold: float,
):
    """Train one scheme and return metrics/model/predictions."""
    model = build_model()
    cv = PurgedKFold(
        n_splits=META_MODEL_CONFIG.get("cv_n_splits", 5),
        t1=t1_train,
        embargo_pct=META_MODEL_CONFIG.get("cv_embargo_pct", 0.05),
    )

    Xtr = X_train[feature_cols]
    Xho = X_holdout[feature_cols]
    oof_probs = np.full(len(Xtr), np.nan)
    fold_f1 = []

    for train_idx, test_idx in cv.split(Xtr):
        x_tr = Xtr.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        w_tr = w_train.iloc[train_idx]
        x_te = Xtr.iloc[test_idx]
        y_te = y_train.iloc[test_idx]

        model.fit(x_tr, y_tr, sample_weight=w_tr)
        prob = model.predict_proba(x_te)[:, 1]
        oof_probs[test_idx] = prob
        pred = (prob >= 0.5).astype(int)
        fold_f1.append(f1_score(y_te, pred))

    oof_pred = (oof_probs >= threshold).astype(int)
    cv_precision = precision_score(y_train, oof_pred, zero_division=0)
    cv_recall = recall_score(y_train, oof_pred, zero_division=0)
    cv_f1 = f1_score(y_train, oof_pred, zero_division=0)

    # final fit on train only
    model.fit(Xtr, y_train, sample_weight=w_train)
    holdout_probs = model.predict_proba(Xho)[:, 1]
    holdout_pred = (holdout_probs >= threshold).astype(int)

    hold_precision = precision_score(y_holdout, holdout_pred, zero_division=0)
    hold_recall = recall_score(y_holdout, holdout_pred, zero_division=0)
    hold_f1 = f1_score(y_holdout, holdout_pred, zero_division=0)
    hold_acc = accuracy_score(y_holdout, holdout_pred)

    oof_df = pd.DataFrame(
        {"y_true": y_train, "y_prob": oof_probs, "y_pred": oof_pred},
        index=Xtr.index,
    )
    hold_df = pd.DataFrame(
        {"y_true": y_holdout, "y_prob": holdout_probs, "y_pred": holdout_pred},
        index=Xho.index,
    )

    metrics = {
        "scheme": scheme,
        "n_features": len(feature_cols),
        "cv_fold_f1_mean": float(np.mean(fold_f1)),
        "cv_fold_f1_std": float(np.std(fold_f1)),
        "cv_precision": float(cv_precision),
        "cv_recall": float(cv_recall),
        "cv_f1": float(cv_f1),
        "holdout_precision": float(hold_precision),
        "holdout_recall": float(hold_recall),
        "holdout_f1": float(hold_f1),
        "holdout_accuracy": float(hold_acc),
        "holdout_pred_positive_rate": float(np.mean(holdout_pred)),
    }

    return metrics, model, oof_df, hold_df


def main():
    parser = argparse.ArgumentParser(description="MDA 正贡献簇特征裁剪重训对比")
    parser.add_argument(
        "--apply-best",
        action="store_true",
        help="将最佳方案覆盖写入 meta_model.pkl / meta_oof_signals.parquet / meta_holdout_signals.parquet",
    )
    args = parser.parse_args()

    models_dir = os.path.join(os.path.dirname(__file__), "output", "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(FEATURES_DIR, exist_ok=True)

    X, y, sample_weight, t1, tbm = load_data()
    holdout_months = META_MODEL_CONFIG.get("holdout_months", 12)
    threshold = META_MODEL_CONFIG.get("precision_threshold", 0.5)
    (X_train, y_train, w_train), (X_holdout, y_holdout, _), holdout_start = split_train_holdout(
        X, y, sample_weight, holdout_months
    )
    t1_train = pd.Series(t1.loc[X_train.index], index=X_train.index)

    full_features = list(X.columns)
    pos_features = load_positive_features(full_features)

    print("=" * 70)
    print("  MDA 正贡献簇特征裁剪重训对比")
    print("=" * 70)
    print(f"训练样本: {len(X_train)} | Holdout样本: {len(X_holdout)}")
    print(f"阈值: {threshold:.2f} | holdout_months: {holdout_months}")
    print(f"full 特征数: {len(full_features)}")
    print(f"mda_positive 特征数: {len(pos_features)}")
    print(f"mda_positive 示例: {pos_features[:8]}")

    scheme_features = {"full": full_features, "mda_positive": pos_features}
    all_metrics = []
    stitched_prob_series = {}

    for scheme, cols in scheme_features.items():
        print(f"\n--- 训练方案: {scheme} ---")
        metrics, model, oof_df, hold_df = train_scheme(
            scheme=scheme,
            feature_cols=cols,
            X_train=X_train,
            y_train=y_train,
            w_train=w_train,
            t1_train=t1_train,
            X_holdout=X_holdout,
            y_holdout=y_holdout,
            threshold=threshold,
        )

        model_path = os.path.join(models_dir, f"meta_model_{scheme}.pkl")
        oof_path = os.path.join(models_dir, f"meta_oof_signals_{scheme}.parquet")
        hold_path = os.path.join(models_dir, f"meta_holdout_signals_{scheme}.parquet")
        joblib.dump(model, model_path)
        oof_df.to_parquet(oof_path)
        hold_df.to_parquet(hold_path)

        stitched_probs = pd.concat(
            [
                oof_df[["y_prob"]].rename(columns={"y_prob": "meta_prob"}),
                hold_df[["y_prob"]].rename(columns={"y_prob": "meta_prob"}),
            ],
            axis=0,
        )
        stitched_probs = stitched_probs[~stitched_probs.index.duplicated(keep="last")].sort_index()
        stitched_prob_series[scheme] = stitched_probs["meta_prob"]

        backtest_metrics = evaluate_backtest_by_probs(
            tbm=tbm,
            probs=stitched_prob_series[scheme],
            oos_start=holdout_start,
            threshold=threshold,
        )

        row = {**metrics, **backtest_metrics}
        all_metrics.append(row)

        print(
            f"holdout_f1={row['holdout_f1']:.4f}, "
            f"oos_sharpe={row['combined_oos_sharpe']:.4f}, "
            f"oos_dsr={row['combined_oos_dsr']:.4f}, "
            f"oos_n={row['combined_oos_n']}"
        )

    result_df = pd.DataFrame(all_metrics).sort_values(
        ["combined_oos_dsr", "combined_oos_sharpe", "holdout_f1"],
        ascending=False,
    )

    out_parquet = os.path.join(FEATURES_DIR, "07b_mda_pruned_comparison.parquet")
    out_csv = os.path.join(FEATURES_DIR, "07b_mda_pruned_comparison.csv")
    result_df.to_parquet(out_parquet, index=False)
    result_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    print("对比结果（按 OOS DSR 排序）")
    print("=" * 70)
    cols_show = [
        "scheme",
        "n_features",
        "holdout_f1",
        "holdout_precision",
        "holdout_recall",
        "combined_oos_n",
        "combined_oos_sharpe",
        "combined_oos_dsr",
        "combined_oos_total_pnl",
    ]
    print(result_df[cols_show].to_string(index=False))
    print(f"\n✅ 已保存: {out_parquet}")
    print(f"✅ 已保存: {out_csv}")

    if args.apply_best:
        best = result_df.iloc[0]["scheme"]
        print(f"\n应用最佳方案到主产物: {best}")
        src_model = os.path.join(models_dir, f"meta_model_{best}.pkl")
        src_oof = os.path.join(models_dir, f"meta_oof_signals_{best}.parquet")
        src_hold = os.path.join(models_dir, f"meta_holdout_signals_{best}.parquet")
        dst_model = os.path.join(models_dir, "meta_model.pkl")
        dst_oof = os.path.join(models_dir, "meta_oof_signals.parquet")
        dst_hold = os.path.join(models_dir, "meta_holdout_signals.parquet")

        joblib.dump(joblib.load(src_model), dst_model)
        pd.read_parquet(src_oof).to_parquet(dst_oof)
        pd.read_parquet(src_hold).to_parquet(dst_hold)

        selected_path = os.path.join(FEATURES_DIR, "selected_features.json")
        pd.Series({"best_scheme": best, "features": scheme_features[best]}).to_json(
            selected_path, force_ascii=False, indent=2
        )
        print(f"✅ 已覆盖主模型产物并记录特征: {selected_path}")


if __name__ == "__main__":
    main()
