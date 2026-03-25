"""
Meta-Model Training Script — AFML Meta-Labeling Pipeline
==========================================================

Meta-Labeling 架构说明：
  Primary Model (RF)：高召回率预测趋势方向（+1/-1），已在 primary_model_training.py 完成。
  Meta-Model（本脚本）：预测 Primary Model 的信号是否会真正盈利（bin=1）。

输入：
  - feature_matrix.csv        : CUSUM 事件时刻的技术特征
  - primary_oof_signals.csv   : Primary RF 的 OOF 概率（prob_pos, pred_high_recall）
  - cusum_sampled_bars.csv    : TBM 标签（bin）与样本权重

Meta-Labeling 工作原理：
  1. Primary Model 生成高召回率信号（宁可多报）
  2. Meta-Model 预测 Primary 信号是否正确（过滤噪声 → 提升精度）
  3. 元标签 bin=1：Primary 信号命中三重屏障的盈利方向
  4. Primary Model 的 OOF 概率 (prob_pos) 作为 Meta-Model 的核心输入特征
     — 体现了"元模型知道主模型有多确信"这一信息优势

Usage
-----
    uv run python scripts/meta_model_training.py
"""

from __future__ import annotations

import joblib
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
    accuracy_score,
    classification_report,
)
from sklearn.calibration import calibration_curve

from afmlkit.validation.purged_cv import PurgedKFold
from afmlkit.importance.clustering import cluster_features, get_feature_distance_matrix, hierarchical_clustering
from afmlkit.importance.mda import clustered_mda
from scipy.cluster.hierarchy import dendrogram


# ── Paths ────────────────────────────────────────────────────────────
FEATURES_PATH    = Path("outputs/dollar_bars/feature_matrix.csv")
OOF_PATH         = Path("outputs/models/primary_model/primary_oof_signals.csv")
LABELS_PATH      = Path("outputs/dollar_bars/cusum_sampled_bars.csv")
OUTPUT_DIR       = Path("outputs/models/meta_model")

# ── 排除的元数据列（禁止信息泄露）───────────────────────────────────
META_COLS = [
    "open", "high", "low", "close", "volume", "vwap",
    "trades", "median_trade_size", "log_return",
    "bin", "t1", "avg_uniqueness", "return_attribution",
    "volatility", "daily_vol_est",
    "side", "t_value", "trend_confidence",
    "trend_weighted_uniqueness", "vertical_touch_weights",
    "event_idx", "touch_idx", "ret",
]

# ── 超参数 ───────────────────────────────────────────────────────────
CV_N_SPLITS    = 5
CV_EMBARGO_PCT = 0.01
N_ESTIMATORS   = 1000
RANDOM_STATE   = 42


# ─────────────────────────────────────────────────────────────────────
# 1. 数据加载
# ─────────────────────────────────────────────────────────────────────

def load_and_prepare_data():
    """
    加载特征矩阵、Primary OOF 信号、TBM 元标签，拼装 Meta-Model 训练集。

    关键设计：
      - prob_pos (Primary RF 置信度) 作为核心 Meta 特征
      - pred_high_recall 作为元特征（Primary 的硬预测）
      - 目标 y = bin (TBM 判定，Primary 信号是否真正盈利)
    """
    print(f"[Data] 加载特征矩阵: {FEATURES_PATH}")
    df_feat = pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])

    print(f"[Data] 加载 Primary OOF 信号: {OOF_PATH}")
    df_oof = pd.read_csv(OOF_PATH, parse_dates=["timestamp"])

    print(f"[Data] 加载 TBM 元标签与权重: {LABELS_PATH}")
    df_lab = pd.read_csv(LABELS_PATH, parse_dates=["timestamp"])

    # ── 合并三个数据源 ────────────────────────────────────────────
    df = pd.merge(df_feat, df_oof[["timestamp", "prob_pos", "prob_neg", "pred_high_recall"]],
                  on="timestamp", how="inner")
    df = pd.merge(df, df_lab[["timestamp", "trend_weighted_uniqueness"]],
                  on="timestamp", how="left")
    df = df.set_index("timestamp").sort_index()

    # ── 删除缺失标签行 ────────────────────────────────────────────
    if "bin" not in df.columns:
        raise ValueError("'bin' 列缺失，请先运行 cusum_filtering.py + feature_engineering.py。")
    df = df.dropna(subset=["bin", "t1"])
    df["t1"] = pd.to_datetime(df["t1"])

    # ── 只保留 Primary 模型有预测的事件 ──────────────────────────
    # OOF 仅覆盖训练周期内的样本，确保对齐
    n_total = len(df)
    df = df.dropna(subset=["prob_pos"])
    print(f"[Data] OOF 对齐：{n_total} → {len(df)} 个样本（含 Primary OOF 预测）")

    # ── 目标：TBM 元标签 ──────────────────────────────────────────
    # bin=1: Primary 信号触碰盈利屏障, bin=0: 触碰止损屏障或时间屏障
    y = df["bin"].astype(int)

    # ── 样本权重 ──────────────────────────────────────────────────
    if "trend_weighted_uniqueness" in df.columns and df["trend_weighted_uniqueness"].notna().any():
        sample_weight = df["trend_weighted_uniqueness"].fillna(
            df.get("avg_uniqueness", pd.Series(1.0, index=df.index))
        ).clip(lower=0)
        print("[Data] 使用 trend_weighted_uniqueness 作为样本权重")
    else:
        sample_weight = df.get("avg_uniqueness", pd.Series(1.0, index=df.index)).fillna(1.0).clip(lower=0)
        print("[Data] 回退: 使用 avg_uniqueness 作为样本权重")

    t1 = df["t1"]
    avg_u = df["avg_uniqueness"].mean() if "avg_uniqueness" in df.columns else 0.5

    # ── 特征选择 ──────────────────────────────────────────────────
    # 在原技术特征基础上，加入 Primary Model 的 OOF 置信度作为关键元特征
    exclude_cols = set(META_COLS)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].copy()

    # ── Robust Cleaning (重要：防止 clustering 崩溃) ──────────────────
    # 1. 删除全为 NaN 的列
    nan_cols = X.columns[X.isna().all()]
    if len(nan_cols) > 0:
        print(f"[Data] 删除全 NaN 特征: {nan_cols.tolist()}")
        X = X.drop(columns=nan_cols)

    # 2. 填充剩余 NaN
    X = X.ffill().bfill()

    # 3. 删除常数列（zero variance，会导致相关性计算出现 NaN）
    const_cols = X.columns[X.nunique() <= 1]
    if len(const_cols) > 0:
        print(f"[Data] 删除常数特征: {const_cols.tolist()}")
        X = X.drop(columns=const_cols)

    # 4. 删除非有限值 (Inf)
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    still_nan = X.columns[X.isna().any()]
    if len(still_nan) > 0:
        print(f"[Data] 删除含 persistent NaN/Inf 特征: {still_nan.tolist()}")
        X = X.drop(columns=still_nan)

    print(f"[Data] Meta 特征矩阵: {X.shape[0]} 样本 × {X.shape[1]} 特征")
    print(f"[Data] 包含 Primary 元特征: {[c for c in ['prob_pos', 'prob_neg', 'pred_high_recall'] if c in X.columns]}")
    print(f"[Data] 元标签分布 (bin): {y.value_counts().sort_index().to_dict()}")
    print(f"[Data] 平均唯一性: {avg_u:.4f}")

    return X, y, t1, sample_weight, avg_u


# ─────────────────────────────────────────────────────────────────────
# 2. 模型构建
# ─────────────────────────────────────────────────────────────────────

def build_model(mean_uniqueness: float, n_samples: int, class_ratio: float) -> BaggingClassifier:
    """
    构建 Meta-Model：唯一性约束 Bagging + Entropy 决策树。

    F1 优化关键设计：
      - class_weight 按实际频率逆比例设置（比 'balanced' 更激进）
        class_ratio = n(bin=0) / n(bin=1)，例如 12.5%/87.5% → ratio≈0.143
        → weight_0 = 1/0.143 ≈ 7x，强制模型重视少数类
      - max_features=1 防止 prob_pos 等强特征遮蔽弱技术信号
      - min_weight_fraction_leaf=0.01：足够宽松以捕捉 12.5% 的少数类
    """
    max_samples_int = max(50, int(n_samples * mean_uniqueness))
    # 少数类 (bin=0) 的权重 = 多数类权重 × 频率倒数
    weight_minority = 1.0 / max(class_ratio, 0.05)   # 防止除以极小值
    weight_majority = 1.0
    cw = {0: weight_minority, 1: weight_majority}
    print(f"[Model] Bagging max_samples     = {max_samples_int} 个样本")
    print(f"[Model] Class weights           = {{0: {weight_minority:.2f}, 1: {weight_majority:.2f}}} (少数类强化)")

    base_tree = DecisionTreeClassifier(
        criterion="entropy",
        max_features=1,              # AFML: 防止强特征遮蔽
        class_weight=cw,
        min_weight_fraction_leaf=0.01,   # 允许捕捉少数类模式
        random_state=RANDOM_STATE,
    )

    model = BaggingClassifier(
        estimator=base_tree,
        n_estimators=N_ESTIMATORS,
        max_samples=max_samples_int,
        max_features=1.0,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return model


# ─────────────────────────────────────────────────────────────────────
# 3. Purged K-Fold CV
# ─────────────────────────────────────────────────────────────────────

def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """在验证集上用 PR 曲线搜索使 Macro F1 最高的阈值。"""
    prec_arr, rec_arr, thr_arr = precision_recall_curve(y_true, y_prob)
    # f1 per threshold（precision_recall_curve 内置处理 0-div）
    f1_arr = np.where(
        (prec_arr + rec_arr) > 0,
        2 * prec_arr * rec_arr / (prec_arr + rec_arr),
        0.0,
    )
    best_idx = np.argmax(f1_arr[:-1])  # 最后一个 threshold 对应 precision=1, rec=0
    return float(thr_arr[best_idx])


def run_purged_cv(X, y, t1, sample_weight, model):
    """
    Purged K-Fold CV，每折在训练集上选 F1 最优阈值，
    然后在测试集上评估 Macro F1。
    """
    print(f"\n[CV] Purged {CV_N_SPLITS}-Fold CV (Embargo={CV_EMBARGO_PCT*100:.0f}%)  目标: Macro F1...")

    cv = PurgedKFold(n_splits=CV_N_SPLITS, t1=t1, embargo_pct=CV_EMBARGO_PCT)

    oof_prob      = np.full(len(y), np.nan)
    oof_pred_f1   = np.full(len(y), np.nan)   # 使用 F1 优化阈值的预测
    fold_thresholds = []
    metrics = {"macro_f1": [], "f1_skip": [], "f1_trade": [], "roc_auc": [], "brier": []}

    X_arr = X.values
    y_arr = y.values
    w_arr = sample_weight.values

    for fold, (train_idx, test_idx) in enumerate(cv.split(X), 1):
        X_tr, y_tr, w_tr = X_arr[train_idx], y_arr[train_idx], w_arr[train_idx]
        X_te, y_te       = X_arr[test_idx],  y_arr[test_idx]

        model.fit(X_tr, y_tr, sample_weight=w_tr)

        # 在训练集上找最优 F1 阈值（OOF 内估计，无泄露）
        train_probs_oof = model.predict_proba(X_tr)[:, 1]
        opt_thr = _best_f1_threshold(y_tr, train_probs_oof)
        fold_thresholds.append(opt_thr)

        # 用最优阈值预测测试集
        probs = model.predict_proba(X_te)[:, 1]
        preds_f1 = (probs >= opt_thr).astype(int)

        oof_prob[test_idx]    = probs
        oof_pred_f1[test_idx] = preds_f1

        auc   = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else np.nan
        brier = brier_score_loss(y_te, probs)
        f1_macro = f1_score(y_te, preds_f1, average="macro",   zero_division=0)
        f1_skip  = f1_score(y_te, preds_f1, labels=[0],        average="macro", zero_division=0)
        f1_trade = f1_score(y_te, preds_f1, labels=[1],        average="macro", zero_division=0)

        metrics["macro_f1"].append(f1_macro)
        metrics["f1_skip"].append(f1_skip)
        metrics["f1_trade"].append(f1_trade)
        metrics["roc_auc"].append(auc)
        metrics["brier"].append(brier)

        print(f"  --> Fold {fold}: thr={opt_thr:.3f}  MacroF1={f1_macro:.3f}  "
              f"F1(skip)={f1_skip:.3f}  F1(trade)={f1_trade:.3f}  "
              f"AUC={auc:.3f}  [Train={len(train_idx)}, Test={len(test_idx)}]")

    print(f"\n[CV] 指标汇总 (F1 优化阈值):")
    for k, v in metrics.items():
        arr = np.array(v)
        arr = arr[~np.isnan(arr)]
        if len(arr) > 0:
            print(f"  {k:<12}: {arr.mean():.4f} ± {arr.std():.4f}")
    mean_thr = np.mean(fold_thresholds)
    print(f"  best_threshold : {mean_thr:.4f} ± {np.std(fold_thresholds):.4f} (各折均值)")

    valid = ~np.isnan(oof_prob)
    return (
        y_arr[valid],
        oof_prob[valid],
        oof_pred_f1[np.where(valid)[0]],   # F1 优化预测
        metrics,
        pd.Series(oof_prob, index=y.index),
        mean_thr,
    )


# ─────────────────────────────────────────────────────────────────────
# 4. 可视化
# ─────────────────────────────────────────────────────────────────────

def plot_roc_pr(y_true, y_prob, output_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax1.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {auc:.4f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    ax1.set(xlabel="FPR", ylabel="TPR", title="ROC Curve (Meta-Model OOF)")
    ax1.legend(); ax1.grid(True, linestyle=":", alpha=0.6)

    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    ax2.plot(rec, prec, color="#2ecc71", lw=2)
    # 标出 Primary 高召回率目标点
    # 找 recall=0.85 附近的精度
    target_rec = 0.85
    idx = np.argmin(np.abs(rec - target_rec))
    ax2.scatter(rec[idx], prec[idx], s=100, color="#e74c3c", zorder=5,
                label=f"Recall={rec[idx]:.2f} → Prec={prec[idx]:.2f}")
    ax2.set(xlabel="Recall", ylabel="Precision",
            title="Precision-Recall Curve (Meta-Model OOF)")
    ax2.legend(); ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Meta-Model Performance (OOF)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "meta_roc_pr.png", dpi=150)
    plt.close(fig)
    print(f"[图表] ROC/PR 曲线 → {output_dir}/meta_roc_pr.png")


def plot_calibration(y_true, y_prob, output_dir: Path):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    brier = brier_score_loss(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k:", label="完美校准")
    ax.plot(prob_pred, prob_true, "s-", color="#3498db", label="Meta-Model")
    ax.text(0.05, 0.9, f"Brier Score: {brier:.4f}", transform=ax.transAxes,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
    ax.set(xlabel="预测概率", ylabel="实际正例比率", title="Meta-Model 概率校准曲线")
    ax.legend(); ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "meta_calibration.png", dpi=150)
    plt.close(fig)
    print(f"[图表] 概率校准 → {output_dir}/meta_calibration.png")


def plot_bet_size(y_prob, output_dir: Path):
    """Bet Size = 2p - 1 (仅在 Meta=1 时下注)"""
    bet_sizes = np.clip(2 * y_prob - 1, 0, 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(bet_sizes, bins=20, kde=True, color="#f39c12", ax=ax)
    ax.set(xlabel="Bet Size", ylabel="Frequency",
           title="Bet Size Distribution (Meta-Model → Position Sizing)")
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "meta_bet_size.png", dpi=150)
    plt.close(fig)
    print(f"[图表] Bet Size 分布 → {output_dir}/meta_bet_size.png")


def plot_confusion(y_true, y_pred, output_dir: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred 0 (Skip)", "Pred 1 (Trade)"],
                yticklabels=["True 0 (Loss)", "True 1 (Win)"])
    ax.set_title("Meta-Model OOF Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "meta_confusion.png", dpi=150)
    plt.close(fig)


def plot_cv_variance(metrics: dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    folds = np.arange(1, len(metrics["roc_auc"]) + 1)

    for key, color in [("roc_auc", "#2980b9"), ("macro_f1", "#27ae60"), ("f1_skip", "#e67e22")]:
        arr = np.array(metrics[key])
        ax.plot(folds, arr, marker="o", label=key.upper(), color=color)
        ax.axhline(np.nanmean(arr), color=color, linestyle="--", alpha=0.4)

    ax.set(xticks=folds, xlabel="Fold", ylabel="Score",
           title="Meta-Model: Purged K-Fold CV Variance")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_dir / "meta_cv_variance.png", dpi=150)
    plt.close(fig)
    print(f"[图表] CV 方差图 → {output_dir}/meta_cv_variance.png")


def plot_mda(mda_results: pd.DataFrame, output_dir: Path):
    mda_sorted = mda_results.sort_values("mean_importance", ascending=True)
    labels = [f"C{int(r.cluster_id)}: {', '.join(r.features[:2])}" for _, r in mda_sorted.iterrows()]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in mda_sorted["mean_importance"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(np.arange(len(labels)), mda_sorted["mean_importance"],
            xerr=mda_sorted["std_importance"], color=colors, alpha=0.85)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="k", linestyle="--", lw=1)
    ax.set(xlabel="Log-Loss 增加量 (越大越重要)",
           title="Clustered MDA Feature Importance (Meta-Model)")
    fig.tight_layout()
    fig.savefig(output_dir / "meta_mda_importance.png", dpi=150)
    plt.close(fig)
    print(f"[图表] Clustered MDA → {output_dir}/meta_mda_importance.png")


# ─────────────────────────────────────────────────────────────────────
# 5. 主流程
# ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 数据
    X, y, t1, sample_weight, mean_u = load_and_prepare_data()

    # 2. 模型
    # 计算类别比率用于权重设置 (ratio = n_minority / n_majority)
    class_ratio = float((y == 0).sum() / max((y == 1).sum(), 1))
    print(f"[Data] 类别比率 bin=0/bin=1 = {class_ratio:.4f} (少数类强化倒数 = {1/max(class_ratio,0.05):.1f}x)")
    model = build_model(mean_u, n_samples=len(X), class_ratio=class_ratio)

    # 3. Purged CV
    y_true, y_prob, y_pred, metrics, oof_prob_series, opt_thr = run_purged_cv(
        X, y, t1, sample_weight, model
    )

    # 4. OOF 分类报告
    print(f"\n[报告] Meta-Model OOF 分类报告 (F1 优化阈值={opt_thr:.3f}):")
    print(classification_report(y_true, y_pred,
                                target_names=["Skip (0)", "Trade (1)"],
                                zero_division=0))

    # 5. 可视化
    print("\n[可视化] 生成图表...")
    plot_roc_pr(y_true, y_prob, OUTPUT_DIR)
    plot_calibration(y_true, y_prob, OUTPUT_DIR)
    plot_bet_size(y_prob, OUTPUT_DIR)
    plot_confusion(y_true, y_pred, OUTPUT_DIR)
    plot_cv_variance(metrics, OUTPUT_DIR)

    # 6. 最终模型拟合
    print("\n[Model] 在完整数据集上训练最终 Meta-Model...")
    model.fit(X.values, y.values, sample_weight=sample_weight.values)
    model_path = OUTPUT_DIR / "meta_model.pkl"
    joblib.dump(model, model_path)
    print(f"✓ Meta-Model 已保存 → {model_path}")

    # 7. Clustered MDA 特征重要性
    print("\n[MDA] 运行 Clustered MDA 特征重要性分析...")
    clusters = cluster_features(X, method="ward")
    print(f"  发现 {len(clusters)} 个特征簇")

    mda_results = clustered_mda(
        X=X, y=y, clusters=clusters,
        t1=t1, sample_weight=sample_weight,
        n_splits=CV_N_SPLITS, embargo_pct=CV_EMBARGO_PCT,
        n_repeats=1, random_state=RANDOM_STATE,
    )
    mda_results.to_csv(OUTPUT_DIR / "meta_mda_results.csv", index=False)
    plot_mda(mda_results, OUTPUT_DIR)

    # 8. 保存 OOF 元概率（可用于 Bet Sizing）
    oof_prob_df = pd.DataFrame({
        "meta_prob": oof_prob_series,
        "meta_pred": (oof_prob_series >= 0.5).astype(int),
        "bet_size":  np.clip(2 * oof_prob_series - 1, 0, 1),
    })
    oof_prob_df.to_csv(OUTPUT_DIR / "meta_oof_signals.csv")
    print(f"✓ Meta OOF 信号已保存 → {OUTPUT_DIR}/meta_oof_signals.csv")

    print(f"\n[完成] 所有输出已写入 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
