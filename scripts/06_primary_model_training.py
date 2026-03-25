"""
Primary Model Training Script — AFML Pipeline (High-Recall Focus)
=================================================================

按照 AFML Meta-Labeling 架构，Primary Model 的核心目标是 **高召回率（High Recall）**。
宁可多报信号（被 Meta-Model 过滤），绝不漏报真实趋势信号。

设计原则：
  - Target (y)      : Trend Scan side (+1 / -1)
  - Sample Weight   : |t_value| × avg_uniqueness（显著趋势点权重更高）
  - 高召回策略      : 降低判断阈值（默认 0.5 → 0.4）+ 宽松叶子节点限制
  - OOF 输出       : 将 OOF 概率保存供 Meta-Model 使用

Usage
-----
    uv run python scripts/primary_model_training.py
"""

from __future__ import annotations

import joblib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)

from afmlkit.validation.purged_cv import PurgedKFold

# ── Paths ─────────────────────────────────────────────────────────────
FEATURES_PATH = Path("outputs/dollar_bars/feature_matrix.csv")
LABELS_PATH   = Path("outputs/dollar_bars/cusum_sampled_bars.csv")
OUTPUT_DIR    = Path("outputs/models/primary_model")

# ── 从特征矩阵中排除的元数据列 ─────────────────────────────────────
# 禁止将 TBM 的 bin、t1 这类未来信息漏入特征空间。
META_COLS = [
    "open", "high", "low", "close", "volume", "vwap",
    "trades", "median_trade_size", "log_return",
    "bin", "t1", "avg_uniqueness", "return_attribution",
    "volatility", "daily_vol_est",
    "side", "t_value", "trend_confidence",
    "trend_weighted_uniqueness", "vertical_touch_weights",
    "event_idx", "touch_idx", "ret",
]

# ── 超参数 ─────────────────────────────────────────────────────────────
CV_N_SPLITS    = 5
CV_EMBARGO_PCT = 0.01
N_ESTIMATORS   = 1000
RANDOM_STATE   = 42

# ── 高召回策略：信号触发阈值 ──────────────────────────────────────────
# predict_proba >= RECALL_THRESHOLD → 触发该方向信号
# 低于 0.5 意味着：宁可多报，不可漏报
RECALL_THRESHOLD = 0.40


# ─────────────────────────────────────────────────────────────────────
# 1. 数据加载
# ─────────────────────────────────────────────────────────────────────

def load_and_prepare_data():
    """加载特征矩阵与 Trend Scan 标签，准备高召回率训练集。"""
    print(f"[Data] 加载特征矩阵: {FEATURES_PATH}")
    df_feat = pd.read_csv(FEATURES_PATH, parse_dates=["timestamp"])

    print(f"[Data] 加载 Trend Scan 标签: {LABELS_PATH}")
    df_lab = pd.read_csv(LABELS_PATH, parse_dates=["timestamp"])

    # 仅合并 trend_weighted_uniqueness（特征矩阵中尚缺少该列）
    df = pd.merge(
        df_feat,
        df_lab[["timestamp", "trend_weighted_uniqueness"]],
        on="timestamp",
        how="inner",
    ).set_index("timestamp").sort_index()

    if "side" not in df.columns:
        raise ValueError("'side' 列缺失，请先运行 feature_engineering.py。")

    # 过滤横盘事件（side == 0，无趋势信号）
    n_before = len(df)
    df = df[df["side"] != 0].copy()
    print(f"[Data] 过滤 side=0: {n_before} → {len(df)} 个有效趋势事件")

    # 目标 (y): Trend Scan 方向标签
    y = df["side"].astype(int)

    # 样本权重
    if "trend_weighted_uniqueness" in df.columns and df["trend_weighted_uniqueness"].notna().any():
        sample_weight = df["trend_weighted_uniqueness"].fillna(
            df.get("avg_uniqueness", pd.Series(1.0, index=df.index))
        ).clip(lower=0)
        print("[Data] 使用 trend_weighted_uniqueness (|t_value| × avg_uniqueness) 作为样本权重")
    else:
        sample_weight = df.get("avg_uniqueness", pd.Series(1.0, index=df.index)).fillna(1.0).clip(lower=0)
        print("[Data] 回退: 使用 avg_uniqueness 作为样本权重")

    t1 = pd.to_datetime(df["t1"])

    # 特征选择：排除所有元数据列
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].dropna(axis=1, how="all").ffill().bfill()

    avg_u = df["avg_uniqueness"].mean() if "avg_uniqueness" in df.columns else 0.5

    print(f"[Data] 特征矩阵: {X.shape[0]} 样本 × {X.shape[1]} 个特征")
    print(f"[Data] 标签分布: {y.value_counts().sort_index().to_dict()}")
    print(f"[Data] 平均唯一性 (avg_u): {avg_u:.4f}")

    return X, y, t1, sample_weight, avg_u, df


# ─────────────────────────────────────────────────────────────────────
# 2. 模型构建（高召回率配置）
# ─────────────────────────────────────────────────────────────────────

def build_model(mean_uniqueness: float, n_samples: int) -> BaggingClassifier:
    """
    构建唯一性约束 Bagging 分类器（AFML 标准 Random Forest 变体）。

    高召回率关键调整：
      - min_weight_fraction_leaf=0：允许树尽可能深，捕捉所有模式
      - class_weight='balanced'：平衡两个方向的学习强度
      - max_samples 取整数：AFML 按唯一性比例限制，但不能低于 50
    """
    # 唯一性约束：采样量 = n_samples × avg_uniqueness，但至少 50
    max_samples_int = max(50, int(n_samples * mean_uniqueness))
    print(f"[Model] Bagging max_samples = {max_samples_int} 个样本 (唯一性={mean_uniqueness:.3f} × N={n_samples})")

    base_tree = DecisionTreeClassifier(
        criterion="entropy",
        max_features=1,              # AFML: 强制单特征分裂，防止遮蔽效应
        class_weight="balanced",     # 自动平衡 ±1 方向
        min_weight_fraction_leaf=0,  # 高召回：不限制叶节点大小
        random_state=RANDOM_STATE,
    )

    model = BaggingClassifier(
        estimator=base_tree,
        n_estimators=N_ESTIMATORS,
        max_samples=max_samples_int,   # 整数，避免加权样本量过少的警告
        max_features=1.0,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return model


# ─────────────────────────────────────────────────────────────────────
# 3. Purged K-Fold CV（返回 OOF 概率）
# ─────────────────────────────────────────────────────────────────────

def run_purged_cv(X, y, t1, sample_weight, model):
    """
    运行 Purged K-Fold CV，收集 OOF 概率供阈值分析和 Meta-Labeling 使用。

    Returns
    -------
    oof_df : pd.DataFrame
        包含 OOF 概率 (prob_pos, prob_neg)、硬预测 (pred_default, pred_high_recall)
        以及真实标签 (y_true)，保留原始 DatetimeIndex。
    """
    print(f"\n[CV] Purged {CV_N_SPLITS}-Fold Cross-Validation (Embargo={CV_EMBARGO_PCT*100:.0f}%)...")

    cv = PurgedKFold(n_splits=CV_N_SPLITS, t1=t1, embargo_pct=CV_EMBARGO_PCT)

    classes = sorted(y.unique())  # [-1, 1]
    pos_idx = classes.index(1)    # index of +1 in predict_proba columns

    # Pre-allocate OOF arrays
    oof_prob_pos = np.full(len(y), np.nan)
    fold_recalls, fold_accs = [], []

    X_arr = X.values
    y_arr = y.values
    w_arr = sample_weight.values

    for fold, (train_idx, test_idx) in enumerate(cv.split(X), 1):
        X_tr, y_tr, w_tr = X_arr[train_idx], y_arr[train_idx], w_arr[train_idx]
        X_te, y_te = X_arr[test_idx], y_arr[test_idx]

        model.fit(X_tr, y_tr, sample_weight=w_tr)
        probs = model.predict_proba(X_te)

        oof_prob_pos[test_idx] = probs[:, pos_idx]

        # 指标：默认阈值 0.5 vs 高召回阈值
        preds_default = np.where(probs[:, pos_idx] >= 0.5, 1, -1)
        preds_recall  = np.where(probs[:, pos_idx] >= RECALL_THRESHOLD, 1, -1)

        from sklearn.metrics import recall_score, accuracy_score
        r_up   = recall_score(y_te, preds_recall, labels=[1],  average="macro", zero_division=0)
        r_down = recall_score(y_te, preds_recall, labels=[-1], average="macro", zero_division=0)
        acc    = accuracy_score(y_te, preds_default)
        fold_recalls.append((r_up, r_down))
        fold_accs.append(acc)

        print(f"  --> Fold {fold}: Acc(0.5)={acc:.3f}  Recall+1(thr={RECALL_THRESHOLD})={r_up:.3f}  Recall-1(thr={RECALL_THRESHOLD})={r_down:.3f}  [Train={len(train_idx)}, Test={len(test_idx)}]")

    # OOF 汇总
    valid = ~np.isnan(oof_prob_pos)
    oof_df = pd.DataFrame(
        {
            "prob_pos":    oof_prob_pos,
            "prob_neg":    1.0 - oof_prob_pos,
            "pred_default":     np.where(oof_prob_pos >= 0.5, 1, -1),
            "pred_high_recall": np.where(oof_prob_pos >= RECALL_THRESHOLD, 1, -1),
            "y_true":      y_arr,
        },
        index=y.index,
    )
    oof_df = oof_df[valid]

    r_arr = np.array(fold_recalls)
    print(f"\n[CV] 召回率汇总 (阈值={RECALL_THRESHOLD}):")
    print(f"     上升趋势 (+1) Recall: {r_arr[:,0].mean():.4f} ± {r_arr[:,0].std():.4f}")
    print(f"     下降趋势 (-1) Recall: {r_arr[:,1].mean():.4f} ± {r_arr[:,1].std():.4f}")
    print(f"     准确率 (阈值=0.5):    {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")

    return oof_df


# ─────────────────────────────────────────────────────────────────────
# 4. 阈值扫描：召回率 vs 精度
# ─────────────────────────────────────────────────────────────────────

def threshold_sweep(oof_df: pd.DataFrame, output_dir: Path):
    """
    扫描 [0.30, 0.70] 的概率阈值，展示 Recall vs Precision 的权衡曲线。
    帮助直观确认：降低阈值 → 信号更多 → 召回率上升 → 精度下降（由 Meta-Model 处理）。
    """
    y_true = oof_df["y_true"].values
    prob_pos = oof_df["prob_pos"].values

    # PR 曲线（+1 方向）
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(
        (y_true == 1).astype(int), prob_pos
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── 左图：PR 曲线 ──────────────────────────────────────────────
    ax = axes[0]
    ax.plot(rec_arr, prec_arr, color="#3498db", lw=2)
    ax.axvline(x=0.75, color="#e74c3c", linestyle="--", alpha=0.7, label="Recall=0.75 参考线")
    ax.set_xlabel("Recall (+1 趋势检出率)", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall 曲线 (上升趋势 +1)", fontsize=13)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    # ── 右图：阈值扫描 ─────────────────────────────────────────────
    ax = axes[1]
    thresholds = np.arange(0.30, 0.71, 0.02)
    recalls_up, recalls_dn = [], []
    precisions_up = []
    from sklearn.metrics import recall_score, precision_score
    for thr in thresholds:
        preds = np.where(prob_pos >= thr, 1, -1)
        recalls_up.append(recall_score(y_true, preds, labels=[1],  average="macro", zero_division=0))
        recalls_dn.append(recall_score(y_true, preds, labels=[-1], average="macro", zero_division=0))
        precisions_up.append(precision_score(y_true, preds, labels=[1], average="macro", zero_division=0))

    ax.plot(thresholds, recalls_up,    color="#2ecc71", lw=2, label="+1 召回率 (Recall)")
    ax.plot(thresholds, recalls_dn,    color="#e67e22", lw=2, label="-1 召回率 (Recall)")
    ax.plot(thresholds, precisions_up, color="#9b59b6", lw=2, linestyle="--", label="+1 精度 (Precision)")
    ax.axvline(x=RECALL_THRESHOLD, color="#e74c3c", linestyle=":", lw=2,
               label=f"当前阈值 {RECALL_THRESHOLD}")
    ax.axvline(x=0.5, color="grey", linestyle=":", lw=1.5, label="默认阈值 0.5")
    ax.set_xlabel("决策阈值 (Probability Threshold)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("阈值扫描：召回率 vs 精度", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylim(0, 1.05)

    fig.suptitle("Primary Model — High-Recall Analysis (OOF)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_path = output_dir / "primary_recall_analysis.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[图表] 召回率分析图 → {out_path}")


# ─────────────────────────────────────────────────────────────────────
# 5. 混淆矩阵图
# ─────────────────────────────────────────────────────────────────────

def plot_confusion(y_true, y_pred, title: str, path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred -1", "Pred +1"],
                yticklabels=["True -1", "True +1"])
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# 6. 主流程
# ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 数据
    X, y, t1, weights, mean_u, df_raw = load_and_prepare_data()

    # 2. 模型
    model = build_model(mean_u, n_samples=len(X))

    # 3. Purged CV → OOF 概率
    oof_df = run_purged_cv(X, y, t1, weights, model)

    # 4. OOF 报告
    print("\n[报告] 默认阈值 (0.5):")
    print(classification_report(
        oof_df["y_true"], oof_df["pred_default"],
        labels=[-1, 1], target_names=["↓ (-1)", "↑ (+1)"]
    ))
    print(f"[报告] 高召回阈值 ({RECALL_THRESHOLD}):")
    print(classification_report(
        oof_df["y_true"], oof_df["pred_high_recall"],
        labels=[-1, 1], target_names=["↓ (-1)", "↑ (+1)"]
    ))

    # 5. 混淆矩阵对比图
    plot_confusion(
        oof_df["y_true"], oof_df["pred_default"],
        f"OOF 混淆矩阵 (阈值=0.5)",
        OUTPUT_DIR / "oof_confusion_default.png"
    )
    plot_confusion(
        oof_df["y_true"], oof_df["pred_high_recall"],
        f"OOF 混淆矩阵 (高召回阈值={RECALL_THRESHOLD})",
        OUTPUT_DIR / "oof_confusion_high_recall.png"
    )

    # 6. 阈值扫描图
    threshold_sweep(oof_df, OUTPUT_DIR)

    # 7. 最终拟合（完整数据）+ 保存
    print("\n[Model] 在完整数据集上拟合最终模型...")
    model.fit(X.values, y.values, sample_weight=weights.values)
    model_path = OUTPUT_DIR / "primary_rf_model.pkl"
    joblib.dump(model, model_path)
    print(f"✓ 模型已保存 → {model_path}")

    # 8. 保存 OOF 信号（供 Meta-Model 使用）
    oof_path = OUTPUT_DIR / "primary_oof_signals.csv"
    oof_df.to_csv(oof_path)
    print(f"✓ OOF 信号已保存 → {oof_path}")
    print("\n[完成] 所有输出已写入 outputs/models/primary_model/")


if __name__ == "__main__":
    main()
