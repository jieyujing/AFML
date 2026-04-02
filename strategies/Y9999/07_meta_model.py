"""
07_meta_model.py - Y9999 Phase 6 Meta Model 训练

根据 AFML Meta-Labeling 方法论：
- Primary Model (Trend Scanning) 决定方向 (side)
- Meta Model 决定是否下注 (bet_size)

Meta Model 的核心目标是 **高精确率（High Precision）**：
宁可错过机会，也不要在错误信号上下注。

流程:
1. 加载 Meta Features 和 Meta Labels
2. 准备特征矩阵 X、标签 y、样本权重
3. 构建 BaggingClassifier（AFML 风格 RF）
4. 运行 Purged K-Fold CV，收集 OOF 概率
5. 评估精确率、召回率、F1
6. 保存模型和 OOF 预测

输出:
  - meta_model.pkl: 训练好的模型
  - meta_oof_signals.parquet: OOF 预测
  - 07_pr_curve.png: Precision-Recall 曲线
  - 07_confusion_matrix.png: 混淆矩阵
  - 07_threshold_sweep.png: 阈值扫描图
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
)

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.Y9999.config import FEATURES_DIR, FIGURES_DIR, META_MODEL_CONFIG
from afmlkit.validation.purged_cv import PurgedKFold

sns.set_theme(style="whitegrid", context="paper")


# ============================================================
# 配置参数
# ============================================================

# CV 参数
CV_N_SPLITS = META_MODEL_CONFIG.get('cv_n_splits', 5)
CV_EMBARGO_PCT = META_MODEL_CONFIG.get('cv_embargo_pct', 0.01)

# 模型参数
N_ESTIMATORS = META_MODEL_CONFIG.get('n_estimators', 1000)
RANDOM_STATE = 42

# Meta Model 策略：高精确率阈值
PRECISION_THRESHOLD = META_MODEL_CONFIG.get('precision_threshold', 0.50)

# 输出目录
MODELS_DIR = os.path.join(os.path.dirname(__file__), "output", "models")


# ============================================================
# 数据加载
# ============================================================

def load_data():
    """
    加载 Meta Features 和 Meta Labels。

    :returns: (X, y, t1, sample_weight, df_merged)
    """
    print("\n[Step 1] 加载数据...")

    # 加载 Meta Features
    features_path = os.path.join(FEATURES_DIR, 'meta_features.parquet')
    features_df = pd.read_parquet(features_path)
    print(f"  Meta Features: {features_df.shape[0]} 样本 × {features_df.shape[1]} 特征")

    # 加载 Meta Labels
    labels_path = os.path.join(FEATURES_DIR, 'meta_labels.parquet')
    labels_df = pd.read_parquet(labels_path)
    print(f"  Meta Labels: {labels_df.shape[0]} 样本")

    # 合并数据（按索引）
    df_merged = pd.merge(
        features_df,
        labels_df,
        left_index=True,
        right_index=True,
        how='inner'
    )
    print(f"  合并后: {df_merged.shape[0]} 样本")

    # 目标 (y): Meta Label (1=一致, 0=不一致)
    y = df_merged['bin'].astype(int)

    # 标签分布
    label_counts = y.value_counts()
    print(f"  标签分布: 1={label_counts.get(1, 0)}, 0={label_counts.get(0, 0)}")

    # 样本权重: 使用 sample_weight 列（基于 |t_value|）
    if 'sample_weight' in df_merged.columns:
        sample_weight = df_merged['sample_weight'].copy()
        print(f"  样本权重: sample_weight (from |t_value|)")
    else:
        sample_weight = pd.Series(1.0, index=df_merged.index)
        print(f"  样本权重: 均匀权重")

    # 归一化样本权重
    sample_weight = (sample_weight - sample_weight.min()) / (sample_weight.max() - sample_weight.min() + 1e-8)
    sample_weight = sample_weight.clip(lower=0.1)
    print(f"  样本权重范围: [{sample_weight.min():.3f}, {sample_weight.max():.3f}]")

    # t1: 趋势结束时间（用于 Purged CV）
    if 'trend_t1' in df_merged.columns:
        t1 = pd.to_datetime(df_merged['trend_t1'])
    elif 't1' in df_merged.columns:
        t1 = pd.to_datetime(df_merged['t1'])
    else:
        # 如果没有 t1，使用索引 + 1 天
        t1 = df_merged.index + pd.Timedelta(days=1)

    # 只保留 feat_ 开头的特征 + 关键元特征
    feature_cols = [c for c in df_merged.columns if c.startswith('feat_')]

    # 添加趋势强度特征（agreement 和 primary_vs_trend 是目标变量的等价表示，不能用）
    if 'trend_strength' in df_merged.columns:
        feature_cols.append('trend_strength')

    X = df_merged[feature_cols].copy()

    # 处理缺失值
    X = X.ffill().bfill().fillna(0)

    print(f"\n[数据统计]")
    print(f"  特征矩阵: {X.shape[0]} 样本 × {X.shape[1]} 特征")
    print(f"  标签分布: 1={(y==1).sum()} ({(y==1).mean()*100:.1f}%), 0={(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
    print(f"  平均样本权重: {sample_weight.mean():.4f}")

    return X, y, t1, sample_weight, df_merged


# ============================================================
# 模型构建
# ============================================================

def build_model(n_samples: int) -> BaggingClassifier:
    """
    构建 AFML 风格 Random Forest（Bagging + DecisionTree）。

    Meta Model 追求高 Precision：
      - max_features=sqrt: 允许特征交互
      - max_depth=5: 限制深度防止过拟合
      - class_weight='balanced': 自动平衡正负样本

    :param n_samples: 样本数量
    :returns: BaggingClassifier 模型
    """
    print("\n[Step 2] 构建模型...")

    # 采样量
    max_samples_int = max(50, int(n_samples * 0.5))
    print(f"  Bagging max_samples = {max_samples_int}")

    # Meta Model: 追求高 Precision
    base_tree = DecisionTreeClassifier(
        criterion="entropy",
        max_features="sqrt",         # 允许特征交互
        max_depth=5,                 # 限制深度防止过拟合
        class_weight="balanced",     # 自动平衡正负样本
        min_weight_fraction_leaf=0,  # 不限制叶节点大小
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

    print(f"  模型: BaggingClassifier + DecisionTree")
    print(f"  n_estimators: {N_ESTIMATORS}")
    print(f"  max_features (base): sqrt")
    print(f"  目标: 高 Precision (精确率)")

    return model


# ============================================================
# Purged K-Fold CV
# ============================================================

def run_purged_cv(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    sample_weight: pd.Series,
    model: BaggingClassifier
) -> pd.DataFrame:
    """
    运行 Purged K-Fold CV，收集 OOF 概率。

    :param X: 特征矩阵
    :param y: 标签
    :param t1: 事件结束时间
    :param sample_weight: 样本权重
    :param model: 模型
    :returns: OOF 预测 DataFrame
    """
    print(f"\n[Step 3] Purged {CV_N_SPLITS}-Fold CV (Embargo={CV_EMBARGO_PCT*100:.0f}%)...")

    cv = PurgedKFold(n_splits=CV_N_SPLITS, t1=t1, embargo_pct=CV_EMBARGO_PCT)

    # Pre-allocate OOF arrays
    oof_prob_pos = np.full(len(y), np.nan)
    fold_precisions, fold_recalls, fold_f1s, fold_accs = [], [], [], []

    X_arr = X.values
    y_arr = y.values
    w_arr = sample_weight.values

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_tr, y_tr, w_tr = X_arr[train_idx], y_arr[train_idx], w_arr[train_idx]
        X_te, y_te = X_arr[test_idx], y_arr[test_idx]

        model.fit(X_tr, y_tr, sample_weight=w_tr)
        probs = model.predict_proba(X_te)

        # bin=1 的概率
        oof_prob_pos[test_idx] = probs[:, 1]

        # 指标：高精确率阈值
        preds = np.where(probs[:, 1] >= PRECISION_THRESHOLD, 1, 0)

        prec = precision_score(y_te, preds, zero_division=0)
        rec = recall_score(y_te, preds, zero_division=0)
        f1 = f1_score(y_te, preds, zero_division=0)
        acc = accuracy_score(y_te, preds)

        fold_precisions.append(prec)
        fold_recalls.append(rec)
        fold_f1s.append(f1)
        fold_accs.append(acc)

        print(f"  Fold {fold}: Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  Acc={acc:.3f}  [Train={len(train_idx)}, Test={len(test_idx)}]")

    # OOF 汇总
    oof_df = pd.DataFrame(
        {
            "prob_pos": oof_prob_pos,
            "prob_neg": 1.0 - oof_prob_pos,
            "pred_high_precision": np.where(oof_prob_pos >= PRECISION_THRESHOLD, 1, 0),
            "pred_default": np.where(oof_prob_pos >= 0.5, 1, 0),
            "y_true": y_arr,
        },
        index=y.index,
    )
    oof_df = oof_df.dropna()

    print(f"\n[CV 结果汇总] (阈值={PRECISION_THRESHOLD}):")
    print(f"  Precision: {np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}")
    print(f"  Recall:    {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}")
    print(f"  F1-Score:  {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print(f"  Accuracy:  {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")

    return oof_df


# ============================================================
# 可视化
# ============================================================

def plot_pr_curve(oof_df: pd.DataFrame, save_path: str):
    """绘制 Precision-Recall 曲线。"""
    y_true = oof_df["y_true"]
    prob_pos = oof_df["prob_pos"].values

    prec, rec, thresh = precision_recall_curve(y_true, prob_pos)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec, prec, color="#3498db", lw=2)
    ax.axvline(x=0.50, color="#e74c3c", linestyle="--", alpha=0.7, label="Recall=0.50 参考线")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall 曲线 (Meta Model)", fontsize=13)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ PR 曲线已保存: {save_path}")


def plot_confusion_matrix(oof_df: pd.DataFrame, threshold: float, save_path: str):
    """绘制混淆矩阵。"""
    y_true = oof_df["y_true"]
    y_pred = np.where(oof_df["prob_pos"] >= threshold, 1, 0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    ax.set_title(f"混淆矩阵 (阈值={threshold})", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存: {save_path}")


def plot_threshold_sweep(oof_df: pd.DataFrame, save_path: str):
    """绘制阈值扫描图。"""
    y_true = oof_df["y_true"].values
    prob_pos = oof_df["prob_pos"].values

    thresholds = np.arange(0.30, 0.81, 0.02)
    precisions, recalls, f1s = [], [], []

    for thr in thresholds:
        preds = np.where(prob_pos >= thr, 1, 0)
        precisions.append(precision_score(y_true, preds, zero_division=0))
        recalls.append(recall_score(y_true, preds, zero_division=0))
        f1s.append(f1_score(y_true, preds, zero_division=0))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precisions, color="#2ecc71", lw=2, label="Precision")
    ax.plot(thresholds, recalls, color="#e67e22", lw=2, label="Recall")
    ax.plot(thresholds, f1s, color="#9b59b6", lw=2, label="F1-Score")
    ax.axvline(x=PRECISION_THRESHOLD, color="#e74c3c", linestyle=":", lw=2,
               label=f"当前阈值 {PRECISION_THRESHOLD}")
    ax.axvline(x=0.5, color="grey", linestyle=":", lw=1.5, label="默认阈值 0.5")

    ax.set_xlabel("决策阈值", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("阈值扫描：Precision vs Recall vs F1", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 阈值扫描图已保存: {save_path}")


# ============================================================
# 主函数
# ============================================================

def main():
    """Y9999 Meta Model 训练主流程。"""
    print("=" * 70)
    print("  Y9999 Phase 6 Meta Model Training")
    print("=" * 70)

    # 确保输出目录存在
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: 加载数据
    X, y, t1, sample_weight, df_merged = load_data()

    # Step 2: 构建模型
    model = build_model(n_samples=len(X))

    # Step 3: Purged CV → OOF 概率
    oof_df = run_purged_cv(X, y, t1, sample_weight, model)

    # Step 4: 分类报告
    print("\n[Step 4] 分类报告...")
    print(f"\n[高精确率阈值 ({PRECISION_THRESHOLD})]")
    print(classification_report(
        oof_df["y_true"], oof_df["pred_high_precision"],
        target_names=["Loss (0)", "Win (1)"]
    ))
    print(f"\n[默认阈值 (0.5)]")
    print(classification_report(
        oof_df["y_true"], oof_df["pred_default"],
        target_names=["Loss (0)", "Win (1)"]
    ))

    # Step 5: 可视化
    print("\n[Step 5] 生成可视化图表...")
    plot_pr_curve(oof_df, os.path.join(FIGURES_DIR, "07_pr_curve.png"))
    plot_confusion_matrix(oof_df, PRECISION_THRESHOLD, os.path.join(FIGURES_DIR, "07_confusion_matrix.png"))
    plot_threshold_sweep(oof_df, os.path.join(FIGURES_DIR, "07_threshold_sweep.png"))

    # Step 6: 最终拟合 + 保存
    print("\n[Step 6] 在完整数据集上拟合最终模型...")
    model.fit(X.values, y.values, sample_weight=sample_weight.values)

    model_path = os.path.join(MODELS_DIR, "meta_model.pkl")
    joblib.dump(model, model_path)
    print(f"✅ 模型已保存: {model_path}")

    # Step 7: 保存 OOF 信号
    oof_path = os.path.join(MODELS_DIR, "meta_oof_signals.parquet")
    oof_df.to_parquet(oof_path)
    print(f"✅ OOF 信号已保存: {oof_path}")

    # Step 8: 保存特征重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
    }).sort_values('importance', ascending=False)

    importance_path = os.path.join(MODELS_DIR, "meta_feature_importance.csv")
    feature_importance.to_csv(importance_path, index=False)
    print(f"✅ 特征重要性已保存: {importance_path}")

    # 完成
    print("\n" + "=" * 70)
    print("  Phase 6 Meta Model Training 完成")
    print("=" * 70)
    print(f"输出目录: {MODELS_DIR}")
    print(f"  - meta_model.pkl")
    print(f"  - meta_oof_signals.parquet")
    print(f"  - meta_feature_importance.csv")
    print(f"图表目录: {FIGURES_DIR}")
    print(f"  - 07_pr_curve.png")
    print(f"  - 07_confusion_matrix.png")
    print(f"  - 07_threshold_sweep.png")


if __name__ == "__main__":
    main()