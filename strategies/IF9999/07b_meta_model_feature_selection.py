"""
07b_meta_model_feature_selection.py - Meta Model 特征筛选对比实验

根据 MDA 分析结果，对比三种特征筛选方案：
1. Full: 所有 39 个特征（原始）
2. Conservative: C3 + C2 (MDA 正 + MDA ≈ 0)
3. Aggressive: 仅 C3 (MDA 正)

AFML 原则：删除 MDA 负贡献特征，保留 OOS 有效特征。
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
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.IF9999.config import FEATURES_DIR, FIGURES_DIR, META_MODEL_CONFIG
from afmlkit.validation.purged_cv import PurgedKFold

sns.set_theme(style="whitegrid", context="paper")


# ============================================================
# 特征筛选方案定义
# ============================================================

# MDA 结果（从 meta_mda_importance.parquet 提取）
FEATURE_CLUSTERS = {
    1: ['feat_rsi_7', 'feat_rsi_14', 'feat_rsi_21', 'feat_roc_5', 'feat_roc_10',
        'feat_roc_20', 'feat_stoch_k_14', 'feat_vwap_dist_20', 'feat_vwap_dist_60',
        'feat_cs_spread_60', 'feat_cross_ma_ratio_5_20', 'feat_cross_ma_sig_5_20',
        'feat_cross_ma_ratio_10_40'],  # MDA: -0.029 ± 0.243 (REMOVE)
    2: ['feat_ewm_vol_10', 'feat_ewm_vol_20', 'feat_ewm_vol_40', 'feat_hl_vol_20',
        'feat_us_sess', 'feat_amihud_20', 'feat_cs_spread_20', 'feat_amihud_60'],  # MDA: -0.008 ± 0.023 (可选)
    3: ['feat_cross_ma_sig_10_40', 'feat_shannon_50', 'feat_lz_entropy_50',
        'feat_shannon_100'],  # MDA: +0.096 ± 0.185 (KEEP)
    4: ['feat_adx_14', 'feat_sin_time', 'feat_cos_time', 'feat_sin_dow', 'feat_cos_dow',
        'feat_asia_sess', 'feat_eu_sess', 'feat_lz_entropy_100', 'feat_serial_corr_lag1_20',
        'feat_serial_corr_lag5_20', 'feat_serial_corr_lag10_20', 'feat_ljung_box_20',
        'feat_adf_test_100'],  # MDA: -0.049 ± 0.506 (REMOVE)
}

SELECTION_SCHEMES = {
    'full': [],  # 空列表表示使用所有特征
    'conservative': FEATURE_CLUSTERS[2] + FEATURE_CLUSTERS[3],  # C2 + C3 (12 features)
    'aggressive': FEATURE_CLUSTERS[3],  # 仅 C3 (4 features)
}


# ============================================================
# 核心函数
# ============================================================

def load_data():
    """加载 Meta Features 和 Meta Labels。"""
    features_path = os.path.join(FEATURES_DIR, 'meta_features.parquet')
    features = pd.read_parquet(features_path)

    labels_path = os.path.join(FEATURES_DIR, 'meta_labels.parquet')
    labels = pd.read_parquet(labels_path)

    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx, 'bin']
    sample_weight = labels.loc[common_idx, 'sample_weight']

    # 只保留 feat_ 开头的特征
    feature_cols = [c for c in X.columns if c.startswith('feat_')]
    X = X[feature_cols].fillna(0)

    return X, y, sample_weight


def filter_features(X, scheme_name):
    """根据筛选方案过滤特征。"""
    keep_features = SELECTION_SCHEMES[scheme_name]

    if len(keep_features) == 0:
        # Full: 使用所有特征
        return X

    # 确保特征存在
    valid_features = [f for f in keep_features if f in X.columns]
    return X[valid_features]


def train_and_evaluate(X, y, sample_weight, scheme_name):
    """训练并评估模型，返回关键指标。"""
    # 构建模型
    base_tree = DecisionTreeClassifier(
        criterion="entropy",
        max_features=1,
        max_depth=5,
        class_weight="balanced",
        random_state=42,
    )

    model = BaggingClassifier(
        estimator=base_tree,
        n_estimators=META_MODEL_CONFIG.get('n_estimators', 1000),
        max_samples=0.5,
        max_features=1.0,
        n_jobs=-1,
        random_state=42,
    )

    # Purged CV
    t1 = pd.Series(X.index + pd.Timedelta(hours=2), index=X.index)
    cv = PurgedKFold(
        n_splits=META_MODEL_CONFIG.get('cv_n_splits', 5),
        t1=t1,
        embargo_pct=META_MODEL_CONFIG.get('cv_embargo_pct', 0.05)
    )

    oof_probs = np.full(len(y), np.nan)
    fold_f1_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        w_tr = sample_weight.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_te = y.iloc[test_idx]

        model.fit(X_tr, y_tr, sample_weight=w_tr)
        oof_probs[test_idx] = model.predict_proba(X_te)[:, 1]

        y_pred_fold = (oof_probs[test_idx] >= 0.5).astype(int)
        fold_f1 = f1_score(y_te, y_pred_fold)
        fold_f1_scores.append(fold_f1)

    # 整体评估
    y_pred = (oof_probs >= 0.5).astype(int)

    metrics = {
        'scheme': scheme_name,
        'n_features': X.shape[1],
        'f1_mean': np.mean(fold_f1_scores),
        'f1_std': np.std(fold_f1_scores),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1_overall': f1_score(y, y_pred),
        'oof_probs': oof_probs,
        'y_pred': y_pred,
    }

    return metrics, model


def plot_comparison(results, output_dir):
    """绘制三种方案的对比图。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    schemes = ['full', 'conservative', 'aggressive']
    colors = {'full': '#3498db', 'conservative': '#2ecc71', 'aggressive': '#e74c3c'}

    # 1. F1 对比
    ax1 = axes[0]
    x_pos = np.arange(len(schemes))
    f1_means = [results[s]['f1_mean'] for s in schemes]
    f1_stds = [results[s]['f1_std'] for s in schemes]
    ax1.bar(x_pos, f1_means, yerr=f1_stds, color=[colors[s] for s in schemes],
            capsize=5, edgecolor='white', linewidth=1.5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([s.capitalize() for s in schemes])
    ax1.set_ylabel('F1 Score')
    ax1.set_title('Cross-Validation F1 Comparison')
    ax1.set_ylim(0, 1)

    # 2. Precision vs Recall
    ax2 = axes[1]
    for s in schemes:
        ax2.scatter(results[s]['recall'], results[s]['precision'],
                   s=150, c=colors[s], label=s.capitalize(), edgecolors='white', linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision vs Recall')
    ax2.legend()
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 3. 特征数量 vs F1
    ax3 = axes[2]
    n_features = [results[s]['n_features'] for s in schemes]
    ax3.bar(x_pos, n_features, color=[colors[s] for s in schemes],
            edgecolor='white', linewidth=1.5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([s.capitalize() for s in schemes])
    ax3.set_ylabel('Number of Features')
    ax3.set_title('Feature Count')
    for i, (s, n) in enumerate(zip(schemes, n_features)):
        ax3.text(i, n + 1, str(n), ha='center', fontsize=12, fontweight='bold')

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, '07b_feature_selection_comparison.png'), dpi=150)
    plt.close()


def plot_pr_curves(results, y_true, output_dir):
    """绘制三种方案的 PR 曲线对比。"""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = {'full': '#3498db', 'conservative': '#2ecc71', 'aggressive': '#e74c3c'}

    for scheme in ['full', 'conservative', 'aggressive']:
        prec, rec, _ = precision_recall_curve(y_true, results[scheme]['oof_probs'])
        ax.plot(rec, prec, color=colors[scheme], lw=2,
               label=f"{scheme.capitalize()} (F1={results[scheme]['f1_overall']:.3f})")

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve Comparison')
    ax.legend(loc='lower left')
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.savefig(os.path.join(output_dir, '07b_pr_curve_comparison.png'), dpi=150)
    plt.close()


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("  Meta Model Feature Selection Comparison Experiment")
    print("=" * 70)

    output_dir = FIGURES_DIR
    os.makedirs(output_dir, exist_ok=True)

    # 加载原始数据
    print("\n[Step 1] 加载数据...")
    X_full, y, sample_weight = load_data()
    print(f"  原始特征数: {X_full.shape[1]}")
    print(f"  样本数: {len(y)}")
    print(f"  标签分布: 正(1)={y.sum()}, 负(0)={len(y)-y.sum()}")

    # 三种方案训练对比
    print("\n[Step 2] 训练三种方案...")
    results = {}

    for scheme in ['full', 'conservative', 'aggressive']:
        print(f"\n  --- {scheme.upper()} ---")

        X_filtered = filter_features(X_full, scheme)
        print(f"    特征数: {X_filtered.shape[1]}")
        print(f"    特征列表: {X_filtered.columns.tolist()[:5]}...")

        metrics, model = train_and_evaluate(X_filtered, y, sample_weight, scheme)
        results[scheme] = metrics

        print(f"    CV F1: {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    Overall F1: {metrics['f1_overall']:.4f}")

    # 保存最佳模型
    print("\n[Step 3] 保存最佳模型...")
    best_scheme = max(results, key=lambda s: results[s]['f1_overall'])
    print(f"  最佳方案: {best_scheme} (F1={results[best_scheme]['f1_overall']:.4f})")

    X_best = filter_features(X_full, best_scheme)
    _, best_model = train_and_evaluate(X_best, y, sample_weight, best_scheme)

    models_dir = os.path.join(os.path.dirname(__file__), "output", "models")
    model_path = os.path.join(models_dir, f'meta_model_{best_scheme}.pkl')
    joblib.dump(best_model, model_path)
    print(f"  ✅ 最佳模型已保存: {model_path}")

    # 可视化对比
    print("\n[Step 4] 可视化对比...")
    plot_comparison(results, output_dir)
    print(f"  ✅ 对比图已保存: {output_dir}/07b_feature_selection_comparison.png")

    plot_pr_curves(results, y, output_dir)
    print(f"  ✅ PR 曲线对比已保存: {output_dir}/07b_pr_curve_comparison.png")

    # 汇总报告
    print("\n" + "=" * 70)
    print("  Feature Selection Comparison Summary")
    print("=" * 70)

    print("\n  | Scheme       | Features | CV F1      | Precision | Recall | Overall F1 |")
    print("  |--------------|----------|------------|-----------|--------|------------|")
    for scheme in ['full', 'conservative', 'aggressive']:
        r = results[scheme]
        print(f"  | {scheme.capitalize():12} | {r['n_features']:8} | {r['f1_mean']:.4f}±{r['f1_std']:.2f} | {r['precision']:.4f}    | {r['recall']:.4f} | {r['f1_overall']:.4f}     |")

    # AFML 结论
    print("\n" + "=" * 70)
    print("  AFML Analysis Conclusion")
    print("=" * 70)

    if results['aggressive']['f1_overall'] > results['full']['f1_overall']:
        print("  ✅ 激进筛选优于全特征 → MDA 负贡献特征确实是噪音")
    elif results['conservative']['f1_overall'] > results['full']['f1_overall']:
        print("  ✅ 保守筛选优于全特征 → 部分噪音特征可删除")
    else:
        print("  ⚠️ 筛选效果不明显 → 需进一步分析特征交互")

    print(f"\n  最佳方案: {best_scheme.upper()} ({results[best_scheme]['n_features']} features)")
    print("=" * 70)


if __name__ == "__main__":
    main()