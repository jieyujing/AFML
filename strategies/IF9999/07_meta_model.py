"""
07_meta_model.py - IF9999 Phase 6 Meta Model 训练 (修正版)

根据 AFML Meta-Labeling 方法论：
- Primary Model 决定方向 (side)
- Meta Model 预测该信号是否会盈利 (bin = 1)
- Meta Model 核心目标：高精确率 (High Precision)

AFML 最佳实践：
- 使用 PurgedKFold 跨时间验证，防止数据泄露。
- Base Learner (DecisionTree) 设置 max_features=1 以消除遮蔽效应 (Masking Effect)。
- 使用 BaggingClassifier 进行去重 Bagging 训练。
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
)

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.IF9999.config import FEATURES_DIR, FIGURES_DIR, META_MODEL_CONFIG
from afmlkit.validation.purged_cv import PurgedKFold
from afmlkit.importance.clustering import cluster_features
from afmlkit.importance.mda import clustered_mda

sns.set_theme(style="whitegrid", context="paper")


def load_data():
    """加载 Meta Features 和 Meta Labels。"""
    print("\n[Step 1] 加载数据...")

    # 加载特征
    features_path = os.path.join(FEATURES_DIR, 'meta_features.parquet')
    features = pd.read_parquet(features_path)
    print(f"  特征矩阵: {features.shape}")

    # 加载修正后的 Meta Labels
    labels_path = os.path.join(FEATURES_DIR, 'meta_labels.parquet')
    labels = pd.read_parquet(labels_path)
    
    # 对齐索引
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx, 'bin']
    
    # 样本权重 (基于盈利幅度)
    sample_weight = labels.loc[common_idx, 'sample_weight']

    print(f"  样本数: {len(X)}")
    print(f"  标签分布: 正确(1)={y.sum()}, 错误(0)={len(y)-y.sum()}")
    print(f"  平均盈利预测率: {y.mean()*100:.1f}%")

    # 只保留 'feat_' 开头的特征
    feature_cols = [c for c in X.columns if c.startswith('feat_')]
    X = X[feature_cols].fillna(0)
    
    return X, y, sample_weight


def build_model(n_samples: int):
    """构建 AFML 风格 Meta Model。"""
    print("\n[Step 2] 构建模型...")
    
    # Base Learner: 决策树
    # IMPORTANT: max_features=1 是 AFML 核心规范，防止强特征遮蔽其他信号的重要性。
    base_tree = DecisionTreeClassifier(
        criterion="entropy",
        max_features=1,              # 强制设为 1，消除遮蔽效应
        max_depth=5,                 # 限制深度防止过拟合
        class_weight="balanced",     # 自动平衡正负样本
        random_state=42,
    )

    # 集成学习：Bagging
    model = BaggingClassifier(
        estimator=base_tree,
        n_estimators=META_MODEL_CONFIG.get('n_estimators', 1000),
        max_samples=0.5,             # 随机子集采样
        max_features=1.0,
        n_jobs=-1,
        random_state=42,
    )
    
    print(f"  模型: BaggingClassifier (n={META_MODEL_CONFIG.get('n_estimators', 1000)})")
    print(f"  Base Learner: DecisionTreeClassifier (max_features=1, depth=5)")
    
    return model


def main():
    print("=" * 70)
    print("  IF9999 Phase 6 Meta Model Training (Corrected)")
    print("=" * 70)

    # 确保输出目录存在
    models_dir = os.path.join(os.path.dirname(__file__), "output", "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: 加载数据
    X, y, sample_weight = load_data()

    # Step 2: 构建模型
    model = build_model(len(X))

    # Step 3: Purged 5-Fold CV
    print("\n[Step 3] Purged 5-Fold CV...")
    
    # 假设每个样本的持仓结束时间 t1（用于 Purge）。若无具体 t1，使用索引 + 估算窗口。
    # 在实际策略中，应从 TBM 结果中提取准确的 t1。
    t1 = pd.Series(X.index + pd.Timedelta(hours=2), index=X.index) 

    cv = PurgedKFold(
        n_splits=META_MODEL_CONFIG.get('cv_n_splits', 5),
        t1=t1,
        embargo_pct=META_MODEL_CONFIG.get('cv_embargo_pct', 0.05)
    )

    oof_probs = np.full(len(y), np.nan)
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        w_tr = sample_weight.iloc[train_idx]
        X_te = X.iloc[test_idx]
        
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        oof_probs[test_idx] = model.predict_proba(X_te)[:, 1]
        
        y_pred_fold = (oof_probs[test_idx] >= 0.5).astype(int)
        f1 = f1_score(y.iloc[test_idx], y_pred_fold)
        print(f"  Fold {fold+1}: F1={f1:.4f} [Train={len(train_idx)}, Test={len(test_idx)}]")

    # OOF 结果汇总
    oof_df = pd.DataFrame({'y_true': y, 'y_prob': oof_probs}, index=X.index)
    
    # Step 4: 性能评估与可视化
    print("\n[Step 4] 性能分析...")
    
    # 默认阈值 0.5 的报告
    y_pred_default = (oof_probs >= 0.5).astype(int)
    print("\n[Default Threshold 0.5 Report]")
    print(classification_report(y, y_pred_default, target_names=['Loss (0)', 'Gain (1)']))

    # 4.1 PR 曲线
    prec, rec, thresholds = precision_recall_curve(y, oof_probs)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec, prec, color='blue', lw=2)
    ax.set_title('Precision-Recall Curve (Meta Model)')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(FIGURES_DIR, '07_pr_curve.png'))
    plt.close()

    # 4.2 混淆矩阵 (默认阈值)
    cm = confusion_matrix(y, y_pred_default)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
    plt.title('Confusion Matrix (Threshold=0.5)')
    plt.savefig(os.path.join(FIGURES_DIR, '07_confusion_matrix.png'))
    plt.close()

    # Step 5: 全量训练与保存
    print("\n[Step 5] 最终训练与保存...")
    model.fit(X, y, sample_weight=sample_weight)
    
    model_path = os.path.join(models_dir, 'meta_model.pkl')
    joblib.dump(model, model_path)
    print(f"  ✅ 模型已保存: {model_path}")
    
    # 保存 OOF 预测用于后续分析
    oof_path = os.path.join(models_dir, 'meta_oof_signals.parquet')
    oof_df.to_parquet(oof_path)
    print(f"  ✅ OOF 数据已保存: {oof_path}")

    # Step 6: 特征重要性 (MDI)
    mdi_importances = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
    feat_imp = pd.Series(mdi_importances, index=X.columns).sort_values(ascending=False)

    imp_path = os.path.join(FIGURES_DIR, '07_feature_importance_mdi.png')
    plt.figure(figsize=(10, 12))
    feat_imp.head(30).plot(kind='barh', color='darkgreen')
    plt.gca().invert_yaxis()
    plt.title('Meta Model Feature Importance (MDI, Top 30)')
    plt.tight_layout()
    plt.savefig(imp_path, dpi=150)
    plt.close()
    print(f"  ✅ MDI 特征重要性图已保存: {imp_path}")

    # Step 7: Clustered MDA 特征重要性
    print("\n[Step 7] Clustered MDA 特征重要性分析...")

    # 特征聚类
    clusters = cluster_features(X, method='ward')
    print(f"  聚类数: {len(clusters)}")

    # Clustered MDA
    df_mda = clustered_mda(
        X=X,
        y=y,
        clusters=clusters,
        t1=t1,
        sample_weight=sample_weight,
        classifier=DecisionTreeClassifier(
            criterion="entropy",
            max_features=1,
            max_depth=5,
            class_weight="balanced",
            random_state=42,
        ),
        n_splits=META_MODEL_CONFIG.get('cv_n_splits', 5),
        embargo_pct=META_MODEL_CONFIG.get('cv_embargo_pct', 0.05),
        n_repeats=1,
        random_state=42,
    )

    # 保存 MDA 结果
    mda_path = os.path.join(FEATURES_DIR, 'meta_mda_importance.parquet')
    df_mda.to_parquet(mda_path)
    print(f"  ✅ MDA 结果已保存: {mda_path}")

    # MDA 可视化
    mda_imp_path = os.path.join(FIGURES_DIR, '07_feature_importance_mda.png')
    fig, ax = plt.subplots(figsize=(10, max(4, len(df_mda) * 0.6)))
    df_plot = df_mda.sort_values('mean_importance', ascending=True)
    labels = [
        f"C{int(row.cluster_id)}: {', '.join(row.features[:2])}"
        + ('…' if len(row.features) > 2 else '')
        for _, row in df_plot.iterrows()
    ]
    y_pos = np.arange(len(labels))
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in df_plot['mean_importance']]
    ax.barh(
        y_pos,
        df_plot['mean_importance'],
        xerr=df_plot['std_importance'],
        color=colors,
        edgecolor='white',
        linewidth=0.5,
        capsize=3,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Mean Importance (neg-log-loss drop)', fontsize=11)
    ax.set_title('Meta Model Feature Importance (Clustered MDA)', fontsize=14)
    ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
    fig.tight_layout()
    plt.savefig(mda_imp_path, dpi=150)
    plt.close()
    print(f"  ✅ MDA 特征重要性图已保存: {mda_imp_path}")

    # MDI vs MDA 对比
    print("\n[Step 8] MDI vs MDA 对比分析...")
    print("\n  MDI Top 10 (In-sample):")
    for i, (feat, score) in enumerate(feat_imp.head(10).items(), 1):
        print(f"    {i:2}. {feat}: {score:.4f}")

    print("\n  MDA Top 10 (Out-of-sample):")
    for _, row in df_mda.head(10).iterrows():
        features_str = ', '.join(row['features'][:2])
        print(f"    C{int(row['cluster_id'])}: {features_str} → {row['mean_importance']:.4f} ± {row['std_importance']:.4f}")

    print("\n" + "=" * 70)
    print("  Phase 6 Meta Model Training 完成")
    print("=" * 70)


if __name__ == "__main__":
    main()