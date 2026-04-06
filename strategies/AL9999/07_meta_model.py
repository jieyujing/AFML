"""
07_meta_model.py - AL9999 Phase 6 Meta Model 训练 (修正版)

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

from strategies.AL9999.config import FEATURES_DIR, FIGURES_DIR, META_MODEL_CONFIG, TSFRESH_CONFIG
from afmlkit.validation.purged_cv import PurgedKFold
from afmlkit.importance.clustering import cluster_features
from afmlkit.importance.mda import clustered_mda

sns.set_theme(style="whitegrid", context="paper")


def merge_rf_probability(features: pd.DataFrame, rf_signals: pd.DataFrame) -> pd.DataFrame:
    """
    Merge RF primary probability into meta feature set.

    :param features: Meta feature DataFrame.
    :param rf_signals: RF signal DataFrame containing y_prob.
    :returns: DataFrame with rf_prob column.
    """
    merged = features.copy()
    merged['rf_prob'] = rf_signals.reindex(merged.index)['y_prob'].astype(float)
    merged['rf_prob'] = merged['rf_prob'].fillna(0.5)
    return merged


def split_train_holdout(X, y, sample_weight, holdout_months=6):
    """
    划分训练集和 Holdout OOS 集。

    保留最后 holdout_months 个月的数据不参与训练，
    用于真正的 Out-of-Sample 验证。

    :param X: 特征矩阵
    :param y: 标签
    :param sample_weight: 样本权重
    :param holdout_months: 保留的月份数
    :returns: (X_train, y_train, w_train), (X_holdout, y_holdout, w_holdout), holdout_start
    """
    holdout_start = X.index.max() - pd.DateOffset(months=holdout_months)

    train_mask = X.index < holdout_start
    holdout_mask = X.index >= holdout_start

    X_train = X[train_mask]
    y_train = y[train_mask]
    w_train = sample_weight[train_mask]

    X_holdout = X[holdout_mask]
    y_holdout = y[holdout_mask]
    w_holdout = sample_weight[holdout_mask]

    print(f"\n  训练集: {X_train.index.min().date()} ~ {X_train.index.max().date()} ({len(X_train)} 样本)")
    print(f"  Holdout: {X_holdout.index.min().date()} ~ {X_holdout.index.max().date()} ({len(X_holdout)} 样本)")

    return (X_train, y_train, w_train), (X_holdout, y_holdout, w_holdout), holdout_start


def load_data(include_tsfresh: bool = True):
    """加载 Meta Features、Meta Labels 与真实事件结束时间 t1。"""
    print("\n[Step 1] 加载数据...")

    # 加载事件特征（L1-L4 特征）
    features_path = os.path.join(FEATURES_DIR, 'events_features.parquet')
    features = pd.read_parquet(features_path)
    print(f"  事件特征: {features.shape}")

    # 可选：合并 tsfresh 特征（Phase 2b）
    if include_tsfresh and TSFRESH_CONFIG.get('enabled', False):
        tsfresh_path = os.path.join(FEATURES_DIR, 'tsfresh_features.parquet')
        if os.path.exists(tsfresh_path):
            tsfresh_df = pd.read_parquet(tsfresh_path)
            # 移除 event_idx 和 timestamp 列，只保留特征
            tsfresh_df = tsfresh_df.drop(columns=['event_idx'], errors='ignore')
            tsfresh_df = tsfresh_df.drop(columns=['timestamp'], errors='ignore')
            # 按索引对齐合并
            common_idx = features.index.intersection(tsfresh_df.index)
            features = features.loc[common_idx].join(
                tsfresh_df.loc[common_idx], rsuffix='_tsfresh'
            )
            print(f"  + tsfresh 特征: {tsfresh_df.shape[1]} 列 → 合并后: {features.shape}")
        else:
            print(f"  ⚠️ tsfresh 特征未找到: {tsfresh_path}")

    # 加载修正后的 Meta Labels
    labels_path = os.path.join(FEATURES_DIR, 'meta_labels.parquet')
    labels = pd.read_parquet(labels_path)

    # 加载 TBM 结果，提取真实事件结束时间（用于 PurgedKFold）
    tbm_path = os.path.join(FEATURES_DIR, 'tbm_results.parquet')
    tbm = pd.read_parquet(tbm_path)

    # 对齐索引
    common_idx = features.index.intersection(labels.index).intersection(tbm.index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx, 'bin']
    t1 = pd.to_datetime(tbm.loc[common_idx, 'exit_ts'], errors='coerce')

    # 样本权重 (基于盈利幅度)
    sample_weight = labels.loc[common_idx, 'sample_weight']

    print(f"  样本数: {len(X)}")
    print(f"  标签分布: 正确(1)={y.sum()}, 错误(0)={len(y)-y.sum()}")
    print(f"  平均盈利预测率: {y.mean()*100:.1f}%")

    # 只保留 'feat_' 开头的特征
    feature_cols = [c for c in X.columns if c.startswith('feat_')]
    X = X[feature_cols].fillna(0)

    # 处理 sample_weight 中的 NaN
    sample_weight = sample_weight.fillna(sample_weight.mean())

    # 注入 RF Primary 置信度特征
    rf_path = os.path.join(FEATURES_DIR, 'rf_primary_signals.parquet')
    if os.path.exists(rf_path):
        rf_signals = pd.read_parquet(rf_path)
        X = merge_rf_probability(X, rf_signals)
        print(f"  RF 概率特征: 已注入 rf_prob (from {rf_path})")
    else:
        X['rf_prob'] = 0.5
        print("  RF 概率特征: 未找到 rf_primary_signals.parquet，使用常数 0.5")

    # 若存在缺失结束时间，用“事件后 1 天”保守兜底，避免 PurgedKFold 漏清洗
    missing_t1 = int(t1.isna().sum())
    if missing_t1 > 0:
        print(f"  ⚠️ t1 缺失样本: {missing_t1}，将使用事件后 1 天作为兜底。")
        fallback_t1 = pd.Series(X.index, index=X.index) + pd.Timedelta(days=1)
        t1 = t1.fillna(fallback_t1)

    return X, y, sample_weight, t1


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
    print("  AL9999 Phase 6 Meta Model Training (With Holdout OOS)")
    print("=" * 70)

    # 确保输出目录存在
    models_dir = os.path.join(os.path.dirname(__file__), "output", "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: 加载数据
    X, y, sample_weight, t1 = load_data()

    # Step 2: 划分训练集和 Holdout OOS 集
    holdout_months = META_MODEL_CONFIG.get('holdout_months', 6)
    print(f"\n[Step 2] 划分 Holdout OOS 集（保留最后 {holdout_months} 个月）...")

    (X_train, y_train, w_train), (X_holdout, y_holdout, w_holdout), holdout_start = \
        split_train_holdout(X, y, sample_weight, holdout_months)

    # Step 3: 构建模型
    model = build_model(len(X_train))

    # Step 4: Purged 5-Fold CV（仅在训练集上）
    print("\n[Step 4] Purged 5-Fold CV（训练集）...")

    # 使用真实持仓结束时间 t1（来自 TBM 退出时间），避免 purge 不足导致的泄露
    t1_train = pd.Series(t1.loc[X_train.index], index=X_train.index)

    cv = PurgedKFold(
        n_splits=META_MODEL_CONFIG.get('cv_n_splits', 5),
        t1=t1_train,
        embargo_pct=META_MODEL_CONFIG.get('cv_embargo_pct', 0.05)
    )

    oof_probs_train = np.full(len(y_train), np.nan)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        w_tr = w_train.iloc[train_idx]
        X_te = X_train.iloc[test_idx]

        model.fit(X_tr, y_tr, sample_weight=w_tr)
        oof_probs_train[test_idx] = model.predict_proba(X_te)[:, 1]

        y_pred_fold = (oof_probs_train[test_idx] >= 0.5).astype(int)
        f1 = f1_score(y_train.iloc[test_idx], y_pred_fold)
        print(f"  Fold {fold+1}: F1={f1:.4f} [Train={len(train_idx)}, Test={len(test_idx)}]")

    # OOF 结果汇总（训练集）
    oof_df = pd.DataFrame({'y_true': y_train, 'y_prob': oof_probs_train}, index=X_train.index)

    # Step 5: 性能评估（训练集 CV）
    print("\n[Step 5] CV 性能分析（训练集）...")

    y_pred_default = (oof_probs_train >= 0.5).astype(int)
    print("\n[CV OOF Report (Threshold=0.5)]")
    print(classification_report(y_train, y_pred_default, target_names=['Loss (0)', 'Gain (1)']))

    # PR 曲线
    prec, rec, thresholds = precision_recall_curve(y_train, oof_probs_train)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(rec, prec, color='blue', lw=2)
    ax.set_title('Precision-Recall Curve (Meta Model - CV)')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(FIGURES_DIR, '07_pr_curve.png'))
    plt.close()

    # 混淆矩阵
    cm = confusion_matrix(y_train, y_pred_default)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
    plt.title('Confusion Matrix (CV OOF, Threshold=0.5)')
    plt.savefig(os.path.join(FIGURES_DIR, '07_confusion_matrix.png'))
    plt.close()

    # Step 6: 最终训练（仅用训练集，不含 Holdout）
    print("\n[Step 6] 最终训练（仅用训练集）...")
    model.fit(X_train, y_train, sample_weight=w_train)

    model_path = os.path.join(models_dir, 'meta_model.pkl')
    joblib.dump(model, model_path)
    print(f"  ✅ 模型已保存: {model_path}")

    # 保存 OOF 预测
    oof_path = os.path.join(models_dir, 'meta_oof_signals.parquet')
    oof_df.to_parquet(oof_path)
    print(f"  ✅ OOF 数据已保存: {oof_path}")

    # Step 7: Holdout OOS 验证（真正的样本外测试）
    print("\n[Step 7] Holdout OOS 验证...")

    if len(X_holdout) > 0:
        holdout_probs = model.predict_proba(X_holdout)[:, 1]
        holdout_pred = (holdout_probs >= META_MODEL_CONFIG.get('precision_threshold', 0.5)).astype(int)

        holdout_df = pd.DataFrame({
            'y_true': y_holdout,
            'y_prob': holdout_probs,
            'y_pred': holdout_pred,
        }, index=X_holdout.index)

        holdout_f1 = f1_score(y_holdout, holdout_pred)
        holdout_accuracy = (holdout_pred == y_holdout).mean()

        print(f"\n  Holdout OOS 结果:")
        print(f"    样本数: {len(X_holdout)}")
        print(f"    F1 Score: {holdout_f1:.4f}")
        print(f"    Accuracy: {holdout_accuracy:.4f}")
        print(f"    预测为 1 的比例: {holdout_pred.mean()*100:.1f}%")

        print("\n[Holdout OOS Classification Report]")
        print(classification_report(y_holdout, holdout_pred, target_names=['Loss (0)', 'Gain (1)']))

        # 保存 Holdout 预测
        holdout_path = os.path.join(models_dir, 'meta_holdout_signals.parquet')
        holdout_df.to_parquet(holdout_path)
        print(f"  ✅ Holdout 预测已保存: {holdout_path}")

        # Holdout 混淆矩阵
        cm_holdout = confusion_matrix(y_holdout, holdout_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_holdout, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
        plt.title(f'Confusion Matrix (Holdout OOS, {holdout_months} months)')
        plt.savefig(os.path.join(FIGURES_DIR, '07_holdout_confusion_matrix.png'))
        plt.close()
        print(f"  ✅ Holdout 混淆矩阵已保存")
    else:
        print("  ⚠️ Holdout 集为空，跳过验证")

    # Step 8: 特征重要性 (MDI)
    print("\n[Step 8] 特征重要性分析 (MDI)...")
    mdi_importances = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
    feat_imp = pd.Series(mdi_importances, index=X_train.columns).sort_values(ascending=False)

    imp_path = os.path.join(FIGURES_DIR, '07_feature_importance_mdi.png')
    plt.figure(figsize=(10, 12))
    feat_imp.head(30).plot(kind='barh', color='darkgreen')
    plt.gca().invert_yaxis()
    plt.title('Meta Model Feature Importance (MDI, Top 30)')
    plt.tight_layout()
    plt.savefig(imp_path, dpi=150)
    plt.close()
    print(f"  ✅ MDI 特征重要性图已保存: {imp_path}")

    # Step 9: Clustered MDA 特征重要性
    print("\n[Step 9] Clustered MDA 特征重要性分析...")

    # 特征聚类（仅在训练集上）
    clusters = cluster_features(X_train, method='ward')
    print(f"  聚类数: {len(clusters)}")

    # Clustered MDA（仅在训练集上）
    df_mda = clustered_mda(
        X=X_train,
        y=y_train,
        clusters=clusters,
        t1=t1_train,
        sample_weight=w_train,
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
    print("\n[Step 10] MDI vs MDA 对比分析...")
    print("\n  MDI Top 10 (In-sample):")
    for i, (feat, score) in enumerate(feat_imp.head(10).items(), 1):
        print(f"    {i:2}. {feat}: {score:.4f}")

    print("\n  MDA Top 10 (Out-of-sample):")
    for _, row in df_mda.head(10).iterrows():
        features_str = ', '.join(row['features'][:2])
        print(f"    C{int(row['cluster_id'])}: {features_str} → {row['mean_importance']:.4f} ± {row['std_importance']:.4f}")

    print("\n" + "=" * 70)
    print("  Phase 6 Meta Model Training 完成（含 Holdout OOS 验证）")
    print("=" * 70)


if __name__ == "__main__":
    main()
