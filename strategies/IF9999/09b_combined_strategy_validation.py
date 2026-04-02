"""
09b_combined_strategy_validation.py - 组合策略样本外验证

使用 Walk-Forward 验证评估组合策略的样本外表现：
1. 按时间顺序分割数据
2. 在训练集上训练 Meta Model
3. 在测试集上评估组合策略表现
4. 计算样本外胜率、Sharpe 等

这是比 CV 更严格的验证方式，模拟真实交易场景。
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.IF9999.config import FEATURES_DIR


def main():
    print("=" * 70)
    print("  组合策略 Walk-Forward 样本外验证")
    print("=" * 70)

    # Step 1: 加载数据
    print("\n[Step 1] 加载数据...")

    # TBM 结果
    tbm = pd.read_parquet(os.path.join(FEATURES_DIR, 'tbm_results.parquet'))
    print(f"  TBM 结果: {len(tbm)} 个信号")

    # Meta 特征
    meta_features = pd.read_parquet(os.path.join(FEATURES_DIR, 'meta_features.parquet'))
    print(f"  Meta 特征: {meta_features.shape}")

    # 对齐索引
    common_idx = tbm.index.intersection(meta_features.index)
    tbm = tbm.loc[common_idx].copy()
    meta_features = meta_features.loc[common_idx].copy()

    # 准备特征
    # 只保留 'feat_' 开头的特征
    feature_cols = [c for c in meta_features.columns if c.startswith('feat_')]
    X = meta_features[feature_cols].fillna(0)
    y = (tbm['ret'] > 0).astype(int)  # 正确预测 = 1

    print(f"  样本数: {len(X)}")
    print(f"  特征数: {len(feature_cols)}")
    print(f"  时间范围: {X.index.min()} ~ {X.index.max()}")

    # Step 2: Walk-Forward 分割
    print("\n[Step 2] Walk-Forward 分割...")

    # 分成 5 个时间段：前 3 个训练，后 2 个测试
    n_splits = 5
    split_points = np.array_split(np.arange(len(X)), n_splits)

    print(f"  分割数: {n_splits}")
    for i, split in enumerate(split_points):
        print(f"    Fold {i+1}: [{split[0]}:{split[-1]}] ({len(split)} 样本)")

    # Step 3: Walk-Forward 验证
    print("\n[Step 3] Walk-Forward 验证...")

    results = []

    # 使用滚动窗口：训练窗口固定，测试窗口滚动
    # Fold 1: train=[0:2], test=[2]
    # Fold 2: train=[0:3], test=[3]
    # Fold 3: train=[0:4], test=[4]

    test_folds = [2, 3, 4]  # 最后 3 个 fold 作为测试

    for test_fold in test_folds:
        # 训练集：从头到测试 fold 之前
        train_idx = np.concatenate(split_points[:test_fold])
        test_idx = split_points[test_fold]

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        tbm_train, tbm_test = tbm.iloc[train_idx], tbm.iloc[test_idx]

        # 训练 Meta Model
        base_tree = DecisionTreeClassifier(
            criterion="entropy",
            max_features=1,
            max_depth=5,
            class_weight="balanced",
        )

        model = BaggingClassifier(
            estimator=base_tree,
            n_estimators=100,
            max_samples=0.5,
            n_jobs=-1,
            random_state=42,
        )

        model.fit(X_train, y_train)

        # 预测
        meta_pred = model.predict(X_test)

        # 评估 Meta Model
        f1 = f1_score(y_test, meta_pred)
        prec = precision_score(y_test, meta_pred)
        rec = recall_score(y_test, meta_pred)

        # 评估组合策略
        # Primary Model 原始表现
        primary_win_rate = (tbm_test['pnl'] > 0).mean()
        primary_sharpe = tbm_test['ret'].mean() / tbm_test['ret'].std() * np.sqrt(1500)
        primary_total_pnl = tbm_test['pnl'].sum()

        # 组合策略表现（只保留 meta_pred=1 的信号）
        combined_mask = meta_pred == 1
        if combined_mask.sum() > 0:
            combined_win_rate = (tbm_test.iloc[combined_mask]['pnl'] > 0).mean()
            combined_sharpe = tbm_test.iloc[combined_mask]['ret'].mean() / tbm_test.iloc[combined_mask]['ret'].std() * np.sqrt(1500) if tbm_test.iloc[combined_mask]['ret'].std() > 0 else 0
            combined_total_pnl = tbm_test.iloc[combined_mask]['pnl'].sum()
            n_signals = combined_mask.sum()
        else:
            combined_win_rate = 0
            combined_sharpe = 0
            combined_total_pnl = 0
            n_signals = 0

        # 被过滤信号表现
        filtered_mask = meta_pred == 0
        if filtered_mask.sum() > 0:
            filtered_win_rate = (tbm_test.iloc[filtered_mask]['pnl'] > 0).mean()
            filtered_total_pnl = tbm_test.iloc[filtered_mask]['pnl'].sum()
        else:
            filtered_win_rate = 0
            filtered_total_pnl = 0

        result = {
            'fold': test_fold + 1,
            'test_period': f"{X_test.index.min().strftime('%Y-%m')} ~ {X_test.index.max().strftime('%Y-%m')}",
            'n_test': len(X_test),
            'n_train': len(X_train),
            # Meta Model 指标
            'meta_f1': f1,
            'meta_precision': prec,
            'meta_recall': rec,
            # Primary 指标
            'primary_win_rate': primary_win_rate,
            'primary_sharpe': primary_sharpe,
            'primary_total_pnl': primary_total_pnl,
            # Combined 指标
            'combined_win_rate': combined_win_rate,
            'combined_sharpe': combined_sharpe,
            'combined_total_pnl': combined_total_pnl,
            'n_combined_signals': n_signals,
            # 被过滤信号
            'filtered_win_rate': filtered_win_rate,
            'filtered_total_pnl': filtered_total_pnl,
            # 提升幅度
            'win_rate_lift': combined_win_rate - primary_win_rate,
            'sharpe_lift': combined_sharpe - primary_sharpe,
        }
        results.append(result)

        print(f"\n  Fold {test_fold+1} ({result['test_period']}):")
        print(f"    训练样本: {len(X_train)}, 测试样本: {len(X_test)}")
        print(f"    Meta F1: {f1:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}")
        print(f"    Primary: 胜率={primary_win_rate*100:.1f}%, Sharpe={primary_sharpe:.2f}")
        print(f"    Combined: 胜率={combined_win_rate*100:.1f}%, Sharpe={combined_sharpe:.2f}, 信号={n_signals}/{len(X_test)}")
        print(f"    提升: 胜率+{(combined_win_rate-primary_win_rate)*100:.1f}%, Sharpe+{combined_sharpe-primary_sharpe:.2f}")

    # Step 4: 汇总统计
    print("\n" + "=" * 70)
    print("  样本外验证汇总")
    print("=" * 70)

    results_df = pd.DataFrame(results)

    print("\n[Meta Model 表现]")
    print(f"  平均 F1: {results_df['meta_f1'].mean():.3f} ± {results_df['meta_f1'].std():.3f}")
    print(f"  平均 Precision: {results_df['meta_precision'].mean():.3f}")
    print(f"  平均 Recall: {results_df['meta_recall'].mean():.3f}")

    print("\n[策略表现对比]")
    print(f"  Primary Model:")
    print(f"    平均胜率: {results_df['primary_win_rate'].mean()*100:.1f}%")
    print(f"    平均 Sharpe: {results_df['primary_sharpe'].mean():.2f}")
    print(f"    总收益: {results_df['primary_total_pnl'].sum():.0f} 点")

    print(f"\n  Combined Strategy:")
    print(f"    平均胜率: {results_df['combined_win_rate'].mean()*100:.1f}%")
    print(f"    平均 Sharpe: {results_df['combined_sharpe'].mean():.2f}")
    print(f"    总收益: {results_df['combined_total_pnl'].sum():.0f} 点")

    print(f"\n  被过滤信号:")
    print(f"    平均胜率: {results_df['filtered_win_rate'].mean()*100:.1f}%")
    print(f"    总收益: {results_df['filtered_total_pnl'].sum():.0f} 点")

    print("\n[提升幅度]")
    print(f"  胜率提升: +{results_df['win_rate_lift'].mean()*100:.1f}%")
    print(f"  Sharpe 提升: +{results_df['sharpe_lift'].mean():.2f}")

    # Step 5: 验证结论
    print("\n" + "=" * 70)
    print("  验证结论")
    print("=" * 70)

    avg_f1 = results_df['meta_f1'].mean()
    avg_win_rate_lift = results_df['win_rate_lift'].mean()
    avg_sharpe_lift = results_df['sharpe_lift'].mean()

    # 检查组合策略是否在样本外也有效
    combined_positive = (results_df['combined_total_pnl'] > 0).sum()
    filtered_negative = (results_df['filtered_total_pnl'] < 0).sum()

    print(f"\n  样本外验证指标:")
    print(f"  1. Meta Model F1 = {avg_f1:.3f} {'✅' if avg_f1 >= 0.5 else '❌'} (目标 >= 0.5)")
    print(f"  2. 胜率提升 = +{avg_win_rate_lift*100:.1f}% {'✅' if avg_win_rate_lift > 0 else '❌'} (目标 > 0)")
    print(f"  3. Sharpe 提升 = +{avg_sharpe_lift:.2f} {'✅' if avg_sharpe_lift > 0 else '❌'} (目标 > 0)")
    print(f"  4. 组合策略盈利次数 = {combined_positive}/3 {'✅' if combined_positive >= 2 else '❌'} (目标 >= 2)")
    print(f"  5. 被过滤信号亏损次数 = {filtered_negative}/3 {'✅' if filtered_negative >= 2 else '❌'} (目标 >= 2)")

    all_passed = (
        avg_f1 >= 0.5 and
        avg_win_rate_lift > 0 and
        avg_sharpe_lift > 0 and
        combined_positive >= 2 and
        filtered_negative >= 2
    )

    if all_passed:
        print("\n  ✅ 组合策略通过样本外验证！")
    else:
        print("\n  ⚠️ 组合策略部分指标未通过验证，需要进一步优化。")

    # Step 6: 保存结果
    output_path = os.path.join(FEATURES_DIR, 'walk_forward_validation_results.parquet')
    results_df.to_parquet(output_path)
    print(f"\n  ✅ 验证结果已保存: {output_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()
