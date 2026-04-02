"""
08_combined_strategy_evaluation.py - 评估 Primary + Meta 组合策略表现

组合策略逻辑：
1. Primary Model 生成信号 (side = +1/-1)
2. Meta Model 预测信号可靠性 (bin = 0/1)
3. 只保留 Meta Model 预测 bin=1 的信号
4. 计算组合策略表现

评估指标：
- 胜率
- 盈亏比
- Sharpe Ratio
- 总收益
- 信号过滤率
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.IF9999.config import FEATURES_DIR


def main():
    print("=" * 70)
    print("  Primary + Meta 组合策略评估")
    print("=" * 70)

    # Step 1: 加载数据
    print("\n[Step 1] 加载数据...")

    # TBM 结果
    tbm_path = os.path.join(FEATURES_DIR, 'tbm_results.parquet')
    tbm = pd.read_parquet(tbm_path)
    print(f"  TBM 结果: {len(tbm)} 个信号")

    # Meta Labels (真实 bin)
    meta_labels_path = os.path.join(FEATURES_DIR, 'meta_labels.parquet')
    meta_labels = pd.read_parquet(meta_labels_path)
    print(f"  Meta Labels: {len(meta_labels)} 个")

    # Meta 特征
    meta_features_path = os.path.join(FEATURES_DIR, 'meta_features.parquet')
    meta_features = pd.read_parquet(meta_features_path)
    print(f"  Meta 特征: {meta_features.shape}")

    # Meta Model
    models_dir = os.path.join(FEATURES_DIR.replace('features', 'models'))
    model_path = os.path.join(models_dir, 'meta_model.pkl')
    meta_model = joblib.load(model_path)
    print(f"  Meta Model: 已加载")

    # Step 2: 对齐索引
    print("\n[Step 2] 对齐数据...")
    common_idx = tbm.index.intersection(meta_features.index).intersection(meta_labels.index)
    print(f"  共同样本数: {len(common_idx)}")

    tbm_aligned = tbm.loc[common_idx]
    features_aligned = meta_features.loc[common_idx]
    labels_aligned = meta_labels.loc[common_idx]

    # Step 3: 准备特征
    # 只保留 'feat_' 开头的特征
    feature_cols = [c for c in meta_features.columns if c.startswith('feat_')]
    X = meta_features[feature_cols].fillna(0)
    print(f"  特征数: {len(feature_cols)}")

    # Step 4: Meta Model 预测
    print("\n[Step 4] Meta Model 预测...")
    meta_pred = meta_model.predict(X)
    pred_proba = meta_model.predict_proba(X)[:, 1] if hasattr(meta_model, 'predict_proba') else None

    tbm_aligned = tbm_aligned.copy()
    tbm_aligned['meta_pred'] = meta_pred
    tbm_aligned['meta_proba'] = pred_proba
    tbm_aligned['true_bin'] = labels_aligned['bin'].values

    print(f"  预测 bin=1: {(meta_pred==1).sum()} ({(meta_pred==1).mean()*100:.1f}%)")
    print(f"  预测 bin=0: {(meta_pred==0).sum()} ({(meta_pred==0).mean()*100:.1f}%)")

    # Step 5: 评估 Primary Model 原始表现
    print("\n[Step 5] 评估 Primary Model 原始表现...")
    primary_stats = compute_stats(tbm_aligned['ret'], tbm_aligned['pnl'])
    print_stats("Primary Model (原始)", primary_stats)

    # Step 6: 评估组合策略表现
    print("\n[Step 6] 评估组合策略表现...")

    # 方案 A: 只保留 meta_pred=1 的信号
    filtered_df = tbm_aligned[tbm_aligned['meta_pred'] == 1]
    combined_stats = compute_stats(filtered_df['ret'], filtered_df['pnl'])
    print_stats("组合策略 (meta_pred=1)", combined_stats)

    # 过滤率
    filter_rate = 1 - len(filtered_df) / len(tbm_aligned)
    print(f"\n  信号过滤率: {filter_rate*100:.1f}%")
    print(f"  保留信号数: {len(filtered_df)} / {len(tbm_aligned)}")

    # Step 7: 评估过滤效果
    print("\n[Step 7] 过滤效果分析...")

    # 被过滤掉的信号表现（如果保留会怎样）
    filtered_out_df = tbm_aligned[tbm_aligned['meta_pred'] == 0]
    if len(filtered_out_df) > 0:
        filtered_out_stats = compute_stats(filtered_out_df['ret'], filtered_out_df['pnl'])
        print_stats("被过滤信号 (meta_pred=0)", filtered_out_stats)

    # Meta Model 预测准确率
    accuracy = (tbm_aligned['meta_pred'] == tbm_aligned['true_bin']).mean()
    precision = ((tbm_aligned['meta_pred'] == 1) & (tbm_aligned['true_bin'] == 1)).sum() / (tbm_aligned['meta_pred'] == 1).sum()
    recall = ((tbm_aligned['meta_pred'] == 1) & (tbm_aligned['true_bin'] == 1)).sum() / (tbm_aligned['true_bin'] == 1).sum()

    print(f"\n  Meta Model 在测试集上的表现:")
    print(f"  Accuracy: {accuracy*100:.1f}%")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  Recall: {recall*100:.1f}%")

    # Step 8: 理想情况分析
    print("\n[Step 8] 理想情况分析 (Meta Model 完美预测)...")

    # 如果只保留真正正确的信号 (true_bin=1)
    ideal_df = tbm_aligned[tbm_aligned['true_bin'] == 1]
    ideal_stats = compute_stats(ideal_df['ret'], ideal_df['pnl'])
    print_stats("理想情况 (true_bin=1)", ideal_stats)

    # Step 9: 汇总对比
    print("\n" + "=" * 70)
    print("  策略表现对比汇总")
    print("=" * 70)

    comparison = pd.DataFrame({
        'Primary Model': [
            primary_stats['n_signals'],
            f"{primary_stats['win_rate']*100:.1f}%",
            f"{primary_stats['pl_ratio']:.2f}",
            f"{primary_stats['sharpe']:.2f}",
            f"{primary_stats['total_pnl']:.0f}",
        ],
        '组合策略': [
            combined_stats['n_signals'],
            f"{combined_stats['win_rate']*100:.1f}%",
            f"{combined_stats['pl_ratio']:.2f}",
            f"{combined_stats['sharpe']:.2f}",
            f"{combined_stats['total_pnl']:.0f}",
        ],
        '理想情况': [
            ideal_stats['n_signals'],
            f"{ideal_stats['win_rate']*100:.1f}%",
            f"{ideal_stats['pl_ratio']:.2f}",
            f"{ideal_stats['sharpe']:.2f}",
            f"{ideal_stats['total_pnl']:.0f}",
        ],
    }, index=['信号数', '胜率', '盈亏比', 'Sharpe', '总收益(点)'])

    print(comparison.to_string())

    # 提升幅度
    print(f"\n  组合策略 vs Primary Model:")
    print(f"  胜率提升: +{(combined_stats['win_rate'] - primary_stats['win_rate'])*100:.1f}%")
    print(f"  Sharpe 提升: +{combined_stats['sharpe'] - primary_stats['sharpe']:.2f}")
    print(f"  信号减少: -{filter_rate*100:.1f}%")

    # Step 10: 保存结果
    print("\n[Step 10] 保存结果...")

    # 保存带预测的 TBM 结果
    output_path = os.path.join(FEATURES_DIR, 'combined_strategy_results.parquet')
    tbm_aligned.to_parquet(output_path)
    print(f"  ✅ 已保存: {output_path}")

    print("\n" + "=" * 70)
    print("  组合策略评估完成")
    print("=" * 70)


def compute_stats(returns: pd.Series, pnl: pd.Series) -> dict:
    """计算策略统计指标。"""
    n_signals = len(returns)
    win_rate = (pnl > 0).mean()

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    sharpe = returns.mean() / returns.std() * np.sqrt(1500) if returns.std() > 0 else 0
    total_pnl = pnl.sum()

    return {
        'n_signals': n_signals,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'pl_ratio': pl_ratio,
        'sharpe': sharpe,
        'total_pnl': total_pnl,
    }


def print_stats(name: str, stats: dict):
    """打印统计指标。"""
    print(f"\n  [{name}]")
    print(f"  信号数: {stats['n_signals']}")
    print(f"  胜率: {stats['win_rate']*100:.1f}%")
    print(f"  盈亏比: {stats['pl_ratio']:.2f}")
    print(f"  Sharpe: {stats['sharpe']:.2f}")
    print(f"  总收益: {stats['total_pnl']:.0f} 点")


if __name__ == "__main__":
    main()