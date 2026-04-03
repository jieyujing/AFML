"""
09_pbo_validation.py - AL9999 策略 PBO 验证（修正版）

根据 AFML 方法论，计算回测过拟合概率：
- 使用 CPCV (Combinatorial Purged Cross-Validation) 生成多条路径
- 计算 Sharpe Ratio 分布
- 估计 PBO (Probability of Backtest Overfitting)

修正版说明：
- 使用滚动回测的交易记录进行验证
- 更准确地模拟策略在不同时间段的绩效分布

流程:
1. 加载滚动回测的交易记录
2. 使用 CPCV 分割数据
3. 在各路径上计算 Sharpe
4. 计算 PBO 并生成报告

输出:
  - PBO 验证报告
  - Sharpe 分布图
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.AL9999.config import FEATURES_DIR, FIGURES_DIR
from afmlkit.validation import (
    CombinatorialPurgedKFold,
    calculate_pbo,
    estimate_optimal_trials,
    pbo_validation_report,
)


def main():
    """AL9999 策略 PBO 验证主流程。"""
    print("=" * 70)
    print("  AL9999 Strategy PBO Validation (Rolling Backtest)")
    print("=" * 70)

    # Step 1: 加载滚动回测交易记录
    print("\n[Step 1] 加载滚动回测交易记录...")

    primary_trades = pd.read_parquet(os.path.join(FEATURES_DIR, 'rolling_primary_trades.parquet'))
    combined_trades = pd.read_parquet(os.path.join(FEATURES_DIR, 'rolling_combined_trades.parquet'))

    print(f"   Primary 交易数: {len(primary_trades)}")
    print(f"   Combined 交易数: {len(combined_trades)}")

    # Step 2: 估计参数试验次数
    print("\n[Step 2] 估计参数试验次数...")

    param_grid = {
        'ma_span': [5, 10, 20, 50, 100],
        'tbm_barriers': [(1.0, 1.0), (1.5, 1.5), (2.0, 2.0), (2.5, 2.5)],
        'min_ret': [0.0, 0.001, 0.002],
        'meta_threshold': [0.4, 0.5, 0.6],
    }
    n_trials = estimate_optimal_trials(param_grid)
    print(f"   参数网格估算试验次数: {n_trials}")

    # Step 3: CPCV 分割（基于 Combined 交易时间）
    print("\n[Step 3] CPCV 分割...")

    # 使用交易的入场时间作为索引
    combined_trades = combined_trades.sort_values('entry_time')
    trade_times = combined_trades['entry_time'].values

    # 创建时间索引
    trade_index = pd.DatetimeIndex(trade_times)

    # 估算持仓时间（使用实际 exit_time）
    t1 = pd.Series(combined_trades['exit_time'].values, index=trade_index)

    cpcv = CombinatorialPurgedKFold(
        n_splits=5,
        n_test_splits=2,
        t1=t1,
        embargo_pct=0.01
    )

    n_paths = cpcv.get_n_splits()
    print(f"   CPCV 配置: 5 folds, 2 test, 共 {n_paths} 条路径")

    # Step 4: 模拟各路径回测
    print("\n[Step 4] 模拟各路径回测...")

    sharpe_paths = []
    annualization_factor = 1500

    # 创建虚拟 X 用于分割
    dummy_X = pd.DataFrame(index=trade_index)

    for fold_info in cpcv.split(dummy_X):
        train_idx, test_idx, _ = fold_info

        # 获取测试集收益
        test_returns = combined_trades.iloc[test_idx]['net_ret']

        # 计算 Sharpe
        if len(test_returns) > 5 and test_returns.std() > 0:
            sr = test_returns.mean() / test_returns.std(ddof=1)
            sr_annual = sr * np.sqrt(annualization_factor)
            sharpe_paths.append(sr_annual)

    print(f"   有效路径数: {len(sharpe_paths)}")

    if len(sharpe_paths) < 5:
        print("   ❌ 有效路径数不足")
        return

    # Step 5: 计算 PBO
    print("\n[Step 5] 计算 PBO...")

    sharpe_array = np.array(sharpe_paths)
    pbo, stats = calculate_pbo(sharpe_array)

    print(f"   Sharpe 均值: {stats['sr_mean']:.4f}")
    print(f"   Sharpe 标准差: {stats['sr_std']:.4f}")
    print(f"   Sharpe 最大: {stats['sr_max']:.4f}")
    print(f"   Sharpe 最小: {stats['sr_min']:.4f}")
    print(f"   正 Sharpe 比例: {stats['positive_rate']*100:.1f}%")
    print(f"   PBO: {pbo:.4f}")

    # Step 6: 生成验证报告
    print("\n[Step 6] 生成验证报告...")

    report = pbo_validation_report(sharpe_array, n_trials)
    print(report)

    # Step 7: 保存结果
    results = {
        'n_paths': len(sharpe_paths),
        'n_trials': n_trials,
        'pbo': pbo,
        'sr_mean': stats['sr_mean'],
        'sr_std': stats['sr_std'],
        'sr_max': stats['sr_max'],
        'sr_min': stats['sr_min'],
        'positive_rate': stats['positive_rate'],
    }

    results_df = pd.DataFrame([results])
    results_path = os.path.join(FEATURES_DIR, 'pbo_validation_results.parquet')
    results_df.to_parquet(results_path)
    print(f"\n✅ PBO 验证结果已保存: {results_path}")

    # Step 8: 绘制 Sharpe 分布图
    fig_path = os.path.join(FIGURES_DIR, '09_pbo_sharpe_distribution.png')
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(sharpe_paths, bins=min(15, len(sharpe_paths)), edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='SR = 0')
    plt.axvline(x=stats['sr_mean'], color='darkorange', linestyle='-', linewidth=2,
                label=f'Mean SR = {stats["sr_mean"]:.2f}')
    plt.xlabel('Annualized Sharpe Ratio', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'CPCV Sharpe Distribution (PBO = {pbo:.2f})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"✅ Sharpe 分布图已保存: {fig_path}")
    plt.close()

    # Step 9: 最终判断
    print("\n" + "=" * 70)
    print("  PBO 验证结论")
    print("=" * 70)

    if pbo < 0.1:
        print("\n✅ PBO < 10%: 策略过拟合风险很低")
    elif pbo < 0.3:
        print("\n⚠️ PBO < 30%: 策略存在一定过拟合风险")
    else:
        print("\n❌ PBO >= 30%: 策略过拟合风险较高")

    if stats['positive_rate'] > 0.8:
        print(f"✅ {stats['positive_rate']*100:.0f}% 的路径 Sharpe > 0")
    elif stats['positive_rate'] > 0.5:
        print(f"⚠️ 仅 {stats['positive_rate']*100:.0f}% 的路径 Sharpe > 0")
    else:
        print(f"❌ 仅 {stats['positive_rate']*100:.0f}% 的路径 Sharpe > 0")

    print("=" * 70)


if __name__ == "__main__":
    main()
