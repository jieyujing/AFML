"""
09_pbo_validation.py - Y9999 策略 PBO 验证

根据 AFML 方法论，计算回测过拟合概率：
- 使用 CPCV (Combinatorial Purged Cross-Validation) 生成多条路径
- 计算 Sharpe Ratio 分布
- 估计 PBO (Probability of Backtest Overfitting)

流程:
1. 加载 TBM 结果和特征矩阵
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

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.Y9999.config import FEATURES_DIR
from afmlkit.validation import (
    CombinatorialPurgedKFold,
    calculate_pbo,
    estimate_optimal_trials,
    pbo_validation_report,
)


def main():
    """Y9999 策略 PBO 验证主流程。"""
    print("=" * 70)
    print("  Y9999 Strategy PBO Validation")
    print("=" * 70)

    # Step 1: 加载特征矩阵和信号
    print("\n[Step 1] 加载数据...")

    # 尝试多个可能的特征文件
    features_path = os.path.join(FEATURES_DIR, 'meta_features.parquet')
    if not os.path.exists(features_path):
        features_path = os.path.join(FEATURES_DIR, 'bars_features_fd.parquet')
    if not os.path.exists(features_path):
        features_path = os.path.join(FEATURES_DIR, 'features.parquet')

    if not os.path.exists(features_path):
        print(f"❌ 特征文件不存在")
        print("   请先运行 02_feature_engineering.py 生成特征")
        return

    features = pd.read_parquet(features_path)
    print(f"   特征矩阵: {features.shape}")

    # 加载 Meta Labels
    meta_path = os.path.join(FEATURES_DIR, 'meta_labels.parquet')
    if os.path.exists(meta_path):
        meta_labels = pd.read_parquet(meta_path)
        meta_signals = meta_labels.get('agreement', meta_labels.iloc[:, 0])
        print(f"   Meta Labels: {len(meta_signals)}")
    else:
        print("   ⚠️ Meta Labels 文件不存在，使用随机信号演示")
        meta_signals = pd.Series(np.random.choice([-1, 0, 1], size=len(features)),
                                  index=features.index)

    # Step 2: 估计参数试验次数
    print("\n[Step 2] 估计参数试验次数...")

    # 根据 Y9999 策略的实际参数网格估算
    param_grid = {
        'ma_span': [5, 10, 20, 50, 100],
        'tbm_barriers': [(1.0, 1.0), (1.5, 1.5), (2.0, 2.0), (2.5, 2.5)],
        'min_ret': [0.0, 0.001, 0.002],
        'meta_threshold': [0.4, 0.5, 0.6],
    }
    n_trials = estimate_optimal_trials(param_grid)
    print(f"   参数网格估算试验次数: {n_trials}")

    # Step 3: CPCV 分割
    print("\n[Step 3] CPCV 分割...")

    # 加载 TBM 结果获取结束时间
    tbm_path = os.path.join(FEATURES_DIR, 'tbm_results.parquet')
    if os.path.exists(tbm_path):
        tbm = pd.read_parquet(tbm_path)
        # 使用 exit_ts 作为结束时间
        if 'exit_ts' in tbm.columns:
            t1 = tbm['exit_ts']
        elif 't1' in tbm.columns:
            t1 = tbm['t1']
        else:
            t1 = features.index + pd.Timedelta(hours=2)
    else:
        t1 = features.index + pd.Timedelta(hours=2)

    cpcv = CombinatorialPurgedKFold(
        n_splits=6,
        n_test_splits=2,
        t1=t1,
        embargo_pct=0.01
    )

    n_paths = cpcv.get_n_splits()
    print(f"   CPCV 配置: 6 folds, 2 test, 共 {n_paths} 条路径")

    # Step 4: 模拟各路径回测
    print("\n[Step 4] 模拟各路径回测...")

    if os.path.exists(tbm_path):
        tbm = pd.read_parquet(tbm_path)
        all_returns = tbm['ret']

        sharpe_paths = []
        annualization_factor = 1500

        for _, (_, test_idx, _) in enumerate(cpcv.split(features)):
            test_returns = all_returns.iloc[test_idx]
            test_signals = meta_signals.iloc[test_idx]
            strategy_returns = test_returns * test_signals.reindex(test_returns.index).fillna(0)

            if len(strategy_returns) > 20:
                sr = strategy_returns.mean() / strategy_returns.std(ddof=1)
                sr_annual = sr * np.sqrt(annualization_factor)
                sharpe_paths.append(sr_annual)

        print(f"   有效路径数: {len(sharpe_paths)}")
    else:
        print("   ⚠️ 无 TBM 结果，使用随机演示")
        sharpe_paths = np.random.normal(loc=0.5, scale=0.3, size=n_paths)

    # Step 5: 计算 PBO
    print("\n[Step 5] 计算 PBO...")

    if len(sharpe_paths) >= 10:
        pbo, stats = calculate_pbo(np.array(sharpe_paths))
        print(f"   Sharpe 均值: {stats['sr_mean']:.4f}")
        print(f"   Sharpe 标准差: {stats['sr_std']:.4f}")
        print(f"   PBO: {pbo:.4f}")
    else:
        print("   ❌ 路径数不足，无法计算 PBO")
        pbo = 0.5
        stats = {'n_paths': len(sharpe_paths), 'sr_mean': np.mean(sharpe_paths), 'sr_std': np.std(sharpe_paths)}

    # Step 6: 生成验证报告
    print("\n[Step 6] 生成验证报告...")

    report = pbo_validation_report(np.array(sharpe_paths), n_trials)
    print(report)

    # Step 7: 保存结果
    results = {
        'n_paths': stats.get('n_paths', len(sharpe_paths)),
        'n_trials': n_trials,
        'pbo': pbo,
        'sr_mean': stats.get('sr_mean', np.mean(sharpe_paths)),
        'sr_std': stats.get('sr_std', np.std(sharpe_paths)),
        'sr_max': stats.get('sr_max', np.max(sharpe_paths)),
        'sr_min': stats.get('sr_min', np.min(sharpe_paths)),
        'positive_rate': stats.get('positive_rate', np.mean(np.array(sharpe_paths) > 0)),
    }

    results_df = pd.DataFrame([results])
    results_path = os.path.join(FEATURES_DIR, 'pbo_validation_results.parquet')
    results_df.to_parquet(results_path)
    print(f"\n✅ PBO 验证结果已保存: {results_path}")

    # Step 8: 绘制 Sharpe 分布图
    fig_dir = FEATURES_DIR.replace('features', 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'pbo_sharpe_distribution.png')

    plt.figure(figsize=(10, 6))
    plt.hist(sharpe_paths, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', label='SR = 0')
    plt.axvline(x=stats.get('sr_mean', np.mean(sharpe_paths)), color='blue', linestyle='-',
                label=f'Mean SR = {stats.get("sr_mean", np.mean(sharpe_paths)):.2f}')
    plt.xlabel('Annualized Sharpe Ratio')
    plt.ylabel('Frequency')
    plt.title(f'CPCV Sharpe Distribution (PBO = {pbo:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(fig_path)
    print(f"✅ Sharpe 分布图已保存: {fig_path}")
    plt.close()


if __name__ == "__main__":
    main()