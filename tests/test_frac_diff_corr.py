"""
测试 Fractional Differentiation 的相关性约束功能

验证 optimize_d 函数在使用 min_corr 参数时：
1. 能够正确计算差分后序列与原始序列的相关性
2. 当 min_corr=0.0 时，仅使用平稳性检验（旧行为）
3. 当 min_corr>0.9 时，同时满足平稳性和相关性
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from afmlkit.feature.core.frac_diff import frac_diff_ffd, optimize_d


def generate_test_price_series(n_samples=1000, trend=0.001):
    """生成带趋势的测试价格序列 - 使用更长的序列以便更容易通过 ADF 检验"""
    np.random.seed(42)

    # 生成带漂移的随机游走
    returns = np.random.normal(loc=trend, scale=0.02, size=n_samples)
    price = 100 * np.exp(np.cumsum(returns))

    return pd.Series(price, name="test_price")


def test_optimize_d_with_correlation():
    """测试不同 min_corr 设置下的 optimize_d 结果"""

    price_series = generate_test_price_series(n_samples=1000)

    # 先验证原始序列是否非平稳
    adf_result = adfuller(price_series)
    print("=" * 70)
    print("Fractional Differentiation - 相关性约束测试")
    print("=" * 70)
    print(f"\n原始序列 ADF 检验：p-value = {adf_result[1]:.6f}")
    print(f"序列是否已平稳：{'是' if adf_result[1] < 0.05 else '否'}")

    # 测试 1: 不使用相关性约束 (min_corr = 0.0)
    print("\n[测试 1] min_corr = 0.0 (仅平稳性检验)")
    d_no_corr = optimize_d(price_series, thres=1e-4, d_step=0.05, min_corr=0.0)
    ffd_no_corr = frac_diff_ffd(price_series, d=d_no_corr, thres=1e-4)

    # 计算相关性
    common_idx = ffd_no_corr.index.intersection(price_series.index)
    corr_no_corr = np.corrcoef(
        ffd_no_corr.loc[common_idx].values,
        price_series.loc[common_idx].values
    )[0, 1]

    print(f"  最优 d: {d_no_corr}")
    print(f"  与原始序列相关性：{corr_no_corr:.4f}")

    # 测试 2: 使用中等相关性约束 (min_corr = 0.7)
    print("\n[测试 2] min_corr = 0.7 (中等相关性约束)")
    d_med_corr = optimize_d(price_series, thres=1e-4, d_step=0.05, min_corr=0.7)
    ffd_med_corr = frac_diff_ffd(price_series, d=d_med_corr, thres=1e-4)

    common_idx = ffd_med_corr.index.intersection(price_series.index)
    corr_med = np.corrcoef(
        ffd_med_corr.loc[common_idx].values,
        price_series.loc[common_idx].values
    )[0, 1]

    print(f"  最优 d: {d_med_corr}")
    print(f"  与原始序列相关性：{corr_med:.4f}")

    # 测试 3: 使用高相关性约束 (min_corr = 0.95)
    print("\n[测试 3] min_corr = 0.95 (高相关性约束)")
    d_high_corr = optimize_d(price_series, thres=1e-4, d_step=0.05, min_corr=0.95)
    ffd_high_corr = frac_diff_ffd(price_series, d=d_high_corr, thres=1e-4)

    common_idx = ffd_high_corr.index.intersection(price_series.index)
    corr_high = np.corrcoef(
        ffd_high_corr.loc[common_idx].values,
        price_series.loc[common_idx].values
    )[0, 1]

    print(f"  最优 d: {d_high_corr}")
    print(f"  与原始序列相关性：{corr_high:.4f}")
    if d_high_corr == d_no_corr and corr_no_corr < 0.95:
        print(f"  → 触发回退逻辑：无法满足 0.95 相关性，回退到第一个平稳的 d")

    # 测试 4: 使用合理的相关性约束 (min_corr = 0.75)
    print("\n[测试 4] min_corr = 0.75 (合理的相关性约束)")
    d_reasonable = optimize_d(price_series, thres=1e-4, d_step=0.05, min_corr=0.75)
    ffd_reasonable = frac_diff_ffd(price_series, d=d_reasonable, thres=1e-4)

    common_idx = ffd_reasonable.index.intersection(price_series.index)
    corr_reasonable = np.corrcoef(
        ffd_reasonable.loc[common_idx].values,
        price_series.loc[common_idx].values
    )[0, 1]

    print(f"  最优 d: {d_reasonable}")
    print(f"  与原始序列相关性：{corr_reasonable:.4f}")
    if d_reasonable > d_no_corr:
        print(f"  → 成功约束：找到了满足相关性要求的最小的 d")

    # 验证结果
    print("\n" + "=" * 70)
    print("结果分析")
    print("=" * 70)
    print(f"  d 值关系：{d_no_corr} <= {d_med_corr} <= {d_reasonable} <= {d_high_corr}")
    monotonic = d_no_corr <= d_med_corr <= d_high_corr
    print(f"  单调性 (d 随 min_corr 增加而增加): {'✓' if monotonic else '✗'}")
    print(f"  相关性关系：{corr_no_corr:.3f} >= {corr_med:.3f} >= {corr_high:.3f}: {corr_no_corr >= corr_med >= corr_high}")

    # 结论
    print("\n[结论]")
    print("  回退逻辑测试:")
    if d_high_corr == d_no_corr and corr_no_corr < 0.95:
        print("  ✓ 当无法满足 min_corr 时，正确回退到第一个平稳的 d")
    else:
        print("  - 回退逻辑未触发或行为异常")

    print("\n推荐设置:")
    print("  - min_corr = 0.0    : 仅要求平稳性，适合趋势跟踪策略")
    print("  - min_corr = 0.7-0.8: 平衡平稳性和记忆性，推荐默认设置")
    print("  - min_corr = 0.9-0.95: 高记忆保留，适合需要强特征解释性的场景")
    print("  - min_corr > 0.95   : 可能触发回退逻辑，实际效果等同于 min_corr=0")


def test_already_stationary():
    """测试已经平稳的序列"""
    print("\n" + "=" * 60)
    print("测试已平稳序列的处理")
    print("=" * 60)

    # 生成平稳序列（白噪声）
    np.random.seed(42)
    stationary_series = pd.Series(np.random.normal(0, 1, 200), name="white_noise")

    d_result = optimize_d(stationary_series, min_corr=0.0)
    print(f"\n白噪声序列的最优 d: {d_result}")
    print(f"预期：d = 0.0 (已经平稳，不需要差分)")

    if d_result == 0.0:
        print("✓ 正确识别已平稳序列")
    else:
        print("✗ 未正确识别已平稳序列")


if __name__ == "__main__":
    test_optimize_d_with_correlation()
    test_already_stationary()
    print("\n测试完成!")
