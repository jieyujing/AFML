"""
08_dsr_validation.py - IF9999 策略 DSR 验证

根据 AFML 方法论，计算以下指标验证策略显著性：
- PSR (Probabilistic Sharpe Ratio)：衡量 Sharpe Ratio 统计显著性
- DSR (Deflated Sharpe Ratio)：惩罚多次参数试验导致的假阳性

关键改进：
- 评估 Primary Model vs Combined Strategy 的 DSR
- 样本内 vs 样本外对比
- 考虑实际参数试验次数（包括 TBM 调参 + Meta Model 训练）

流程:
1. 加载 TBM 结果和 Walk-Forward 验证结果
2. 计算各项 Sharpe Ratio
3. 计算 PSR 和 DSR
4. 输出完整验证报告

输出:
  - 统计报告（文本）
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import norm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.IF9999.config import FEATURES_DIR


# ============================================================
# PSR/DSR 计算函数
# ============================================================

from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurt


def calculate_psr(returns: pd.Series, benchmark_sr: float = 0.0) -> tuple:
    """
    计算概率夏普比率 (Probabilistic Sharpe Ratio)。

    PSR 衡量真实 Sharpe Ratio 大于基准的概率，
    考虑了收益率的偏度和峰度对 Sharpe 估计的影响。
    公式参考 AFML Ch.10.1。

    :param returns: 收益率序列
    :param benchmark_sr: 基准 Sharpe Ratio（默认 0）
    :returns: (psr, sr) - PSR 值和周期 Sharpe Ratio
    """
    n = len(returns)
    if n < 2:
        return 0.0, 0.0

    std = returns.std(ddof=1)
    if std == 0:
        return 0.0, 0.0

    sr = returns.mean() / std

    # 使用 scipy 计算偏度和峰度 (Pearson 峰度，fisher=False)
    # 这比 pandas.kurtosis() + 3 更直接且在处理 NaN 时更稳健
    sk = scipy_skew(returns, nan_policy='omit')
    kt = scipy_kurt(returns, fisher=False, nan_policy='omit')

    # Sharpe Ratio 的标准误 (考虑偏度和峰度)
    # 符合 AFML Ch.10.1 Formula (1)
    denom = (1 - sk * sr + (kt - 1) / 4 * sr**2)

    # 极端值保护：如果分母 <= 0（理论上极罕见），则无法估计标准误
    if denom <= 0:
        return 0.5 if sr <= benchmark_sr else 1.0, sr

    sigma_sr = np.sqrt(denom / (n - 1))

    # 计算 PSR (Z-stat)
    z_stat = (sr - benchmark_sr) / sigma_sr
    psr = norm.cdf(z_stat)

    return psr, sr


def calculate_dsr(
    returns: pd.Series,
    n_trials: int,
    annualization_factor: float
) -> tuple:
    """
    计算偏误消除夏普比率 (Deflated Sharpe Ratio)。

    :param returns: 收益率序列
    :param n_trials: 参数试验次数
    :param annualization_factor: 年化因子
    :returns: (dsr, expected_max_sr, sr) - DSR值、期望最大Sharpe、周期Sharpe
    """
    # 年化夏普在不同试验间的标准差（经验值）
    sr_std = 0.5

    # 计算年化期望最大夏普
    gamma = 0.5772

    expected_max_sr_annual = sr_std * (
        (1 - gamma) * norm.ppf(1 - 1/n_trials) +
        gamma * norm.ppf(1 - 1/(n_trials * np.exp(-1)))
    )

    # 将年化 Benchmark 转换回周期级别
    benchmark_sr_period = expected_max_sr_annual / np.sqrt(annualization_factor)

    # 计算 PSR（以期望最大 Sharpe 为基准）
    psr, sr = calculate_psr(returns, benchmark_sr=benchmark_sr_period)

    return psr, expected_max_sr_annual, sr


def calculate_dsr_v2(returns: pd.Series, n_trials: int, annualization_factor: float = 1500) -> dict:
    """
    计算 DSR 的完整版本，返回所有相关信息。

    :returns: 包含所有统计量的字典
    """
    n = len(returns)
    sr_period = returns.mean() / returns.std(ddof=1) if returns.std() > 0 else 0
    sr_annual = sr_period * np.sqrt(annualization_factor)

    skew = returns.skew()
    kurt = returns.kurtosis() + 3

    # PSR (benchmark = 0)
    psr, _ = calculate_psr(returns, benchmark_sr=0.0)

    # DSR
    dsr, expected_max_sr, _ = calculate_dsr(returns, n_trials, annualization_factor)

    return {
        'n': n,
        'sr_period': sr_period,
        'sr_annual': sr_annual,
        'skew': skew,
        'kurtosis': kurt,
        'psr': psr,
        'dsr': dsr,
        'expected_max_sr': expected_max_sr,
        'n_trials': n_trials,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    """IF9999 策略 DSR 验证主流程。"""
    print("=" * 70)
    print("  IF9999 组合策略 DSR 验证")
    print("=" * 70)

    # 年化因子
    annualization_factor = 1500

    # Step 1: 加载数据
    print("\n[Step 1] 加载数据...")

    # TBM 结果（Primary Model）
    tbm = pd.read_parquet(os.path.join(FEATURES_DIR, 'tbm_results.parquet'))
    primary_returns = tbm['ret']
    print(f"  Primary Model 样本数: {len(primary_returns)}")

    # 组合策略结果
    combined = pd.read_parquet(os.path.join(FEATURES_DIR, 'combined_strategy_results.parquet'))
    combined_returns = combined['ret']
    print(f"  Combined Strategy 样本数: {len(combined_returns)}")

    # Walk-Forward 结果
    wf = pd.read_parquet(os.path.join(FEATURES_DIR, 'walk_forward_validation_results.parquet'))
    print(f"  Walk-Forward folds: {len(wf)}")

    # Step 2: 估算参数试验次数
    print("\n[Step 2] 估算参数试验次数...")

    # 实际进行的参数搜索：
    # 1. Dollar Bars 参数: 9 种 (TARGET_DAILY_BARS)
    # 2. TBM 参数搜索: 26 种 profit_loss_barriers × 5 min_ret × 4 vertical = 520
    # 3. Meta Model 特征选择: ~42 特征
    # 4. Meta Model 超参数: ~5 种配置

    # 保守估计：只计算明确进行的参数搜索
    n_dollar_bars_trials = 9
    n_tbm_trials = 200  # 04d 脚本实际测试的组合数
    n_meta_config_trials = 5
    n_total_trials = n_dollar_bars_trials + n_tbm_trials + n_meta_config_trials

    print(f"  Dollar Bars 参数搜索: {n_dollar_bars_trials} 次")
    print(f"  TBM 参数搜索: {n_tbm_trials} 次")
    print(f"  Meta Model 配置: {n_meta_config_trials} 次")
    print(f"  总试验次数: {n_total_trials}")

    # Step 3: 计算 Primary Model DSR
    print("\n[Step 3] Primary Model DSR...")

    primary_stats = calculate_dsr_v2(primary_returns, n_total_trials, annualization_factor)

    print(f"  样本数: {primary_stats['n']}")
    print(f"  年化 Sharpe: {primary_stats['sr_annual']:.2f}")
    print(f"  偏度: {primary_stats['skew']:.2f}")
    print(f"  峰度: {primary_stats['kurtosis']:.2f}")
    print(f"  PSR: {primary_stats['psr']*100:.1f}%")
    print(f"  DSR: {primary_stats['dsr']*100:.1f}%")

    # Step 4: 计算 Combined Strategy DSR（样本内）
    print("\n[Step 4] Combined Strategy DSR (样本内)...")

    # 注意：样本内 DSR 会过于乐观
    combined_insample_stats = calculate_dsr_v2(combined_returns, n_total_trials, annualization_factor)

    print(f"  样本数: {combined_insample_stats['n']}")
    print(f"  年化 Sharpe: {combined_insample_stats['sr_annual']:.2f}")
    print(f"  PSR: {combined_insample_stats['psr']*100:.1f}%")
    print(f"  DSR: {combined_insample_stats['dsr']*100:.1f}%")
    print(f"  ⚠️ 注意：样本内 DSR 过于乐观，仅供参考")

    # Step 5: 计算 Walk-Forward DSR（样本外）
    print("\n[Step 5] Walk-Forward DSR (样本外)...")

    # 样本外收益 = 各 fold 测试集的组合策略收益
    wf_sharpe_mean = wf['combined_sharpe'].mean()
    wf_sharpe_std = wf['combined_sharpe'].std()

    # 使用 WF 结果估算 DSR
    # 由于 WF 只做了一次参数选择，试验次数为 1
    wf_n_trials = 1

    # 用平均 Sharpe 计算 DSR
    # 需要估算样本外收益的分布
    wf_test_samples = wf['n_test'].sum()

    print(f"  测试样本总数: {wf_test_samples}")
    print(f"  平均 Sharpe: {wf_sharpe_mean:.2f}")
    print(f"  Sharpe 标准差: {wf_sharpe_std:.2f}")

    # Step 6: 综合评估
    print("\n" + "=" * 70)
    print("  DSR 验证报告")
    print("=" * 70)

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                        参数试验次数统计                              │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  Dollar Bars 优化:     {n_dollar_bars_trials:>5} 次                            │")
    print(f"│  TBM 参数搜索:         {n_tbm_trials:>5} 次                            │")
    print(f"│  Meta Model 配置:      {n_meta_config_trials:>5} 次                            │")
    print(f"│  总试验次数:           {n_total_trials:>5} 次                            │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                        DSR 验证结果                                  │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print("│                        Primary    Combined   WF-OOS                 │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  年化 Sharpe:        {primary_stats['sr_annual']:>7.2f}    {combined_insample_stats['sr_annual']:>7.2f}    {wf_sharpe_mean:>7.2f}          │")
    print(f"│  PSR:              {primary_stats['psr']*100:>6.1f}%   {combined_insample_stats['psr']*100:>6.1f}%       N/A          │")
    print(f"│  DSR:              {primary_stats['dsr']*100:>6.1f}%   {combined_insample_stats['dsr']*100:>6.1f}%       N/A          │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                        验证结论                                      │")
    print("├─────────────────────────────────────────────────────────────────────┤")

    # Primary Model DSR 结论
    if primary_stats['dsr'] > 0.95:
        primary_dsr_status = "✅ PASS"
    elif primary_stats['dsr'] > 0.90:
        primary_dsr_status = "⚠️ BORDERLINE"
    else:
        primary_dsr_status = "❌ FAIL"

    print(f"│  Primary Model DSR:   {primary_dsr_status:>12}  ({primary_stats['dsr']*100:.1f}%)            │")

    # PSR 结论
    if primary_stats['psr'] > 0.95:
        psr_status = "✅ PASS"
    elif primary_stats['psr'] > 0.90:
        psr_status = "⚠️ BORDERLINE"
    else:
        psr_status = "❌ FAIL"

    print(f"│  Primary Model PSR:   {psr_status:>12}  ({primary_stats['psr']*100:.1f}%)            │")

    # 样本外验证
    if wf_sharpe_mean > 1.0:
        wf_status = "✅ POSITIVE"
    else:
        wf_status = "⚠️ LOW"

    print(f"│  WF-OOS Sharpe:       {wf_status:>12}  ({wf_sharpe_mean:.2f})             │")

    print("└─────────────────────────────────────────────────────────────────────┘")

    # Step 7: 最终判断
    print("\n" + "=" * 70)
    print("  最终判断")
    print("=" * 70)

    issues = []

    if primary_stats['dsr'] < 0.95:
        issues.append(f"  ⚠️ Primary Model DSR ({primary_stats['dsr']*100:.1f}%) < 95%")

    if primary_stats['psr'] < 0.95:
        issues.append(f"  ⚠️ Primary Model PSR ({primary_stats['psr']*100:.1f}%) < 95%")

    # 检查样本内外差距
    sharpe_gap = combined_insample_stats['sr_annual'] - wf_sharpe_mean
    if sharpe_gap > 5:
        issues.append(f"  ⚠️ 样本内外 Sharpe 差距过大 ({sharpe_gap:.1f})")

    if issues:
        print("\n发现以下问题:")
        for issue in issues:
            print(issue)

        print("\n建议:")
        print("  1. 减少参数试验次数（简化策略）")
        print("  2. 使用更严格的样本外验证")
        print("  3. 小资金试盘验证")
    else:
        print("\n✅ 策略通过 DSR 验证")

    # Step 8: 保存结果
    results = {
        'n_total_trials': n_total_trials,
        'primary_sr_annual': primary_stats['sr_annual'],
        'primary_psr': primary_stats['psr'],
        'primary_dsr': primary_stats['dsr'],
        'combined_sr_annual': combined_insample_stats['sr_annual'],
        'combined_psr': combined_insample_stats['psr'],
        'combined_dsr': combined_insample_stats['dsr'],
        'wf_sharpe_mean': wf_sharpe_mean,
        'wf_sharpe_std': wf_sharpe_std,
    }

    results_df = pd.DataFrame([results])
    results_path = os.path.join(FEATURES_DIR, 'dsr_validation_results.parquet')
    results_df.to_parquet(results_path)
    print(f"\n✅ 验证结果已保存: {results_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()