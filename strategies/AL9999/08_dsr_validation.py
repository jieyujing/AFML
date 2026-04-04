"""
08_dsr_validation.py - AL9999 策略 DSR 验证

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
from typing import Optional
import pandas as pd
import numpy as np
from scipy.stats import norm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.AL9999.config import FEATURES_DIR, FILTER_FIRST_CONFIG
from strategies.AL9999.threshold_optimizer import calculate_trade_shrinkage


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


def compute_filter_first_diagnostics(
    primary_trades: pd.DataFrame,
    combined_trades: pd.DataFrame,
    side_mode: str,
    shrinkage_min: float,
    shrinkage_max: float,
    selection_info: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Compute side contribution and shrinkage pass/fail diagnostics.
    """
    primary_n = int(len(primary_trades))
    combined_n = int(len(combined_trades))
    full_shrinkage = calculate_trade_shrinkage(
        trade_count=combined_n,
        baseline_trade_count=primary_n,
    )

    selected_threshold = np.nan
    baseline_oos_n = 0
    selected_oos_n = 0
    oos_trade_shrinkage = np.nan
    if selection_info is not None and len(selection_info) > 0:
        row = selection_info.iloc[0]
        selected_threshold = float(row.get("selected_threshold", np.nan))
        baseline_oos_n = int(row.get("baseline_oos_n", 0))
        selected_oos_n = int(row.get("selected_oos_n", 0))
        oos_trade_shrinkage = float(
            row.get(
                "trade_shrinkage",
                calculate_trade_shrinkage(
                    trade_count=selected_oos_n,
                    baseline_trade_count=baseline_oos_n,
                ),
            )
        )

    shrinkage_to_check = oos_trade_shrinkage if not np.isnan(oos_trade_shrinkage) else full_shrinkage
    shrinkage_pass = (shrinkage_to_check >= shrinkage_min) and (shrinkage_to_check <= shrinkage_max)

    short_mask = combined_trades["side"] == -1 if "side" in combined_trades.columns else pd.Series([], dtype=bool)
    long_mask = combined_trades["side"] == 1 if "side" in combined_trades.columns else pd.Series([], dtype=bool)
    short_pnl = float(combined_trades.loc[short_mask, "net_pnl"].sum()) if "net_pnl" in combined_trades.columns else 0.0
    long_pnl = float(combined_trades.loc[long_mask, "net_pnl"].sum()) if "net_pnl" in combined_trades.columns else 0.0
    total_pnl = float(combined_trades["net_pnl"].sum()) if "net_pnl" in combined_trades.columns else 0.0
    short_contrib_ratio = (short_pnl / total_pnl) if total_pnl != 0 else 0.0

    return {
        "side_mode": side_mode,
        "primary_n_trades_full": primary_n,
        "combined_n_trades_full": combined_n,
        "full_trade_shrinkage": full_shrinkage,
        "oos_trade_shrinkage": oos_trade_shrinkage,
        "baseline_oos_n": baseline_oos_n,
        "selected_oos_n": selected_oos_n,
        "selected_threshold": selected_threshold,
        "shrinkage_min": shrinkage_min,
        "shrinkage_max": shrinkage_max,
        "shrinkage_pass": bool(shrinkage_pass),
        "short_net_pnl": short_pnl,
        "long_net_pnl": long_pnl,
        "short_contribution_ratio": short_contrib_ratio,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    """AL9999 策略 DSR 验证主流程。"""
    print("=" * 70)
    print("  AL9999 组合策略 DSR 验证（修正版）")
    print("=" * 70)

    # 年化因子（基于 Dollar Bars 数量）
    annualization_factor = 1500

    # Step 1: 加载数据
    print("\n[Step 1] 加载数据（使用修正后的滚动回测结果）...")

    # 优先使用 Filter-First 主回测产物；若不存在则回退滚动回测结果
    ff_primary_path = os.path.join(FEATURES_DIR, 'filter_first_primary_trades.parquet')
    ff_combined_path = os.path.join(FEATURES_DIR, 'filter_first_combined_trades.parquet')
    if os.path.exists(ff_primary_path) and os.path.exists(ff_combined_path):
        primary_trades = pd.read_parquet(ff_primary_path)
        combined_trades = pd.read_parquet(ff_combined_path)
        print("  使用 Filter-First 交易产物进行 DSR 验证")
    else:
        primary_trades = pd.read_parquet(os.path.join(FEATURES_DIR, 'rolling_primary_trades.parquet'))
        combined_trades = pd.read_parquet(os.path.join(FEATURES_DIR, 'rolling_combined_trades.parquet'))
        print("  ⚠️ 未找到 Filter-First 交易产物，回退到 rolling_* 产物")

    # 使用净收益率
    primary_returns = primary_trades['net_ret']
    combined_returns = combined_trades['net_ret']

    print(f"  Primary Model 交易数: {len(primary_trades)}")
    print(f"  Combined Strategy 交易数: {len(combined_trades)}")

    # 样本内/外分割：与真实 holdout 起点保持一致
    models_dir = FEATURES_DIR.replace('features', 'models')
    holdout_path = os.path.join(models_dir, 'meta_holdout_signals.parquet')
    if not os.path.exists(holdout_path):
        raise FileNotFoundError(f"缺少 Holdout 文件: {holdout_path}")
    holdout_df = pd.read_parquet(holdout_path)
    oos_start = pd.to_datetime(holdout_df.index.min())
    is_mask = combined_trades['exit_time'] < oos_start
    oos_mask = combined_trades['exit_time'] >= oos_start

    combined_is_returns = combined_trades.loc[is_mask, 'net_ret']
    combined_oos_returns = combined_trades.loc[oos_mask, 'net_ret']

    print(f"  OOS 起点: {oos_start}")
    print(f"  Combined IS 交易数: {is_mask.sum()}")
    print(f"  Combined OOS 交易数: {oos_mask.sum()}")

    side_mode = FILTER_FIRST_CONFIG.get('side_mode', 'both')
    shrinkage_min = float(FILTER_FIRST_CONFIG.get('shrinkage_min', 0.0))
    shrinkage_max = float(FILTER_FIRST_CONFIG.get('shrinkage_max', 1.0))
    selection_path = os.path.join(FEATURES_DIR, 'filter_first_selection.parquet')
    selection_info = pd.read_parquet(selection_path) if os.path.exists(selection_path) else None
    ff_diag = compute_filter_first_diagnostics(
        primary_trades=primary_trades,
        combined_trades=combined_trades,
        side_mode=side_mode,
        shrinkage_min=shrinkage_min,
        shrinkage_max=shrinkage_max,
        selection_info=selection_info,
    )

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

    # Step 5: 计算 Combined Strategy DSR（样本外）
    print("\n[Step 5] Combined Strategy DSR (样本外)...")

    # 样本外 DSR（使用 IS/OOS 分割）
    combined_oos_stats = calculate_dsr_v2(combined_oos_returns, n_total_trials, annualization_factor)

    print(f"  样本数: {combined_oos_stats['n']}")
    print(f"  年化 Sharpe: {combined_oos_stats['sr_annual']:.2f}")
    print(f"  PSR: {combined_oos_stats['psr']*100:.1f}%")
    print(f"  DSR: {combined_oos_stats['dsr']*100:.1f}%")

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
    print("│                        Primary    Combined   OOS                    │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  交易次数:           {primary_stats['n']:>7}    {combined_insample_stats['n']:>7}    {combined_oos_stats['n']:>7}          │")
    print(f"│  年化 Sharpe:        {primary_stats['sr_annual']:>7.2f}    {combined_insample_stats['sr_annual']:>7.2f}    {combined_oos_stats['sr_annual']:>7.2f}          │")
    print(f"│  PSR:              {primary_stats['psr']*100:>6.1f}%   {combined_insample_stats['psr']*100:>6.1f}%   {combined_oos_stats['psr']*100:>6.1f}%          │")
    print(f"│  DSR:              {primary_stats['dsr']*100:>6.1f}%   {combined_insample_stats['dsr']*100:>6.1f}%   {combined_oos_stats['dsr']*100:>6.1f}%          │")
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

    # Combined OOS DSR 结论
    if combined_oos_stats['dsr'] > 0.95:
        oos_dsr_status = "✅ PASS"
    elif combined_oos_stats['dsr'] > 0.90:
        oos_dsr_status = "⚠️ BORDERLINE"
    else:
        oos_dsr_status = "❌ FAIL"

    print(f"│  Combined OOS DSR:    {oos_dsr_status:>12}  ({combined_oos_stats['dsr']*100:.1f}%)            │")

    # 样本外 Sharpe
    if combined_oos_stats['sr_annual'] > 1.0:
        oos_sharpe_status = "✅ POSITIVE"
    else:
        oos_sharpe_status = "⚠️ LOW"

    print(f"│  OOS Sharpe:          {oos_sharpe_status:>12}  ({combined_oos_stats['sr_annual']:.2f})             │")
    shrinkage_status = "✅ PASS" if ff_diag["shrinkage_pass"] else "❌ FAIL"
    shrinkage_value = ff_diag["oos_trade_shrinkage"] if not np.isnan(ff_diag["oos_trade_shrinkage"]) else ff_diag["full_trade_shrinkage"]
    print(f"│  收缩约束:            {shrinkage_status:>12}  ({shrinkage_value:.3f})             │")

    print("└─────────────────────────────────────────────────────────────────────┘")

    print("\n[Filter-First 诊断]")
    print(f"  Side Mode: {ff_diag['side_mode']}")
    print(f"  Full Trade Shrinkage: {ff_diag['full_trade_shrinkage']:.4f} "
          f"(target: {ff_diag['shrinkage_min']:.2f} ~ {ff_diag['shrinkage_max']:.2f})")
    if not np.isnan(ff_diag["oos_trade_shrinkage"]):
        print(f"  OOS Trade Shrinkage: {ff_diag['oos_trade_shrinkage']:.4f} "
              f"(baseline_oos_n={ff_diag['baseline_oos_n']}, selected_oos_n={ff_diag['selected_oos_n']})")
    if not np.isnan(ff_diag["selected_threshold"]):
        print(f"  Selected Threshold: {ff_diag['selected_threshold']:.4f}")
    print(f"  Shrinkage Pass: {ff_diag['shrinkage_pass']}")
    print(f"  Long Net PnL: {ff_diag['long_net_pnl']:.2f}")
    print(f"  Short Net PnL: {ff_diag['short_net_pnl']:.2f}")
    print(f"  Short Contribution Ratio: {ff_diag['short_contribution_ratio']:.4f}")

    # Step 7: 最终判断
    print("\n" + "=" * 70)
    print("  最终判断")
    print("=" * 70)

    issues = []

    if primary_stats['dsr'] < 0.95:
        issues.append(f"  ⚠️ Primary Model DSR ({primary_stats['dsr']*100:.1f}%) < 95%")

    if combined_oos_stats['dsr'] < 0.95:
        issues.append(f"  ⚠️ Combined OOS DSR ({combined_oos_stats['dsr']*100:.1f}%) < 95%")

    # 检查样本内外差距
    sharpe_gap = combined_insample_stats['sr_annual'] - combined_oos_stats['sr_annual']
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
        'side_mode': ff_diag['side_mode'],
        'primary_n_trades_full': ff_diag['primary_n_trades_full'],
        'combined_n_trades_full': ff_diag['combined_n_trades_full'],
        'full_trade_shrinkage': ff_diag['full_trade_shrinkage'],
        'oos_trade_shrinkage': ff_diag['oos_trade_shrinkage'],
        'baseline_oos_n': ff_diag['baseline_oos_n'],
        'selected_oos_n': ff_diag['selected_oos_n'],
        'selected_threshold': ff_diag['selected_threshold'],
        'shrinkage_min': ff_diag['shrinkage_min'],
        'shrinkage_max': ff_diag['shrinkage_max'],
        'shrinkage_pass': ff_diag['shrinkage_pass'],
        'long_net_pnl': ff_diag['long_net_pnl'],
        'short_net_pnl': ff_diag['short_net_pnl'],
        'short_contribution_ratio': ff_diag['short_contribution_ratio'],
        'primary_n_trades': primary_stats['n'],
        'primary_sr_annual': primary_stats['sr_annual'],
        'primary_psr': primary_stats['psr'],
        'primary_dsr': primary_stats['dsr'],
        'combined_n_trades': combined_insample_stats['n'],
        'combined_sr_annual': combined_insample_stats['sr_annual'],
        'combined_psr': combined_insample_stats['psr'],
        'combined_dsr': combined_insample_stats['dsr'],
        'oos_n_trades': combined_oos_stats['n'],
        'oos_sr_annual': combined_oos_stats['sr_annual'],
        'oos_psr': combined_oos_stats['psr'],
        'oos_dsr': combined_oos_stats['dsr'],
    }

    results_df = pd.DataFrame([results])
    results_path = os.path.join(FEATURES_DIR, 'dsr_validation_results.parquet')
    results_df.to_parquet(results_path)
    print(f"\n✅ 验证结果已保存: {results_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()
