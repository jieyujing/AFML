"""
08_dsr_validation.py - Y9999 策略 DSR 验证

根据 AFML 方法论，计算以下指标验证策略显著性：
- PSR (Probabilistic Sharpe Ratio)：衡量 Sharpe Ratio 统计显著性
- DSR (Deflated Sharpe Ratio)：惩罚多次参数试验导致的假阳性

改进：
- 自动估算参数试验次数（基于实际参数网格）

流程:
1. 加载 TBM 结果（收益率序列）
2. 计算 Sharpe Ratio（年化）
3. 计算 PSR（考虑偏度、峰度）
4. 计算 DSR（惩罚参数试验）
5. 输出验证结论

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

from strategies.Y9999.config import FEATURES_DIR
from afmlkit.validation import estimate_optimal_trials


# ============================================================
# PSR/DSR 计算函数
# ============================================================

def calculate_psr(returns: pd.Series, benchmark_sr: float = 0.0) -> tuple:
    """
    计算概率夏普比率 (Probabilistic Sharpe Ratio)。

    PSR 衡量真实 Sharpe Ratio 大于基准的概率，
    考虑了收益率的偏度和峰度对 Sharpe 估计的影响。

    :param returns: 收益率序列
    :param benchmark_sr: 基准 Sharpe Ratio（默认 0）
    :returns: (psr, sr) - PSR 值和周期 Sharpe Ratio
    """
    if len(returns) < 50:
        return 0.0, 0.0

    n = len(returns)
    sr = returns.mean() / returns.std(ddof=1)

    # 计算偏度和峰度
    skew = returns.skew()
    kurt = returns.kurtosis() + 3  # Fisher 峰度 + 3 = Pearson 峰度

    # Sharpe Ratio 的标准误（考虑偏度和峰度）
    # 参考: Bailey & López de Prado (2012)
    sigma_sr = np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1))

    # 计算 PSR
    psr = norm.cdf((sr - benchmark_sr) / sigma_sr)

    return psr, sr


def calculate_dsr(
    returns: pd.Series,
    n_trials: int,
    annualization_factor: float
) -> tuple:
    """
    计算偏误消除夏普比率 (Deflated Sharpe Ratio)。

    DSR 惩罚多次参数试验导致的假阳性。当我们进行多次回测时，
    即使真实 Sharpe 为 0，也可能偶然观察到高 Sharpe。

    :param returns: 收益率序列
    :param n_trials: 参数试验次数
    :param annualization_factor: 年化因子
    :returns: (dsr, expected_max_sr, sr) - DSR值、期望最大Sharpe、周期Sharpe
    """
    # 年化夏普在不同试验间的标准差（经验值）
    sr_std = 0.5

    # 计算年化期望最大夏普（Expected Maximum Sharpe）
    # 参考: Bailey & López de Prado (2014)
    # E[max SR] = σ_SR * [(1 - γ) * Φ⁻¹(1 - 1/N) + γ * Φ⁻¹(1 - 1/(N*e))]
    # 其中 γ = 0.5772... (Euler-Mascheroni 常数)
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


def calculate_additional_stats(returns: pd.Series) -> dict:
    """
    计算额外统计指标。

    :param returns: 收益率序列
    :returns: 统计指标字典
    """
    return {
        'n_samples': len(returns),
        'mean': returns.mean(),
        'std': returns.std(),
        'skew': returns.skew(),
        'kurtosis': returns.kurtosis(),
        'min': returns.min(),
        'max': returns.max(),
        'positive_rate': (returns > 0).mean(),
    }


# ============================================================
# 主函数
# ============================================================

def main():
    """Y9999 策略 DSR 验证主流程。"""
    print("=" * 70)
    print("  Y9999 Strategy DSR Validation")
    print("=" * 70)

    # Step 1: 加载 TBM 结果
    print("\n[Step 1] 加载 TBM 结果...")
    tbm_path = os.path.join(FEATURES_DIR, 'tbm_results.parquet')

    if not os.path.exists(tbm_path):
        print(f"❌ TBM 结果文件不存在: {tbm_path}")
        print("   请先运行 04_ma_primary_model.py 生成 TBM 结果")
        return

    tbm = pd.read_parquet(tbm_path)
    returns = tbm['ret']

    print(f"   样本数: {len(returns)}")
    print(f"   时间范围: {returns.index.min()} ~ {returns.index.max()}")

    # Step 2: 计算年化因子
    print("\n[Step 2] 计算年化因子...")
    # 6 bars/day × 250 trading days ≈ 1500 bars/year
    annualization_factor = 1500
    print(f"   年化因子: {annualization_factor} bars/year")

    # Step 3: 计算统计指标
    print("\n[Step 3] 计算统计指标...")
    stats = calculate_additional_stats(returns)

    print(f"   收益率均值: {stats['mean']:.6f}")
    print(f"   收益率标准差: {stats['std']:.6f}")
    print(f"   偏度: {stats['skew']:.4f}")
    print(f"   峰度: {stats['kurtosis']:.4f}")
    print(f"   正收益率比例: {stats['positive_rate']*100:.1f}%")

    # Step 4: 计算 PSR
    print("\n[Step 4] 计算 PSR...")
    psr, sr_period = calculate_psr(returns)
    sr_annual = sr_period * np.sqrt(annualization_factor)

    print(f"   周期 Sharpe Ratio: {sr_period:.6f}")
    print(f"   年化 Sharpe Ratio: {sr_annual:.4f}")
    print(f"   PSR (SR > 0): {psr:.4f}")

    # Step 5: 计算 DSR
    print("\n[Step 5] 计算 DSR...")

    # 自动估算参数试验次数
    # 基于 Y9999 策略的实际参数网格
    param_grid = {
        'ma_span': [5, 10, 20, 50, 100],
        'tbm_barriers': [(1.0, 1.0), (1.5, 1.5), (2.0, 2.0), (2.5, 2.5)],
        'min_ret': [0.0, 0.001, 0.002],
        'meta_threshold': [0.4, 0.5, 0.6],
    }
    n_trials = estimate_optimal_trials(param_grid)
    print(f"   自动估算试验次数: {n_trials}")

    dsr, exp_max_sr, sr_period = calculate_dsr(returns, n_trials, annualization_factor)

    print(f"   参数试验次数: {n_trials}")
    print(f"   期望最大年化 Sharpe: {exp_max_sr:.4f}")
    print(f"   DSR: {dsr:.4f}")

    # Step 6: 输出验证结论
    print("\n" + "=" * 70)
    print("  验证结论")
    print("=" * 70)

    print(f"\n样本数: {stats['n_samples']}")
    print(f"时间范围: {returns.index.min().strftime('%Y-%m-%d')} ~ {returns.index.max().strftime('%Y-%m-%d')}")
    print("-" * 70)
    print(f"年化 Sharpe Ratio: {sr_annual:.4f}")
    print(f"PSR (SR > 0): {psr:.4f}")
    print(f"DSR (N={n_trials} trials): {dsr:.4f}")
    print("-" * 70)

    # PSR 结论
    if psr > 0.95:
        psr_verdict = "✅ ACCEPT (统计显著)"
    elif psr > 0.90:
        psr_verdict = "⚠️ BORDERLINE (边缘)"
    else:
        psr_verdict = "❌ REJECT (不显著)"

    # DSR 结论
    if dsr > 0.95:
        dsr_verdict = "✅ ACCEPT (非假阳性)"
    elif dsr > 0.90:
        dsr_verdict = "⚠️ BORDERLINE (边缘)"
    else:
        dsr_verdict = "❌ REJECT (假阳性风险)"

    print(f"PSR 结论: {psr_verdict}")
    print(f"DSR 结论: {dsr_verdict}")
    print("-" * 70)

    # 综合结论
    if dsr > 0.95 and psr > 0.95:
        final_verdict = "✅ 策略通过验证，具有统计显著性"
    elif dsr > 0.90 and psr > 0.90:
        final_verdict = "⚠️ 策略边缘通过，需要更多验证"
    else:
        final_verdict = "❌ 策略未通过验证，存在假阳性风险"

    print(f"\n最终结论: {final_verdict}")
    print("=" * 70)

    # Step 7: 保存结果
    results = {
        'n_samples': stats['n_samples'],
        'annualization_factor': annualization_factor,
        'n_trials': n_trials,
        'sr_period': sr_period,
        'sr_annual': sr_annual,
        'psr': psr,
        'dsr': dsr,
        'expected_max_sr': exp_max_sr,
        'mean_ret': stats['mean'],
        'std_ret': stats['std'],
        'skew': stats['skew'],
        'kurtosis': stats['kurtosis'],
        'positive_rate': stats['positive_rate'],
    }

    results_df = pd.DataFrame([results])
    results_path = os.path.join(FEATURES_DIR, 'dsr_validation_results.parquet')
    results_df.to_parquet(results_path)
    print(f"\n✅ 验证结果已保存: {results_path}")


if __name__ == "__main__":
    main()