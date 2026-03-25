"""
Strategy Backtesting — Performance & Risk Evaluation
=====================================================

本脚本通过整合 Primary 信号、Meta 过滤与 Bet Sizing，在大连商品交易所 (DCE) 棕榈油样本上进行回测。
依据 AFML 第 14 章标准，计算以下核心指标：
  - Sharpe Ratio (年化)
  - Probabilistic Sharpe Ratio (PSR)：考虑偏度与峰度的夏普显著性
  - Deflated Sharpe Ratio (DSR)：剔除过拟合假设多次试验的假阳性概率
  - Max Drawdown：最大回撤分析
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────
BARS_PATH        = Path("outputs/dollar_bars/dollar_bars_freq20.csv") # 原始持续 K 线
SIZES_PATH       = Path("outputs/bet_sizing/final_bet_sizes.csv")     # 仓位序列
OUTPUT_DIR       = Path("outputs/backtest")

# ── Transaction Costs ──────────────────────────────────────────────
FEES_PER_TRADE = 0.0001  # 1 bps (1/10000) — 涵盖交易手续费
SLIPPAGE      = 0.0001   # 1 bps — 涵盖滑点
COST_PER_SIDE = FEES_PER_TRADE + SLIPPAGE

# ── Performance Functions ───────────────────────────────────────────

def calculate_psr(returns, benchmark_sr=0):
    """
    计算概率夏普比率 (PSR)，衡量夏普比率的统计显著性。
    """
    if len(returns) < 50: return 0.0
    n = len(returns)
    sr = returns.mean() / returns.std(ddof=1)
    skew = returns.skew()
    kurt = returns.kurtosis() + 3 
    
    sigma_sr = np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1))
    psr = norm.cdf((sr - benchmark_sr) / sigma_sr)
    return psr

def calculate_dsr(returns, n_trials=100, sr_std=0.5, annualization_factor=252):
    """
    计算偏误消除夏普比率 (DSR)，防止通过多次参数搜索得到的过拟合。
    
    :param sr_std: 年化夏普在不同试验间的标准差 (通常取 0.5)
    :param annualization_factor: 用于将年化 benchmark 转换为周期内 benchmark
    """
    # 计算年化期望最大夏普 (Expected Maximum Sharpe)
    expected_max_sr_annual = sr_std * ((1 - 0.5772) * norm.ppf(1 - 1/n_trials) + 0.5772 * norm.ppf(1 - 1/n_trials * np.exp(-1)))
    
    # 关键：由于 calculate_psr 内部使用的是周期收益率的 SR，
    # 我们必须将年化 Benchmark 转换回周期级别进行比较
    benchmark_sr_period = expected_max_sr_annual / np.sqrt(annualization_factor)
    
    dsr = calculate_psr(returns, benchmark_sr=benchmark_sr_period)
    return dsr, expected_max_sr_annual

# ── Main Backtest ───────────────────────────────────────────────────

def run_backtest():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("[Backtest] 加载数据与聚合仓位...")
    # 1. 加载原始价格数据进行连续盈亏计算
    bars = pd.read_csv(BARS_PATH, index_col='timestamp', parse_dates=True)
    # 2. 加载离散化后的仓位序列
    sizes = pd.read_csv(SIZES_PATH, index_col=0, parse_dates=True)
    
    # ── 对齐数据 ──────────────────────────────────────────────────
    # 我们的仓位是事件驱动的，需要前向填充到基础 K 线
    df = bars[['close']].copy()
    df = df.join(sizes[['discretized_size']], how='left')
    df['pos'] = df['discretized_size'].ffill().fillna(0)
    
    print("[Backtest] 计算策略收益率 (无前视偏差)...")
    # 计算 K 线对 K 线的收益率
    df['ret'] = df['close'].pct_change()
    
    # 核心：当前时刻 t 的收益由 t-1 时刻的持仓决定（Pos_t-1 * Ret_t）
    df['raw_strat_ret'] = df['pos'].shift(1) * df['ret']
    
    # ── 扣除交易成本 ──────────────────────────────────────────────
    # 计算仓位变化：|Pos_t - Pos_t-1|
    df['pos_diff'] = df['pos'].diff().abs()
    # 交易成本 = 仓位变化 * 成本率 (单边)
    df['turnover_costs'] = df['pos_diff'] * COST_PER_SIDE
    
    df['strat_ret'] = df['raw_strat_ret'] - df['turnover_costs']
    df['strat_ret'] = df['strat_ret'].fillna(0)
    
    # 累积收益
    df['cum_ret'] = (1 + df['strat_ret']).cumprod()
    
    # ── 指标计算 ──────────────────────────────────────────────────
    total_ret = df['cum_ret'].iloc[-1] - 1
    
    # 年化因子：根据样本频率估算
    avg_delta = df.index.to_series().diff().mean()
    annualization_factor = (pd.Timedelta(days=365) / avg_delta)
    
    # 常规夏普比率
    std = df['strat_ret'].std()
    if std > 0:
        sr = (df['strat_ret'].mean() / std) * np.sqrt(annualization_factor)
    else:
        sr = 0
        
    # 回撤分析
    rolling_max = df['cum_ret'].cummax()
    drawdown = (df['cum_ret'] - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    # AFML 健壮性指标
    psr = calculate_psr(df['strat_ret'])
    # 注意：这里的 n_trials 设为 100，表示我们假设进行了 100 次参数试验
    dsr, exp_max_sr = calculate_dsr(df['strat_ret'], n_trials=100, annualization_factor=annualization_factor)
    
    print("\n" + "="*35)
    print("      AFML BACKTEST RESULTS")
    print("="*35)
    print(f"Total Return:       {total_ret:.2%}")
    print(f"Annualized Sharpe: {sr:.2f}")
    print(f"Max Drawdown:      {max_dd:.2%}")
    print(f"PSR (SR > 0):      {psr:.4f}")
    print(f"DSR (N=100 trials): {dsr:.4f}")
    print(f"Verdict:            {'✅ ACCEPT' if dsr > 0.95 else '❌ REJECT (False Positive Risk)'}")
    print("="*35)
    
    # ── 导出结果 ──────────────────────────────────────────────────
    df.to_csv(OUTPUT_DIR / "backtest_full_results.csv")
    
    # ── 可视化 ────────────────────────────────────────────────────
    print("\n[可视化] 绘制性能分析曲线...")
    plot_performance(df, drawdown, OUTPUT_DIR)
    print(f"✓ 结果已保存至 {OUTPUT_DIR}/")

def plot_performance(df, drawdown, output_dir):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # 1. 累积净值
    ax1.plot(df.index, df['cum_ret'], label='Final Strategy (Primary + Meta + BetSize)', 
             color='#2980b9', lw=1.5)
    ax1.axhline(1.0, color='black', linestyle='--', alpha=0.3)
    ax1.set_title("Strategy Cumulative Equity Curve", fontweight="bold")
    ax1.set_ylabel("Equity")
    ax1.legend(); ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. 回撤
    ax2.fill_between(df.index, drawdown, 0, color='#e74c3c', alpha=0.3, label='Drawdown')
    ax2.set_ylabel("Drawdown")
    ax2.set_ylim(min(drawdown.min()*1.1, -0.05), 0.01)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # 3. 实时仓位
    ax3.plot(df.index, df['pos'], color='#27ae60', lw=1, alpha=0.7)
    ax3.set_ylabel("Bet Size (Exposure)")
    ax3.set_xlabel("Time")
    ax3.grid(True, linestyle=":", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / "performance_summary.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    run_backtest()
