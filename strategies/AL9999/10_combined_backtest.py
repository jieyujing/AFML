"""
10_combined_backtest.py - AL9999 组合策略真实回测

本脚本将 Primary Model 和 Meta Model 整合，进行由于真实交易场景的回测：
1. 信号过滤：仅保留 Meta Model 预测为 1 的 Primary 信号。
2. 成本模拟：扣除双边手续费 (0.23 bp) 和滑点 (1.0 pnt/RT)。
3. 指标统计：计算年化夏普、PSR、DSR、最大回撤等。
4. 可视化：主模型 vs 组合策略、回撤曲线、月度收益热力图。

AFML 规范：
- 考虑参数试验次数 (N=214) 对 DSR 的影响。
- 使用 TBM (Triple Barrier Method) 的离散收益计算累积净值。
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.AL9999.config import FEATURES_DIR, FIGURES_DIR, BARS_DIR, CONTRACT_MULTIPLIER
from afmlkit.utils.log import get_logger

logger = get_logger("Backtest")
sns.set_theme(style="whitegrid", context="paper")

# ============================================================
# 配置与常量
# ============================================================

# 交易成本设值
COMMISSION_RATE = 0.000023  # 手续费率 (双边约 0.23 bp)
SLIPPAGE_POINTS = 0.5       # 单边滑点 (双边 1.0 点)
ANNUALIZATION_FACTOR = 1500 # 每年约 1500 个 Dollar Bars
N_TRIALS = 214              # 08_dsr_validation.py 估算的参数试验总次数
OOS_START_DATE = '2024-01-01' # 样本内/外分割日期

# ============================================================
# 指标计算函数
# ============================================================

def calculate_psr(returns: pd.Series, benchmark_sr: float = 0.0) -> float:
    """计算概率夏普比率 (Probabilistic Sharpe Ratio)。"""
    n = len(returns)
    if n < 5: return 0.0
    std = returns.std()
    if std == 0: return 0.0
    sr = returns.mean() / std
    skew = returns.skew()
    kurt = returns.kurtosis() + 3
    sigma_sr = np.sqrt((1 - skew * sr + (kurt - 1) / 4 * sr**2) / (n - 1))
    z_stat = (sr - benchmark_sr) / sigma_sr
    return norm.cdf(z_stat)

def calculate_dsr(returns: pd.Series, n_trials: int) -> float:
    """计算 Deflated Sharpe Ratio。"""
    if len(returns) < 5: return 0.0
    sr_std = 0.5 # 经验值
    gamma = 0.5772
    exp_max_sr_annual = sr_std * ((1 - gamma) * norm.ppf(1 - 1/n_trials) + gamma * norm.ppf(1 - 1/(n_trials * np.exp(-1))))
    benchmark_sr_period = exp_max_sr_annual / np.sqrt(ANNUALIZATION_FACTOR)
    return calculate_psr(returns, benchmark_sr_period)

def calculate_performance(pnl_points: pd.Series, rets: pd.Series) -> dict:
    """计算核心回测指标。"""
    if len(pnl_points) == 0:
        return {k: 0 for k in ['total_pnl', 'annual_pnl', 'sharpe', 'mdd', 'calmar', 'win_rate', 'profit_factor', 'psr', 'dsr', 'n_trades']}
    
    cum_pnl = pnl_points.cumsum()
    n_days = (pnl_points.index.max() - pnl_points.index.min()).days
    years = max(n_days / 365, 0.1)
    
    annual_ret_points = pnl_points.sum() / years
    sr_annual = rets.mean() / rets.std() * np.sqrt(ANNUALIZATION_FACTOR) if rets.std() > 0 else 0
    
    # 回撤
    cum_max = cum_pnl.cummax()
    drawdown = cum_pnl - cum_max
    mdd = drawdown.min()
    
    # 盈亏分布
    wins = pnl_points[pnl_points > 0]
    losses = pnl_points[pnl_points < 0]
    win_rate = len(wins) / len(pnl_points)
    profit_factor = abs(wins.sum() / losses.sum()) if len(losses) > 0 else np.inf
    
    return {
        'total_pnl': pnl_points.sum(),
        'annual_pnl': annual_ret_points,
        'sharpe': sr_annual,
        'mdd': mdd,
        'calmar': abs(annual_ret_points / mdd) if mdd != 0 else 0,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'psr': calculate_psr(rets),
        'dsr': calculate_dsr(rets, N_TRIALS),
        'n_trades': len(pnl_points)
    }

# ============================================================
# 主回测流程
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("  AL9999 Primary + Meta Combined Backtest")
    logger.info("=" * 60)
    
    # 1. 加载数据
    logger.info("[Step 1] 加载信号与模型...")
    tbm = pd.read_parquet(os.path.join(FEATURES_DIR, 'tbm_results.parquet'))
    meta_features = pd.read_parquet(os.path.join(FEATURES_DIR, 'events_features.parquet'))
    meta_model = joblib.load(os.path.join(FEATURES_DIR.replace('features', 'models'), 'meta_model.pkl'))
    
    # 2. 对齐与特征工程
    common_idx = tbm.index.intersection(meta_features.index)
    tbm = tbm.loc[common_idx]
    X = meta_features.loc[common_idx]
    feature_cols = [c for c in X.columns if c.startswith('feat_')]
    X = X[feature_cols].fillna(0)
    
    # 3. Meta Model 过滤
    meta_pred = meta_model.predict(X)
    tbm['meta_pred'] = meta_pred
    
    # 4. 计算成本后的真实 PnL
    logger.info("[Step 2] 模拟交易成本与滑点...")
    
    def apply_costs(row):
        # 手续费：双边 (开仓 + 平仓)
        # 简化计算：0.23 bp * entry + 0.23 bp * exit
        comm = (row['entry_price'] + row['exit_price']) * COMMISSION_RATE
        # 滑点：单边 0.5 点，双边 1.0 点
        slip = SLIPPAGE_POINTS * 2
        # 净收益 (点数)
        net_pnl = row['pnl'] - comm - slip
        # 净收益率
        net_ret = net_pnl / row['entry_price']
        return pd.Series({'net_pnl': net_pnl, 'net_ret': net_ret})

    # Primary (无过滤)
    primary_costs = tbm.apply(apply_costs, axis=1)
    tbm['primary_net_pnl'] = primary_costs['net_pnl']
    tbm['primary_net_ret'] = primary_costs['net_ret']
    
    # Combined (Meta 过滤)
    combined_tbm = tbm[tbm['meta_pred'] == 1].copy()
    
    # 5. 绩效统计 (IS / OOS 分开)
    logger.info("[Step 3] 计算绩效指标 (全样本 & IS/OOS 分解)...")
    
    # 全样本
    primary_perf = calculate_performance(tbm['primary_net_pnl'], tbm['primary_net_ret'])
    combined_perf = calculate_performance(combined_tbm['primary_net_pnl'], combined_tbm['primary_net_ret'])
    
    # IS / OOS 分解 (仅对 Combined)
    is_mask = (combined_tbm.index < OOS_START_DATE)
    oos_mask = (combined_tbm.index >= OOS_START_DATE)
    combined_is_perf = calculate_performance(combined_tbm.loc[is_mask, 'primary_net_pnl'], combined_tbm.loc[is_mask, 'primary_net_ret'])
    combined_oos_perf = calculate_performance(combined_tbm.loc[oos_mask, 'primary_net_pnl'], combined_tbm.loc[oos_mask, 'primary_net_ret'])
    
    # 输出报表
    report = pd.DataFrame({
        'Metric': ['信号数', '总收益(点)', '年化收益(点)', '年化夏普', '最大回撤(点)', '胜率', '盈亏比', 'DSR'],
        'Primary (Full)': [
            primary_perf['n_trades'], primary_perf['total_pnl'], primary_perf['annual_pnl'],
            primary_perf['sharpe'], primary_perf['mdd'], primary_perf['win_rate'], primary_perf['profit_factor'], primary_perf['dsr']
        ],
        'Combined (Full)': [
            combined_perf['n_trades'], combined_perf['total_pnl'], combined_perf['annual_pnl'],
            combined_perf['sharpe'], combined_perf['mdd'], combined_perf['win_rate'], combined_perf['profit_factor'], combined_perf['dsr']
        ],
        'Combined (IS)': [
            combined_is_perf['n_trades'], combined_is_perf['total_pnl'], combined_is_perf['annual_pnl'],
            combined_is_perf['sharpe'], combined_is_perf['mdd'], combined_is_perf['win_rate'], combined_is_perf['profit_factor'], combined_is_perf['dsr']
        ],
        'Combined (OOS)': [
            combined_oos_perf['n_trades'], combined_oos_perf['total_pnl'], combined_oos_perf['annual_pnl'],
            combined_oos_perf['sharpe'], combined_oos_perf['mdd'], combined_oos_perf['win_rate'], combined_oos_perf['profit_factor'], combined_oos_perf['dsr']
        ]
    })
    
    logger.info("\n" + report.to_string(index=False))
    
    # 6. 可视化
    logger.info("[Step 4] 生成高对比图表...")
    
    # 加载原始价格数据进行对比
    bars = pd.read_parquet(os.path.join(BARS_DIR, 'dollar_bars_target4.parquet'))
    price_bh = bars['close'] - bars['close'].iloc[0] # 计算买入持有收益 (点数)
    
    # 累积收益曲线 (高对比配色)
    plt.figure(figsize=(14, 8))
    
    # 背景阴影区分 IS / OOS
    plt.axvspan(bars.index.min(), pd.to_datetime(OOS_START_DATE), color='gray', alpha=0.05, label='In-Sample (Training)')
    plt.axvspan(pd.to_datetime(OOS_START_DATE), bars.index.max(), color='green', alpha=0.05, label='Out-of-Sample (Live/Test)')
    plt.axvline(pd.to_datetime(OOS_START_DATE), color='darkred', linestyle='--', alpha=0.5, lw=1.5)
    
    # PnL 多曲线对比
    plt.plot(price_bh.index, price_bh.values, label='Benchmark: Buy & Hold (AL9999)', color='#555555', alpha=0.3, linestyle=(0, (3, 5, 1, 5)))
    plt.plot(tbm.index, tbm['primary_net_pnl'].cumsum(), label='Primary Model (Net PnL)', alpha=0.7, color='#9b59b6', lw=1.5)
    plt.plot(combined_tbm.index, combined_tbm['primary_net_pnl'].cumsum(), label='Combined Strategy (Net PnL)', color='#e67e22', lw=2.5)
    
    plt.title("AL9999 Strategy Cumulative PnL: In-Sample vs Out-of-Sample Comparison", fontsize=15, fontweight='bold')
    plt.ylabel("Net PnL Marks (Points)", fontsize=12)
    plt.xlabel("Time", fontsize=12)
    plt.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
    plt.grid(True, alpha=0.2, linestyle=':')
    
    # 标记 OOS 夏普
    plt.text(pd.to_datetime(OOS_START_DATE) + pd.Timedelta(days=30), combined_perf['total_pnl']*0.1, 
             f"OOS Sharpe: {combined_oos_perf['sharpe']:.2f}\nDSR: {combined_oos_perf['dsr']:.4f}", 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'), fontweight='bold', color='green')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "10_cumulative_pnl.png"), dpi=200)
    plt.close()
    
    # 回撤曲线
    plt.figure(figsize=(12, 4))
    cum_pnl = combined_tbm['primary_net_pnl'].cumsum()
    drawdown = cum_pnl - cum_pnl.cummax()
    plt.fill_between(drawdown.index, drawdown.values, 0, color='#e74c3c', alpha=0.3)
    plt.title("Combined Strategy Drawdown (Points)", fontsize=14)
    plt.ylabel("Drawdown")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, "10_drawdown.png"), dpi=150)
    plt.close()
    
    # 月度收益热力图
    combined_tbm['month'] = combined_tbm.index.to_period('M')
    monthly_ret = combined_tbm.groupby('month')['primary_net_pnl'].sum()
    monthly_pivot = monthly_ret.to_frame()
    monthly_pivot['year'] = monthly_pivot.index.year
    monthly_pivot['month_num'] = monthly_pivot.index.month
    pivot_table = monthly_pivot.pivot_table(index='year', columns='month_num', values='primary_net_pnl').fillna(0)
    
    plt.figure(figsize=(12, 5))
    sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="RdYlGn", center=0, cbar_kws={'label': 'Net PnL (Points)'})
    plt.title("Monthly Net PnL Heatmap (Points)", fontsize=14)
    plt.xlabel("Month")
    plt.ylabel("Year")
    plt.savefig(os.path.join(FIGURES_DIR, "10_monthly_heatmap.png"), dpi=150)
    plt.close()
    
    # 保存统计报告
    report.to_parquet(os.path.join(FEATURES_DIR, 'backtest_stats.parquet'))
    logger.info(f"✅ 回测统计已保存至: {FEATURES_DIR}/backtest_stats.parquet")
    logger.info(f"✅ 图表已保存至: {FIGURES_DIR}/ (10_cumulative_pnl.png, 10_drawdown.png, 10_monthly_heatmap.png)")
    
    # 判定
    if combined_perf['dsr'] > 0.95:
        logger.info("🚀 判定结果：✅ 策略具备实盘潜力 (DSR > 95%)")
    else:
        logger.info("⚠️ 判定结果：❌ 策略存在过拟合风险 (DSR <= 95%)")

if __name__ == "__main__":
    main()
