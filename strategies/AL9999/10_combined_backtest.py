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
import json
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.AL9999.config import (
    FEATURES_DIR,
    FIGURES_DIR,
    BARS_DIR,
    META_MODEL_CONFIG,
    FILTER_FIRST_CONFIG,
    COMMISSION_RATE,
    SLIPPAGE_POINTS,
)
from strategies.AL9999.threshold_optimizer import build_threshold_report, select_best_threshold
from strategies.AL9999.backtest_utils import rolling_backtest
from afmlkit.utils.log import get_logger

logger = get_logger("Backtest")
sns.set_theme(style="whitegrid", context="paper")

# ============================================================
# 配置与常量
# ============================================================

ANNUALIZATION_FACTOR = 1500 # 每年约 1500 个 Dollar Bars
N_TRIALS = 214              # 08_dsr_validation.py 估算的参数试验总次数

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


def _resolve_meta_signal_paths(models_dir: str) -> tuple[str, str, str]:
    """
    Resolve preferred meta signal files based on selected_features.json.
    """
    default_oof = os.path.join(models_dir, 'meta_oof_signals.parquet')
    default_holdout = os.path.join(models_dir, 'meta_holdout_signals.parquet')
    selected_path = os.path.join(FEATURES_DIR, 'selected_features.json')

    if not os.path.exists(selected_path):
        return default_oof, default_holdout, "default"

    try:
        with open(selected_path, 'r', encoding='utf-8') as f:
            selected = json.load(f)
        scheme = selected.get("best_scheme", "")
    except (OSError, json.JSONDecodeError):
        return default_oof, default_holdout, "default"

    if not scheme or scheme == "full":
        return default_oof, default_holdout, "full"

    preferred_oof = os.path.join(models_dir, f"meta_oof_signals_{scheme}.parquet")
    preferred_holdout = os.path.join(models_dir, f"meta_holdout_signals_{scheme}.parquet")
    if os.path.exists(preferred_oof) and os.path.exists(preferred_holdout):
        return preferred_oof, preferred_holdout, scheme

    return default_oof, default_holdout, "default"


def load_honest_meta_signals(models_dir: str, precision_threshold: float) -> tuple[pd.DataFrame, pd.Timestamp, str]:
    """
    加载“无前视”元模型信号：
    - 训练期使用 OOF 预测
    - Holdout 期使用真正 OOS 预测

    :returns: (signals_df, holdout_start)
    """
    oof_path, holdout_path, scheme = _resolve_meta_signal_paths(models_dir)

    if not os.path.exists(oof_path):
        raise FileNotFoundError(f"缺少 OOF 文件: {oof_path}。请先运行 07_meta_model.py。")
    if not os.path.exists(holdout_path):
        raise FileNotFoundError(f"缺少 Holdout 文件: {holdout_path}。请先运行 07_meta_model.py。")

    oof_df = pd.read_parquet(oof_path)
    holdout_df = pd.read_parquet(holdout_path)

    oof_df = oof_df.copy()
    holdout_df = holdout_df.copy()
    oof_df['meta_pred'] = (oof_df['y_prob'] >= precision_threshold).astype(int)
    if 'meta_pred' not in holdout_df.columns:
        holdout_df['meta_pred'] = (holdout_df['y_prob'] >= precision_threshold).astype(int)

    signals_df = pd.concat(
        [
            oof_df[['meta_pred', 'y_prob']].rename(columns={'y_prob': 'meta_prob'}),
            holdout_df[['meta_pred', 'y_prob']].rename(columns={'y_prob': 'meta_prob'}),
        ],
        axis=0,
    )
    signals_df = signals_df[~signals_df.index.duplicated(keep='last')].sort_index()

    holdout_start = holdout_df.index.min()
    return signals_df, holdout_start, scheme


def _apply_threshold_to_signals(
    signals: pd.DataFrame,
    threshold: float,
    side_mode: str,
    short_penalty_delta: float,
) -> pd.DataFrame:
    """
    Apply threshold and side governance to signal table.
    """
    out = signals.copy()
    if side_mode == "both_with_short_penalty":
        short_threshold = threshold + short_penalty_delta
        out["meta_pred"] = np.where(
            out["side"] == -1,
            (out["meta_prob"] >= short_threshold).astype(int),
            (out["meta_prob"] >= threshold).astype(int),
        )
    else:
        out["meta_pred"] = (out["meta_prob"] >= threshold).astype(int)
    return out


def _perf_from_trades(trades_df: pd.DataFrame) -> dict:
    """
    Convert trades dataframe to existing performance schema.
    """
    if trades_df is None or len(trades_df) == 0:
        return {k: 0 for k in ['total_pnl', 'annual_pnl', 'sharpe', 'mdd', 'calmar', 'win_rate', 'profit_factor', 'psr', 'dsr', 'n_trades']}

    use_cols = trades_df.copy()
    use_cols["exit_time"] = pd.to_datetime(use_cols["exit_time"])
    use_cols = use_cols.sort_values("exit_time").set_index("exit_time")
    return calculate_performance(use_cols["net_pnl"], use_cols["net_ret"])


def _build_filter_first_threshold_report(
    signals: pd.DataFrame,
    bars: pd.DataFrame,
    oos_start: pd.Timestamp,
    threshold_grid: list[float],
    side_mode: str,
    short_penalty_delta: float,
    guard_cfg: dict,
    baseline_threshold: float,
    shrinkage_min: float,
    shrinkage_max: float,
) -> tuple[pd.DataFrame, Optional[dict], int]:
    """
    Build threshold report using guard-aware single-position rolling backtest.
    """
    guard_enabled = bool(guard_cfg.get("enabled", False))
    min_hold_bars = int(guard_cfg.get("min_hold_bars", 0))
    cooldown_bars = int(guard_cfg.get("cooldown_bars", 0))
    reverse_confirmation_delta = float(guard_cfg.get("reverse_confirmation_delta", 0.0))

    # Use a neutral baseline (no short penalty, no execution guard) so shrinkage is
    # measured versus the original combined strategy, not versus a self-penalized baseline.
    base_signals = signals.copy()
    base_signals["meta_pred"] = (base_signals["meta_prob"] >= baseline_threshold).astype(int)
    base_trades = rolling_backtest(
        base_signals,
        bars,
        use_meta_filter=True,
        side_mode="both",
        guard_enabled=False,
        min_hold_bars=0,
        cooldown_bars=0,
        reverse_confirmation_delta=0.0,
        entry_threshold=baseline_threshold,
    )
    baseline_oos_n = int((pd.to_datetime(base_trades["exit_time"]) >= oos_start).sum()) if len(base_trades) > 0 else 0

    rows = []
    for th in threshold_grid:
        threshold_signals = _apply_threshold_to_signals(
            signals=signals,
            threshold=float(th),
            side_mode=side_mode,
            short_penalty_delta=short_penalty_delta,
        )
        test_side_mode = "both" if side_mode == "both_with_short_penalty" else side_mode
        trades = rolling_backtest(
            threshold_signals,
            bars,
            use_meta_filter=True,
            side_mode=test_side_mode,
            guard_enabled=guard_enabled,
            min_hold_bars=min_hold_bars,
            cooldown_bars=cooldown_bars,
            reverse_confirmation_delta=reverse_confirmation_delta,
            entry_threshold=float(th),
        )
        if len(trades) > 0:
            trades["exit_time"] = pd.to_datetime(trades["exit_time"])
            trades = trades.sort_values("exit_time")
            trades["net_pnl"] = (
                trades["pnl"] - (trades["entry_price"] + trades["exit_price"]) * COMMISSION_RATE - SLIPPAGE_POINTS * 2
            )
            trades["net_ret"] = trades["net_pnl"] / trades["entry_price"]
            oos_trades = trades[trades["exit_time"] >= oos_start].copy()
        else:
            oos_trades = pd.DataFrame()

        full_perf = _perf_from_trades(trades) if len(trades) > 0 else _perf_from_trades(pd.DataFrame())
        oos_perf = _perf_from_trades(oos_trades)
        rows.append(
            {
                "threshold": float(th),
                "full_n": int(full_perf.get("n_trades", 0)),
                "oos_n": int(oos_perf.get("n_trades", 0)),
                "oos_sharpe": float(oos_perf.get("sharpe", 0.0)),
                "oos_dsr": float(oos_perf.get("dsr", 0.0)),
            }
        )

    report = build_threshold_report(rows=rows, baseline_trade_count=max(1, baseline_oos_n))
    best = select_best_threshold(report, shrinkage_min=shrinkage_min, shrinkage_max=shrinkage_max)
    return report, best, baseline_oos_n

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
    models_dir = FEATURES_DIR.replace('features', 'models')
    threshold = META_MODEL_CONFIG.get('precision_threshold', 0.5)
    meta_signals, oos_start, scheme_used = load_honest_meta_signals(
        models_dir=models_dir,
        precision_threshold=threshold,
    )

    # 2. 对齐（只保留有“可交易时可得”的预测样本）
    common_idx = tbm.index.intersection(meta_signals.index)
    tbm = tbm.loc[common_idx].copy()
    tbm[['meta_pred', 'meta_prob']] = meta_signals.loc[common_idx, ['meta_pred', 'meta_prob']]
    logger.info(f"  可用无前视样本: {len(tbm)}")
    logger.info(f"  OOS 起点(来自 holdout): {oos_start}")
    logger.info(f"  Meta 信号方案: {scheme_used}")

    # 2.1 Filter-First 阈值扫描（guard-aware rolling backtest）
    filter_cfg = FILTER_FIRST_CONFIG
    threshold_grid = filter_cfg.get("threshold_grid", [threshold])
    side_mode = filter_cfg.get("side_mode", "both")
    short_penalty_delta = float(filter_cfg.get("short_penalty_delta", 0.0))
    guard_cfg = filter_cfg.get("execution_guard", {})
    threshold_report, best_threshold, baseline_oos_n = _build_filter_first_threshold_report(
        signals=tbm,
        bars=pd.read_parquet(os.path.join(BARS_DIR, 'dollar_bars_target4.parquet')),
        oos_start=oos_start,
        threshold_grid=threshold_grid,
        side_mode=side_mode,
        short_penalty_delta=short_penalty_delta,
        guard_cfg=guard_cfg,
        baseline_threshold=threshold,
        shrinkage_min=filter_cfg.get("shrinkage_min", 0.0),
        shrinkage_max=filter_cfg.get("shrinkage_max", 1.0),
    )
    if best_threshold is not None:
        threshold = float(best_threshold["threshold"])
        logger.info(
            "  Filter-First 选中阈值: %.2f (oos_dsr=%.4f, oos_sharpe=%.4f, shrinkage=%.4f)",
            threshold,
            best_threshold.get("oos_dsr", 0.0),
            best_threshold.get("oos_sharpe", 0.0),
            best_threshold.get("trade_shrinkage", 0.0),
        )
    else:
        logger.warning("  Filter-First 未找到满足收缩约束的阈值，回退默认阈值: %.2f", threshold)

    threshold_report.to_parquet(os.path.join(FEATURES_DIR, "filter_first_threshold_report.parquet"), index=False)
    logger.info(f"✅ 阈值扫描报告已保存至: {FEATURES_DIR}/filter_first_threshold_report.parquet")
    
    # 3. 使用 guard-aware 单仓位回测主口径
    logger.info("[Step 2] 运行 guard-aware 单仓位回测...")
    bars = pd.read_parquet(os.path.join(BARS_DIR, 'dollar_bars_target4.parquet'))

    primary_trades = rolling_backtest(
        tbm,
        bars,
        use_meta_filter=False,
        side_mode="both",
    )
    if len(primary_trades) > 0:
        primary_trades["net_pnl"] = (
            primary_trades["pnl"] - (primary_trades["entry_price"] + primary_trades["exit_price"]) * COMMISSION_RATE - SLIPPAGE_POINTS * 2
        )
        primary_trades["net_ret"] = primary_trades["net_pnl"] / primary_trades["entry_price"]

    selected_signals = _apply_threshold_to_signals(
        signals=tbm,
        threshold=threshold,
        side_mode=side_mode,
        short_penalty_delta=short_penalty_delta,
    )
    combined_side_mode = "both" if side_mode == "both_with_short_penalty" else side_mode
    combined_trades = rolling_backtest(
        selected_signals,
        bars,
        use_meta_filter=True,
        side_mode=combined_side_mode,
        guard_enabled=bool(guard_cfg.get("enabled", False)),
        min_hold_bars=int(guard_cfg.get("min_hold_bars", 0)),
        cooldown_bars=int(guard_cfg.get("cooldown_bars", 0)),
        reverse_confirmation_delta=float(guard_cfg.get("reverse_confirmation_delta", 0.0)),
        entry_threshold=threshold,
    )
    if len(combined_trades) > 0:
        combined_trades["net_pnl"] = (
            combined_trades["pnl"] - (combined_trades["entry_price"] + combined_trades["exit_price"]) * COMMISSION_RATE - SLIPPAGE_POINTS * 2
        )
        combined_trades["net_ret"] = combined_trades["net_pnl"] / combined_trades["entry_price"]

    primary_trades.to_parquet(os.path.join(FEATURES_DIR, "filter_first_primary_trades.parquet"), index=False)
    combined_trades.to_parquet(os.path.join(FEATURES_DIR, "filter_first_combined_trades.parquet"), index=False)
    logger.info(f"✅ Filter-First 交易明细已保存至: {FEATURES_DIR}/filter_first_primary_trades.parquet")
    logger.info(f"✅ Filter-First 交易明细已保存至: {FEATURES_DIR}/filter_first_combined_trades.parquet")
    
    # 4. 绩效统计 (IS / OOS 分开)
    logger.info("[Step 3] 计算绩效指标 (全样本 & IS/OOS 分解)...")
    
    # 全样本
    primary_perf = _perf_from_trades(primary_trades)
    combined_perf = _perf_from_trades(combined_trades)
    
    # IS / OOS 分解 (仅对 Combined)
    combined_is_trades = combined_trades[pd.to_datetime(combined_trades["exit_time"]) < oos_start].copy() if len(combined_trades) > 0 else pd.DataFrame()
    combined_oos_trades = combined_trades[pd.to_datetime(combined_trades["exit_time"]) >= oos_start].copy() if len(combined_trades) > 0 else pd.DataFrame()
    combined_is_perf = _perf_from_trades(combined_is_trades)
    combined_oos_perf = _perf_from_trades(combined_oos_trades)
    
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
    logger.info(f"  Side Mode: {side_mode}")

    selected_oos_n = int((pd.to_datetime(combined_trades["exit_time"]) >= oos_start).sum()) if len(combined_trades) > 0 else 0
    selected_trade_shrinkage = float(best_threshold.get("trade_shrinkage", 0.0)) if best_threshold is not None else 0.0
    selection_df = pd.DataFrame(
        [
            {
                "selected_threshold": float(threshold),
                "side_mode": side_mode,
                "baseline_oos_n": int(baseline_oos_n),
                "selected_oos_n": selected_oos_n,
                "trade_shrinkage": selected_trade_shrinkage,
                "shrinkage_min": float(filter_cfg.get("shrinkage_min", 0.0)),
                "shrinkage_max": float(filter_cfg.get("shrinkage_max", 1.0)),
                "scheme_used": scheme_used,
            }
        ]
    )
    selection_df.to_parquet(os.path.join(FEATURES_DIR, "filter_first_selection.parquet"), index=False)
    logger.info(f"✅ Filter-First 选择元数据已保存至: {FEATURES_DIR}/filter_first_selection.parquet")
    
    # 5. 可视化
    logger.info("[Step 4] 生成高对比图表...")
    
    # 加载原始价格数据进行对比
    price_bh = bars['close'] - bars['close'].iloc[0] # 计算买入持有收益 (点数)
    
    # 累积收益曲线 (高对比配色)
    plt.figure(figsize=(14, 8))
    
    # 背景阴影区分 IS / OOS
    plt.axvspan(bars.index.min(), pd.to_datetime(oos_start), color='gray', alpha=0.05, label='In-Sample (Training)')
    plt.axvspan(pd.to_datetime(oos_start), bars.index.max(), color='green', alpha=0.05, label='Out-of-Sample (Live/Test)')
    plt.axvline(pd.to_datetime(oos_start), color='darkred', linestyle='--', alpha=0.5, lw=1.5)
    
    # PnL 多曲线对比
    plt.plot(price_bh.index, price_bh.values, label='Benchmark: Buy & Hold (AL9999)', color='#555555', alpha=0.3, linestyle=(0, (3, 5, 1, 5)))
    if len(primary_trades) > 0:
        p = primary_trades.copy()
        p["exit_time"] = pd.to_datetime(p["exit_time"])
        p = p.sort_values("exit_time")
        plt.plot(p["exit_time"], p['net_pnl'].cumsum(), label='Primary Model (Net PnL)', alpha=0.7, color='#9b59b6', lw=1.5)
    if len(combined_trades) > 0:
        c = combined_trades.copy()
        c["exit_time"] = pd.to_datetime(c["exit_time"])
        c = c.sort_values("exit_time")
        plt.plot(c["exit_time"], c['net_pnl'].cumsum(), label='Combined Strategy (Net PnL)', color='#e67e22', lw=2.5)
    
    plt.title("AL9999 Strategy Cumulative PnL: In-Sample vs Out-of-Sample Comparison", fontsize=15, fontweight='bold')
    plt.ylabel("Net PnL Marks (Points)", fontsize=12)
    plt.xlabel("Time", fontsize=12)
    plt.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)
    plt.grid(True, alpha=0.2, linestyle=':')
    
    # 标记 OOS 夏普
    plt.text(pd.to_datetime(oos_start) + pd.Timedelta(days=30), combined_perf['total_pnl']*0.1,
             f"OOS Sharpe: {combined_oos_perf['sharpe']:.2f}\nDSR: {combined_oos_perf['dsr']:.4f}", 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'), fontweight='bold', color='green')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "10_cumulative_pnl.png"), dpi=200)
    plt.close()
    
    # 回撤曲线
    plt.figure(figsize=(12, 4))
    if len(combined_trades) > 0:
        draw_input = combined_trades.copy()
        draw_input["exit_time"] = pd.to_datetime(draw_input["exit_time"])
        draw_input = draw_input.sort_values("exit_time")
        cum_pnl = draw_input['net_pnl'].cumsum()
        drawdown = cum_pnl - cum_pnl.cummax()
        x_vals = draw_input["exit_time"]
    else:
        drawdown = pd.Series([0.0])
        x_vals = pd.RangeIndex(1)
    plt.fill_between(x_vals, drawdown.values, 0, color='#e74c3c', alpha=0.3)
    plt.title("Combined Strategy Drawdown (Points)", fontsize=14)
    plt.ylabel("Drawdown")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIGURES_DIR, "10_drawdown.png"), dpi=150)
    plt.close()
    
    # 月度收益热力图
    if len(combined_trades) > 0:
        month_input = combined_trades.copy()
        month_input["exit_time"] = pd.to_datetime(month_input["exit_time"])
        month_input = month_input.sort_values("exit_time").set_index("exit_time")
        month_input['month'] = month_input.index.to_period('M')
        monthly_ret = month_input.groupby('month')['net_pnl'].sum()
    else:
        monthly_ret = pd.Series(dtype=float)
    monthly_pivot = monthly_ret.to_frame()
    monthly_pivot['year'] = monthly_pivot.index.year
    monthly_pivot['month_num'] = monthly_pivot.index.month
    pivot_table = monthly_pivot.pivot_table(index='year', columns='month_num', values='net_pnl').fillna(0)
    
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
    if combined_oos_perf['dsr'] > 0.95:
        logger.info("🚀 判定结果：✅ 策略具备实盘潜力 (DSR > 95%)")
    else:
        logger.info("⚠️ 判定结果：❌ 策略存在过拟合风险 (DSR <= 95%)")

if __name__ == "__main__":
    main()
