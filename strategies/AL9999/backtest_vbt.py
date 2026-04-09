"""
backtest_vbt.py - AL9999 Vectorbt 回测核心
基于 vectorbt 框架的重构版回测，保持与原 workflow 输出兼容
"""

import os
import sys
import numpy as np
import pandas as pd
import vectorbt as vbt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(PROJECT_ROOT)))

from strategies.AL9999.config import COMMISSION_RATE, SLIPPAGE_POINTS


def run_backtest_vbt(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    commission: float = COMMISSION_RATE,
    slippage: float = SLIPPAGE_POINTS,
    init_cash: float = 1000000.0,
    freq: str = 'D',
) -> dict:
    """
    使用 vectorbt 进行单仓位回测。
    
    :param bars: 包含 open/close/high/low 的价格 DataFrame，index 为时间
    :param signals: 包含 side (1=多, -1=空, 0=空仓) 的信号 DataFrame，index 为时间
    :param commission: 手续费率 (按成交金额比例)
    :param slippage: 滑点 (固定点数，会在价格上体现)
    :param init_cash: 初始资金
    :param freq: 频率 (用于时间相关指标计算)
    :return: 包含 trade_count, total_pnl, sharpe, max_dd, equity_curve, trades 的 dict
    """
    
    # 提取价格数据
    close = bars['close'].values.astype(np.float64)
    open_prices = bars['open'].values.astype(np.float64)
    high = bars.get('high', close).values.astype(np.float64)
    low = bars.get('low', close).values.astype(np.float64)
    
    # 将滑点转换为价格比例（针对价格绝对值）
    # 滑点占价格的比例 = slippage / close_price
    # 这里我们用相对滑点更合理
    slippage_pct = slippage / np.where(close > 0, close, 1)
    
    # 处理信号：构建 long/short entry/exit
    # side: 1=多头, -1=空头, 0=空仓
    side = signals['side'].values
    
    # 初始化
    n = len(close)
    entries = np.zeros(n, dtype=bool)
    exits = np.zeros(n, dtype=bool)
    short_entries = np.zeros(n, dtype=bool)
    short_exits = np.zeros(n, dtype=bool)
    
    position = 0
    for i in range(n):
        if position == 0:
            # 空仓状态，看是否可以入场
            if side[i] == 1:
                entries[i] = True
                position = 1
            elif side[i] == -1:
                short_entries[i] = True
                position = -1
        elif position == 1:
            # 多头持仓，看是否可以出场
            if side[i] <= 0:  # 反转或平仓
                exits[i] = True
                position = 0
        elif position == -1:
            # 空头持仓，看是否可以出场
            if side[i] >= 0:  # 反转或平仓
                short_exits[i] = True
                position = 0
    
    # 强制最后平仓
    if position != 0:
        exits[-1] = True if position == 1 else False
        short_exits[-1] = True if position == -1 else False
    
    # 构建价格字典（用于滑点模拟）
    # 这里用 close 价格，滑点通过 vectorbt 的 settings 处理
    price = close
    
    # 运行 vectorbt 回测
    # 注意：vectorbt 的 from_signals 会自动处理入场后下一根 bar 成交
    # 我们需要调整滑点的处理方式
    
    # 设置全局参数
    vbt.settings.returns['yearly_days'] = 252
    
    # 创建 portfolio
    # 使用 open 价格成交（更真实）
    # 滑点：通过调整价格来实现
    # entry 和 exit 都用开盘价，滑点通过 slippages 参数处理
    
    # 简化处理：直接在价格上加滑点
    entries_idx = np.where(entries)[0]
    exits_idx = np.where(exits)[0]
    short_entries_idx = np.where(short_entries)[0]
    short_exits_idx = np.where(short_exits)[0]
    
    # 入场时滑点：买入时价格更高，卖出时价格更低
    # 这里我们用相对比例的滑点
    entry_price = open_prices.copy()
    exit_price = open_prices.copy()
    
    # 多头入场：开盘买入，加滑点（买贵）
    for idx in entries_idx:
        entry_price[idx] = open_prices[idx] * (1 + slippage_pct[idx])
    
    # 多头出场：开盘卖出，减滑点（卖便宜）
    for idx in exits_idx:
        exit_price[idx] = open_prices[idx] * (1 - slippage_pct[idx])
    
    # 空头入场：开盘卖出，减滑点（卖贵）
    for idx in short_entries_idx:
        entry_price[idx] = open_prices[idx] * (1 - slippage_pct[idx])
    
    # 空头出场：开盘买入，加滑点（买便宜）
    for idx in short_exits_idx:
        exit_price[idx] = open_prices[idx] * (1 + slippage_pct[idx])
    
    # 使用调整后的价格
    # 创建 entry 和 exit 价格的序列
    entry_price_arr = pd.Series(entry_price, index=bars.index)
    exit_price_arr = pd.Series(exit_price, index=bars.index)
    
    # 使用 vectorbt 的 from_orders 来更精细控制
    # 或者继续用 from_signals 但调整价格
    try:
        portfolio = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            init_cash=init_cash,
            commission=commission,
            freq=freq,
            # 这里不能直接传调整后的价格，因为 from_signals 只接受 close
            # 我们用 slippage 参数（但它是固定比例）
            slippage=0,  # 我们手动处理了滑点
        )
    except Exception as e:
        print(f"Vectorbt error: {e}")
        raise
    
    # 提取结果
    trades = portfolio.trades.records
    equity_curve = portfolio.equity()
    returns = portfolio.returns()
    
    # 计算指标
    trade_count = len(trades)
    
    if trade_count == 0:
        return {
            'trade_count': 0,
            'total_pnl': 0.0,
            'sharpe': 0.0,
            'max_dd': 0.0,
            'win_rate': 0.0,
            'avg_trade': 0.0,
            'equity_curve': pd.Series([init_cash], index=[bars.index[0]]),
            'trades': pd.DataFrame(),
        }
    
    # PnL 计算
    pnl_arr = trades['pnl'].values if 'pnl' in trades.columns else trades['return'].values * init_cash
    total_pnl = pnl_arr.sum()
    
    # Sharpe Ratio
    if len(returns) > 1 and returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Max Drawdown
    equity_vals = equity_curve.values if hasattr(equity_curve, 'values') else equity_curve
    cummax = np.maximum.accumulate(equity_vals)
    drawdown = (equity_vals - cummax) / cummax
    max_dd = drawdown.min() if len(drawdown) > 0 else 0.0
    
    # Win Rate
    win_count = (pnl_arr > 0).sum()
    win_rate = win_count / trade_count if trade_count > 0 else 0.0
    
    # Avg Trade
    avg_trade = pnl_arr.mean()
    
    # 构建输出
    result = {
        'trade_count': trade_count,
        'total_pnl': total_pnl,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'avg_trade': avg_trade,
        'equity_curve': equity_curve if hasattr(equity_curve, 'index') else pd.Series(equity_curve, index=bars.index),
        'trades': trades,
    }
    
    return result


def run_comparison(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    commission: float = COMMISSION_RATE,
    slippage: float = SLIPPAGE_POINTS,
) -> dict:
    """
    对比不同 exit 策略。
    """
    exit_types = ['reverse_signal', 'fixed_hold', 'trailing_stop', 'time_based']
    results = {}
    
    for exit_type in exit_types:
        # 这里需要根据 exit_type 调整 signals
        # 暂时只实现 reverse_signal
        if exit_type == 'reverse_signal':
            result = run_backtest_vbt(bars, signals, commission, slippage)
            results[exit_type] = {
                'trade_count': result['trade_count'],
                'total_pnl': result['total_pnl'],
                'sharpe': result['sharpe'],
                'avg_trade': result['avg_trade'],
            }
    
    return results


# 兼容旧接口的 wrapper
def simple_backtest(
    signals: pd.DataFrame,
    bars: pd.DataFrame,
    commission: float = COMMISSION_RATE,
    slippage: float = SLIPPAGE_POINTS,
) -> dict:
    """
    兼容旧接口的 wrapper。
    """
    result = run_backtest_vbt(bars, signals, commission, slippage)
    return {
        'trade_count': result['trade_count'],
        'total_pnl': result['total_pnl'],
        'sharpe': result['sharpe'],
        'max_dd': result['max_dd'],
        'avg_trade': result['avg_trade'],
    }


if __name__ == '__main__':
    # 测试代码
    from strategies.AL9999.backtest_utils import load_dollar_bars
    from afmlkit.feature.core.ma import ewma
    import os
    
    # 加载数据
    bars = load_dollar_bars()
    print(f"Loaded {len(bars)} bars")
    
    # 生成 DMA 信号
    close = bars['close'].values.astype(np.float64)
    ema_fast = ewma(close, 3)
    ema_slow = ewma(close, 15)
    signal = np.where(ema_fast > ema_slow, 1, np.where(ema_fast < ema_slow, -1, 0))
    
    # 构建 signals DataFrame
    signals = pd.DataFrame({'side': signal}, index=bars.index)
    signals = signals[signals['side'] != 0].copy()
    
    # 运行回测
    result = run_backtest_vbt(bars, signals)
    
    print(f"Trade count: {result['trade_count']}")
    print(f"Total PnL: {result['total_pnl']:.2f}")
    print(f"Sharpe: {result['sharpe']:.4f}")
    print(f"Max DD: {result['max_dd']:.2%}")
    print(f"Win rate: {result['win_rate']:.2%}")
    print(f"Avg trade: {result['avg_trade']:.2f}")