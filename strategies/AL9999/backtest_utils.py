"""
backtest_utils.py - AL9999 单仓位回测共享工具
"""

import os

import numpy as np
import pandas as pd

from strategies.AL9999.config import BARS_DIR, TARGET_DAILY_BARS, COMMISSION_RATE, SLIPPAGE_POINTS


def load_dollar_bars() -> pd.DataFrame:
    """
    Load AL9999 dollar bars used by the research backtest.
    """
    return pd.read_parquet(os.path.join(BARS_DIR, f"dollar_bars_target{TARGET_DAILY_BARS}.parquet"))


def rolling_backtest(
    signals: pd.DataFrame,
    bars: pd.DataFrame,
    use_meta_filter: bool = True,
    side_mode: str = "both",
    guard_enabled: bool = False,
    min_hold_bars: int = 0,
    cooldown_bars: int = 0,
    reverse_confirmation_delta: float = 0.0,
    entry_threshold: float = 0.5,
    reversed_meta_filter: bool = False,
    position_multiplier_col: str = None,
) -> pd.DataFrame:
    """
    Run the single-position rolling backtest with optional meta filter.

    :param reversed_meta_filter: If True, meta_pred=0 signals are REVERSED (direction flipped)
        instead of being filtered. This implements the "trade the meta=0 signals in reverse"
        strategy discovered in the PnL analysis.
    :param position_multiplier_col: Column name in signals DataFrame containing position
        multiplier values (float). Multiplier is applied at entry. Default None (multiplier=1.0).
    """
    # 保存原始信号（用于 reversed_meta_filter 判断）
    if reversed_meta_filter and "meta_pred" in signals.columns:
        # 对 meta=0 的信号方向取反
        signals = signals.copy()
        # 记录原始方向用于分析
        signals["_orig_side"] = signals["side"]
        meta0_mask = signals["meta_pred"] == 0
        signals.loc[meta0_mask, "side"] = -signals.loc[meta0_mask, "side"]
        # 移除过滤，让所有信号参与（因为已被反向处理）
        use_meta_filter = False

    if use_meta_filter:
        signals = signals[signals["meta_pred"] == 1].copy()

    if side_mode == "long_only":
        signals = signals[signals["side"] == 1].copy()
    elif side_mode == "short_only":
        signals = signals[signals["side"] == -1].copy()

    signals = signals.sort_index()

    position = 0
    entry_idx = None
    entry_price = None
    position_touch_idx = None
    entry_multiplier = 1.0
    last_exit_idx = None
    trades = []

    def _norm_touch_idx(raw_touch_idx):
        if pd.isna(raw_touch_idx):
            return None
        idx_val = int(raw_touch_idx)
        if idx_val < 0 or idx_val >= len(bars):
            return None
        return idx_val

    open_col = "open" if "open" in bars.columns else "close"

    for event_time, signal in signals.iterrows():
        event_idx = bars.index.get_loc(event_time) if event_time in bars.index else None
        if event_idx is None:
            continue

        signal_side = signal["side"]
        signal_touch_idx = _norm_touch_idx(signal["touch_idx"])

        fill_idx = event_idx + 1
        if fill_idx >= len(bars):
            continue

        fill_price = bars.iloc[fill_idx][open_col]
        fill_time = bars.index[fill_idx]

        if position != 0 and position_touch_idx is not None and position_touch_idx < event_idx:
            closed_idx = position_touch_idx
            exit_price = bars.iloc[closed_idx]["close"]
            pnl = (exit_price - entry_price) * position * entry_multiplier
            ret = np.log(exit_price / entry_price) * entry_multiplier

            trades.append(
                {
                    "entry_time": bars.index[entry_idx],
                    "exit_time": bars.index[closed_idx],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "side": position,
                    "pnl": pnl,
                    "ret": ret,
                    "exit_reason": "TBM_trigger",
                    "multiplier": entry_multiplier,
                }
            )

            position = 0
            entry_multiplier = 1.0
            entry_idx = None
            entry_price = None
            position_touch_idx = None
            last_exit_idx = closed_idx

        if position == 0:
            if guard_enabled and cooldown_bars > 0 and last_exit_idx is not None:
                bars_since_exit = fill_idx - last_exit_idx
                if bars_since_exit <= cooldown_bars:
                    continue

            position = signal_side
            entry_idx = fill_idx
            entry_price = fill_price
            position_touch_idx = signal_touch_idx
            # 读取仓位乘数（默认 1.0）
            if position_multiplier_col and position_multiplier_col in signal.index:
                entry_multiplier = float(signal[position_multiplier_col])
            else:
                entry_multiplier = 1.0
        elif position != signal_side:
            if guard_enabled:
                if entry_idx is not None and min_hold_bars > 0:
                    held_bars = fill_idx - entry_idx
                    if held_bars < min_hold_bars:
                        continue

                if reverse_confirmation_delta > 0:
                    meta_prob = signal.get("meta_prob", signal.get("y_prob", np.nan))
                    if pd.isna(meta_prob) or float(meta_prob) < entry_threshold + reverse_confirmation_delta:
                        continue

            exit_price = fill_price
            pnl = (exit_price - entry_price) * position * entry_multiplier
            ret = np.log(exit_price / entry_price) * entry_multiplier

            trades.append(
                {
                    "entry_time": bars.index[entry_idx],
                    "exit_time": fill_time,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "side": position,
                    "pnl": pnl,
                    "ret": ret,
                    "exit_reason": "reverse_signal",
                    "multiplier": entry_multiplier,
                }
            )

            position = signal_side
            entry_idx = fill_idx
            entry_price = fill_price
            position_touch_idx = signal_touch_idx
            last_exit_idx = fill_idx
            if position_multiplier_col and position_multiplier_col in signal.index:
                entry_multiplier = float(signal[position_multiplier_col])
            else:
                entry_multiplier = 1.0

    if position != 0 and entry_idx is not None:
        touch_idx = position_touch_idx
        if touch_idx is not None and touch_idx > entry_idx and touch_idx < len(bars):
            exit_price = bars.iloc[touch_idx]["close"]
        else:
            exit_price = bars.iloc[-1]["close"]
            touch_idx = len(bars) - 1

        pnl = (exit_price - entry_price) * position * entry_multiplier
        ret = np.log(exit_price / entry_price) * entry_multiplier

        trades.append(
            {
                "entry_time": bars.index[entry_idx],
                "exit_time": bars.index[touch_idx],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "side": position,
                "pnl": pnl,
                "ret": ret,
                "exit_reason": "final_close",
                "multiplier": entry_multiplier,
            }
        )
        last_exit_idx = touch_idx

    return pd.DataFrame(trades)


def calculate_performance(trades_df: pd.DataFrame, annualization_factor: int = 1000) -> dict:
    """
    Calculate cost-adjusted performance statistics for trade records.
    """
    if len(trades_df) == 0:
        return {}

    trades_df = trades_df.copy()
    trades_df["net_pnl"] = (
        trades_df["pnl"]
        - (trades_df["entry_price"] + trades_df["exit_price"]) * COMMISSION_RATE
        - SLIPPAGE_POINTS * 2
    )
    trades_df["net_ret"] = trades_df["net_pnl"] / trades_df["entry_price"]

    total_pnl = trades_df["net_pnl"].sum()
    total_ret = trades_df["net_ret"].sum()

    win_trades = trades_df[trades_df["net_pnl"] > 0]
    lose_trades = trades_df[trades_df["net_pnl"] < 0]

    n_trades = len(trades_df)
    win_rate = len(win_trades) / n_trades if n_trades > 0 else 0.0
    avg_win = win_trades["net_pnl"].mean() if len(win_trades) > 0 else 0.0
    avg_loss = lose_trades["net_pnl"].mean() if len(lose_trades) > 0 else 0.0
    profit_factor = (
        abs(win_trades["net_pnl"].sum() / lose_trades["net_pnl"].sum())
        if len(lose_trades) > 0 and lose_trades["net_pnl"].sum() != 0
        else np.inf
    )

    returns = trades_df["net_ret"]
    sharpe = returns.mean() / returns.std() * np.sqrt(annualization_factor) if returns.std() > 0 else 0.0

    cum_pnl = trades_df["net_pnl"].cumsum()
    drawdown = cum_pnl - cum_pnl.cummax()

    return {
        "n_trades": n_trades,
        "total_pnl": total_pnl,
        "total_ret": total_ret,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "sharpe": sharpe,
        "mdd": drawdown.min(),
        "trades_df": trades_df,
    }
