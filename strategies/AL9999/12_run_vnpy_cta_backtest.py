"""
12_run_vnpy_cta_backtest.py - 使用 vn.py CTA BacktestingEngine 回测 AL9999 策略

说明：
- 为了和研究口径可比，本脚本将预计算的 Dollar Bars 写入 sqlite，
  然后使用 vnpy_ctastrategy.backtesting.BacktestingEngine 做回测。
- 策略以 `input_bar_mode="dollar"` 直接消费这些 Dollar Bars，
  避免分钟线重建 Dollar Bars 带来的巨大回测开销。
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.AL9999.config import (
    BARS_DIR,
    COMMISSION_RATE,
    FEATURES_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    SLIPPAGE_POINTS,
)
from strategies.AL9999.vnpy_strategy import Al9999CtaStrategy

warnings.filterwarnings("ignore", category=FutureWarning)


def normalize_timestamp(value) -> pd.Timestamp:
    """
    Normalize vn.py timestamps to naive pandas timestamps.
    """
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        return timestamp.tz_localize(None)
    return timestamp


def prepare_vnpy_database(db_path: Path) -> tuple[int, datetime, datetime]:
    """
    Save precomputed Dollar Bars into vn.py sqlite database.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from vnpy.trader.setting import SETTINGS

    SETTINGS["database.name"] = "sqlite"
    SETTINGS["database.database"] = str(db_path)

    from vnpy.trader import database as database_module
    database_module.database = None

    from vnpy.trader.constant import Exchange, Interval
    from vnpy.trader.object import BarData
    from vnpy.trader.database import get_database

    bars_df = pd.read_parquet(os.path.join(BARS_DIR, "dollar_bars_target4.parquet")).sort_index()

    if db_path.exists():
        db_path.unlink()

    db = get_database()

    bars = []
    for ts, row in bars_df.iterrows():
        bars.append(
            BarData(
                gateway_name="DB",
                symbol="AL9999",
                exchange=Exchange.SHFE,
                datetime=pd.Timestamp(ts).to_pydatetime(),
                interval=Interval.MINUTE,
                volume=float(row["volume"]),
                turnover=float(row["dollar_volume"]),
                open_interest=float(row["open_interest"]),
                open_price=float(row["open"]),
                high_price=float(row["high"]),
                low_price=float(row["low"]),
                close_price=float(row["close"]),
            )
        )

    db.save_bar_data(bars)
    if hasattr(db, "db") and not db.db.is_closed():
        db.db.close()
    return len(bars), bars[0].datetime, bars[-1].datetime


def build_research_parity_engine():
    """
    Build a BacktestingEngine subclass that can fill replay orders on the signal bar.
    """
    from vnpy_ctastrategy.backtesting import BacktestingEngine
    from vnpy.trader.object import TradeData
    from vnpy.trader.constant import Status

    class ResearchParityBacktestingEngine(BacktestingEngine):
        def new_bar(self, bar) -> None:
            self.bar = bar
            self.datetime = bar.datetime

            # First fill outstanding orders from prior bars using native semantics.
            self.cross_limit_order()
            self.cross_stop_order()

            # Let strategy react to the current research event bar.
            self.strategy.on_bar(bar)

            # In research replay mode, orders submitted on this bar should trade on this bar.
            self.cross_limit_order_research_close()
            self.update_daily_close(bar.close_price)

        def cross_limit_order_research_close(self) -> None:
            """
            Fill same-bar replay orders at the submitted price to mirror research logic.
            """
            for order in list(self.active_limit_orders.values()):
                if order.status != Status.SUBMITTING:
                    continue

                order.status = Status.NOTTRADED
                self.strategy.on_order(order)

                order.traded = order.volume
                order.status = Status.ALLTRADED
                self.strategy.on_order(order)

                if order.vt_orderid in self.active_limit_orders:
                    self.active_limit_orders.pop(order.vt_orderid)

                self.trade_count += 1
                pos_change = order.volume if order.direction.name == "LONG" else -order.volume

                trade = TradeData(
                    symbol=order.symbol,
                    exchange=order.exchange,
                    orderid=order.orderid,
                    tradeid=str(self.trade_count),
                    direction=order.direction,
                    offset=order.offset,
                    price=order.price,
                    volume=order.volume,
                    datetime=self.datetime,
                    gateway_name=self.gateway_name,
                )

                self.strategy.pos += pos_change
                self.strategy.on_trade(trade)
                self.trades[trade.vt_tradeid] = trade

    return ResearchParityBacktestingEngine


def extract_round_trip_trades(engine) -> pd.DataFrame:
    """
    Reconstruct round-trip trades from vn.py trade fills.
    """
    fills = sorted(engine.get_all_trades(), key=lambda trade: trade.datetime)
    strategy_actions = [item for item in getattr(engine.strategy, "action_log", []) if item.get("action") in {"enter", "exit"}]
    exit_reasons = [item.get("reason", "") for item in strategy_actions if item.get("action") == "exit"]

    trades = []
    open_fill = None
    exit_reason_iter = iter(exit_reasons)

    for fill in fills:
        offset_name = getattr(fill.offset, "name", str(fill.offset))
        direction_name = getattr(fill.direction, "name", str(fill.direction))

        if offset_name == "OPEN":
            open_fill = fill
            continue

        if open_fill is None:
            continue

        side = 1 if getattr(open_fill.direction, "name", str(open_fill.direction)) == "LONG" else -1
        entry_price = float(open_fill.price)
        exit_price = float(fill.price)
        pnl = (exit_price - entry_price) * side
        net_pnl = pnl - (entry_price + exit_price) * COMMISSION_RATE - SLIPPAGE_POINTS * 2

        trades.append(
            {
                "entry_time": normalize_timestamp(open_fill.datetime),
                "exit_time": normalize_timestamp(fill.datetime),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "side": side,
                "pnl": pnl,
                "ret": (exit_price / entry_price - 1.0) * side,
                "net_pnl": net_pnl,
                "net_ret": net_pnl / entry_price,
                "exit_reason": next(exit_reason_iter, ""),
                "entry_direction": direction_name,
                "exit_offset": offset_name,
            }
        )
        open_fill = None

    strategy = getattr(engine, "strategy", None)
    if open_fill is not None and getattr(strategy, "use_replay_signals", False):
        replay_table = getattr(strategy, "replay_signal_table", pd.DataFrame())
        filtered_replay = replay_table[replay_table["meta_pred"] == 1] if not replay_table.empty else pd.DataFrame()
        if not filtered_replay.empty:
            final_row = filtered_replay.iloc[-1]
            side = 1 if getattr(open_fill.direction, "name", str(open_fill.direction)) == "LONG" else -1
            entry_price = float(open_fill.price)
            exit_price = float(final_row["exit_price"])
            pnl = (exit_price - entry_price) * side
            net_pnl = pnl - (entry_price + exit_price) * COMMISSION_RATE - SLIPPAGE_POINTS * 2
            trades.append(
                {
                    "entry_time": normalize_timestamp(open_fill.datetime),
                    "exit_time": normalize_timestamp(final_row["exit_ts"]),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "side": side,
                    "pnl": pnl,
                    "ret": (exit_price / entry_price - 1.0) * side,
                    "net_pnl": net_pnl,
                    "net_ret": net_pnl / entry_price,
                    "exit_reason": "final_close",
                    "entry_direction": getattr(open_fill.direction, "name", str(open_fill.direction)),
                    "exit_offset": "SYNTHETIC_FINAL_CLOSE",
                }
            )

    return pd.DataFrame(trades)


def run_backtest(db_path: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Run vn.py backtest on precomputed Dollar Bars.
    """
    from vnpy.trader.setting import SETTINGS

    SETTINGS["database.name"] = "sqlite"
    SETTINGS["database.database"] = str(db_path)

    from vnpy.trader import database as database_module
    if getattr(database_module, "database", None) is not None:
        current_db = database_module.database
        if hasattr(current_db, "db") and not current_db.db.is_closed():
            current_db.db.close()
    database_module.database = None

    from vnpy.trader.constant import Interval
    ResearchParityBacktestingEngine = build_research_parity_engine()

    bars_df = pd.read_parquet(os.path.join(BARS_DIR, "dollar_bars_target4.parquet")).sort_index()

    engine = ResearchParityBacktestingEngine()
    engine.set_parameters(
        vt_symbol="AL9999.SHFE",
        interval=Interval.MINUTE,
        start=pd.Timestamp(bars_df.index.min()).to_pydatetime(),
        end=pd.Timestamp(bars_df.index.max()).to_pydatetime(),
        rate=COMMISSION_RATE,
        slippage=SLIPPAGE_POINTS,
        size=1,
        pricetick=5,
        capital=1_000_000,
    )
    engine.add_strategy(
        Al9999CtaStrategy,
        {
            "research_symbol": "AL9999",
            "model_path": os.path.join(MODELS_DIR, "meta_model.pkl"),
            "fixed_size": 1,
            "emit_orders": True,
            "input_bar_mode": "dollar",
            "symbol_mapping": {"AL9999": "AL9999.SHFE"},
            "manual_symbol_overrides": {"AL9999": "AL9999.SHFE"},
            "state_path": os.path.join(FEATURES_DIR, "vnpy_backtest_state.json"),
            "replay_tbm_path": os.path.join(FEATURES_DIR, "tbm_results.parquet"),
            "replay_features_path": os.path.join(FEATURES_DIR, "events_features.parquet"),
        },
    )
    engine.load_data()
    engine.run_backtesting()
    result_df = engine.calculate_result()
    stats = engine.calculate_statistics(result_df, output=False)
    trade_df = extract_round_trip_trades(engine)
    return result_df, stats, trade_df


def compare_with_research(vnpy_stats: dict, vnpy_trades: pd.DataFrame) -> dict:
    """
    Compare vn.py backtest output against research artifacts.
    """
    research_trades = pd.read_parquet(os.path.join(FEATURES_DIR, "rolling_combined_trades.parquet"))
    research_stats = pd.read_parquet(os.path.join(FEATURES_DIR, "backtest_stats.parquet"))

    research_total_net_pnl = float(research_trades["net_pnl"].sum())
    research_trade_count = int(len(research_trades))
    research_win_rate = float((research_trades["net_pnl"] > 0).mean())
    vnpy_total_net_pnl = float(vnpy_trades["net_pnl"].sum()) if not vnpy_trades.empty else 0.0
    vnpy_trade_count = int(len(vnpy_trades))
    vnpy_win_rate = float((vnpy_trades["net_pnl"] > 0).mean()) if not vnpy_trades.empty else 0.0

    matched_entries = 0
    matched_exits = 0
    matched_sides = 0
    if vnpy_trade_count and research_trade_count:
        overlap = min(vnpy_trade_count, research_trade_count)
        vnpy_head = vnpy_trades.head(overlap).reset_index(drop=True)
        research_head = research_trades.head(overlap).reset_index(drop=True)
        matched_entries = int((vnpy_head["entry_time"] == research_head["entry_time"]).sum())
        matched_exits = int((vnpy_head["exit_time"] == research_head["exit_time"]).sum())
        matched_sides = int((vnpy_head["side"] == research_head["side"]).sum())

    report = {
        "research_trade_count": research_trade_count,
        "research_total_net_pnl": research_total_net_pnl,
        "research_win_rate": research_win_rate,
        "vnpy_trade_count": vnpy_trade_count,
        "vnpy_total_net_pnl": vnpy_total_net_pnl,
        "vnpy_win_rate": vnpy_win_rate,
        "trade_count_delta": vnpy_trade_count - research_trade_count,
        "net_pnl_delta": vnpy_total_net_pnl - research_total_net_pnl,
        "matched_entry_timestamps": matched_entries,
        "matched_exit_timestamps": matched_exits,
        "matched_sides": matched_sides,
        "vnpy_statistics": {key: value for key, value in vnpy_stats.items() if key not in {"daily_returns", "return_drawdown_ratio"}},
        "research_reference_table": research_stats.to_dict(orient="records"),
    }
    return report


def build_equity_frame(trades: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Build cumulative pnl and drawdown series from round-trip trades.
    """
    frame = trades.copy()
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "cum_pnl", "drawdown", "series"])

    frame["timestamp"] = pd.to_datetime(frame["exit_time"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame["cum_pnl"] = frame["net_pnl"].cumsum()
    frame["drawdown"] = frame["cum_pnl"] - frame["cum_pnl"].cummax()
    frame["series"] = label
    return frame[["timestamp", "cum_pnl", "drawdown", "series"]]


def build_daily_pnl_frame(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate round-trip trades into a daily pnl series.
    """
    if trades.empty:
        return pd.DataFrame(columns=["date", "net_pnl"])

    frame = trades.copy()
    frame["date"] = pd.to_datetime(frame["exit_time"]).dt.normalize()
    daily = frame.groupby("date", as_index=False)["net_pnl"].sum()
    return daily


def build_difference_frame(vnpy_equity: pd.DataFrame, research_equity: pd.DataFrame) -> pd.DataFrame:
    """
    Build cumulative pnl difference series on the union of exit timestamps.
    """
    if vnpy_equity.empty and research_equity.empty:
        return pd.DataFrame(columns=["timestamp", "cum_pnl_diff"])

    combined = pd.DataFrame(
        {
            "timestamp": pd.Index(
                sorted(
                    set(pd.to_datetime(vnpy_equity.get("timestamp", pd.Series(dtype="datetime64[ns]"))))
                    | set(pd.to_datetime(research_equity.get("timestamp", pd.Series(dtype="datetime64[ns]"))))
                )
            )
        }
    )
    if not vnpy_equity.empty:
        combined = combined.merge(
            vnpy_equity[["timestamp", "cum_pnl"]].rename(columns={"cum_pnl": "vnpy_cum_pnl"}),
            on="timestamp",
            how="left",
        )
    else:
        combined["vnpy_cum_pnl"] = 0.0
    if not research_equity.empty:
        combined = combined.merge(
            research_equity[["timestamp", "cum_pnl"]].rename(columns={"cum_pnl": "research_cum_pnl"}),
            on="timestamp",
            how="left",
        )
    else:
        combined["research_cum_pnl"] = 0.0

    combined["vnpy_cum_pnl"] = combined["vnpy_cum_pnl"].ffill().fillna(0.0)
    combined["research_cum_pnl"] = combined["research_cum_pnl"].ffill().fillna(0.0)
    combined["cum_pnl_diff"] = combined["vnpy_cum_pnl"] - combined["research_cum_pnl"]
    return combined[["timestamp", "cum_pnl_diff"]]


def export_comparison_html(
    vnpy_trades: pd.DataFrame,
    research_trades: pd.DataFrame,
    comparison: dict,
    output_path: str | Path,
) -> Path:
    """
    Export a single-file interactive HTML comparing vn.py and research backtests.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vnpy_equity = build_equity_frame(vnpy_trades, "vn.py")
    research_equity = build_equity_frame(research_trades, "research")
    vnpy_daily = build_daily_pnl_frame(vnpy_trades)
    diff_frame = build_difference_frame(vnpy_equity, research_equity)
    overlap_exact = (
        not vnpy_equity.empty
        and len(vnpy_equity) == len(research_equity)
        and vnpy_equity["timestamp"].equals(research_equity["timestamp"])
        and vnpy_equity["cum_pnl"].equals(research_equity["cum_pnl"])
    )

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        row_heights=[0.34, 0.20, 0.16, 0.14, 0.16],
        subplot_titles=("Equity Curve", "Drawdown", "Daily Pnl", "Pnl Distribution", "Difference vs Research"),
    )

    fig.add_trace(
        go.Scatter(
            x=vnpy_equity["timestamp"],
            y=vnpy_equity["cum_pnl"],
            mode="lines+markers",
            name="vn.py Equity Curve",
            line={"color": "#1f77b4", "width": 2.5},
            marker={"size": 4, "symbol": "circle"},
            hovertemplate="Time=%{x}<br>vn.py cum pnl=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=research_equity["timestamp"],
            y=research_equity["cum_pnl"],
            mode="lines",
            name="Research Equity Curve",
            line={"color": "#d62728", "width": 6, "dash": "dash"},
            opacity=0.45,
            hovertemplate="Time=%{x}<br>research cum pnl=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=vnpy_equity["timestamp"],
            y=vnpy_equity["drawdown"],
            mode="lines",
            name="vn.py Drawdown",
            line={"color": "#6baed6", "width": 1.5},
            hovertemplate="Time=%{x}<br>vn.py drawdown=%{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=research_equity["timestamp"],
            y=research_equity["drawdown"],
            mode="lines",
            name="Research Drawdown",
            line={"color": "#fb6a4a", "width": 1.5},
            hovertemplate="Time=%{x}<br>research drawdown=%{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=vnpy_daily["date"],
            y=vnpy_daily["net_pnl"],
            name="vn.py Daily Pnl",
            marker={"color": "#4daf4a"},
            hovertemplate="Date=%{x}<br>vn.py daily pnl=%{y:.2f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=vnpy_daily["net_pnl"],
            nbinsx=60,
            name="vn.py Pnl Distribution",
            marker={"color": "#984ea3"},
            hovertemplate="PnL bin=%{x}<br>Count=%{y}<extra></extra>",
            showlegend=False,
        ),
        row=4,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=diff_frame["timestamp"],
            y=diff_frame["cum_pnl_diff"],
            mode="lines",
            name="Cumulative PnL Difference",
            line={"color": "#ff7f00", "width": 2},
            hovertemplate="Time=%{x}<br>vn.py - research=%{y:.2f}<extra></extra>",
            showlegend=False,
        ),
        row=5,
        col=1,
    )

    summary_html = (
        f"<b>Research trades:</b> {comparison.get('research_trade_count', 0)}"
        f" | <b>vn.py trades:</b> {comparison.get('vnpy_trade_count', 0)}"
        f" | <b>Research net pnl:</b> {comparison.get('research_total_net_pnl', 0.0):.2f}"
        f" | <b>vn.py net pnl:</b> {comparison.get('vnpy_total_net_pnl', 0.0):.2f}"
        f" | <b>Delta:</b> {comparison.get('net_pnl_delta', 0.0):.2f}"
        f" | <b>Matched entries:</b> {comparison.get('matched_entry_timestamps', 0)}"
        f" | <b>Matched exits:</b> {comparison.get('matched_exit_timestamps', 0)}"
        f" | <b>Panels:</b> Equity Curve / Drawdown / Daily Pnl / Pnl Distribution / Difference vs Research"
    )
    if overlap_exact:
        summary_html += " | <b>Overlap note:</b> Curves overlap exactly on matched exits."

    fig.update_layout(
        title={"text": "AL9999 vn.py vs Research Backtest", "x": 0.5},
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        height=1400,
        margin={"l": 60, "r": 40, "t": 120, "b": 60},
        annotations=[
            {
                "text": summary_html,
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": 1.15,
                "showarrow": False,
                "align": "left",
                "font": {"size": 12},
            }
        ],
    )
    fig.update_yaxes(title_text="Cumulative Net PnL", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)
    fig.update_yaxes(title_text="Daily Net PnL", row=3, col=1)
    fig.update_yaxes(title_text="Count", row=4, col=1)
    fig.update_yaxes(title_text="PnL Delta", row=5, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_xaxes(title_text="Net PnL", row=4, col=1)
    fig.update_xaxes(title_text="Time", row=5, col=1)

    fig.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    return output_path


def main() -> None:
    """
    Execute sqlite-backed vn.py backtest and save a comparison report.
    """
    print("=" * 72)
    print("  AL9999 vn.py CTA Backtest")
    print("=" * 72)

    db_path = Path(FEATURES_DIR) / "al9999_vnpy_backtest.db"
    bar_count, start, end = prepare_vnpy_database(db_path)
    print(f"\n[DB] sqlite={db_path}")
    print(f"[DB] bars_saved={bar_count}, start={start}, end={end}")

    result_df, vnpy_stats, vnpy_trades = run_backtest(db_path)
    print("\n[vn.py] statistics:")
    for key in [
        "start_date",
        "end_date",
        "total_days",
        "profit_days",
        "loss_days",
        "capital",
        "end_balance",
        "max_drawdown",
        "total_net_pnl",
        "daily_net_pnl",
        "total_trade_count",
        "win_ratio",
        "sharpe_ratio",
    ]:
        if key in vnpy_stats:
            print(f"  {key}: {vnpy_stats[key]}")

    comparison = compare_with_research(vnpy_stats, vnpy_trades)
    research_trades = pd.read_parquet(os.path.join(FEATURES_DIR, "rolling_combined_trades.parquet"))
    print("\n[Compare]")
    print(f"  research_trade_count={comparison['research_trade_count']}")
    print(f"  vnpy_trade_count={comparison['vnpy_trade_count']}")
    print(f"  research_total_net_pnl={comparison['research_total_net_pnl']}")
    print(f"  vnpy_total_net_pnl={comparison['vnpy_total_net_pnl']}")
    print(f"  net_pnl_delta={comparison['net_pnl_delta']}")
    print(f"  matched_entry_timestamps={comparison['matched_entry_timestamps']}")
    print(f"  matched_exit_timestamps={comparison['matched_exit_timestamps']}")
    print(f"  matched_sides={comparison['matched_sides']}")

    result_path = os.path.join(FEATURES_DIR, "vnpy_backtest_daily_result.parquet")
    trades_path = os.path.join(FEATURES_DIR, "vnpy_backtest_trades.parquet")
    report_path = os.path.join(FEATURES_DIR, "vnpy_backtest_comparison.json")
    html_path = os.path.join(FIGURES_DIR, "vnpy_vs_research_backtest.html")
    result_to_save = result_df.drop(columns=["trades"], errors="ignore").copy()
    result_to_save.to_parquet(result_path)
    vnpy_trades.to_parquet(trades_path)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False, default=str)
    export_comparison_html(vnpy_trades, research_trades, comparison, html_path)

    print(f"\n[Save] daily_result={result_path}")
    print(f"[Save] trades={trades_path}")
    print(f"[Save] comparison={report_path}")
    print(f"[Save] html={html_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
