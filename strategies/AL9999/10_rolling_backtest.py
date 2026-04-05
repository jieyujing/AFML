"""
10_rolling_backtest.py - AL9999 真实单仓位滚动回测

注意：为避免口径分叉，本仓库以 `10_combined_backtest.py` 作为唯一权威回测入口。
该脚本保留用于历史对照/调试，不再作为正式验收与报告来源。
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.AL9999.backtest_utils import calculate_performance, load_dollar_bars, rolling_backtest
from strategies.AL9999.config import FEATURES_DIR, FIGURES_DIR, META_MODEL_CONFIG


def load_data():
    """加载所需数据。"""
    tbm = pd.read_parquet(os.path.join(FEATURES_DIR, "tbm_results.parquet"))
    bars = load_dollar_bars()
    return tbm, bars


def get_meta_predictions(tbm: pd.DataFrame) -> pd.DataFrame:
    """拼接 OOF + Holdout 预测，避免全样本预测泄露。"""
    models_dir = FEATURES_DIR.replace("features", "models")
    oof_path = os.path.join(models_dir, "meta_oof_signals.parquet")
    holdout_path = os.path.join(models_dir, "meta_holdout_signals.parquet")
    threshold = META_MODEL_CONFIG.get("precision_threshold", 0.5)

    if not os.path.exists(oof_path):
        raise FileNotFoundError(f"缺少 OOF 文件: {oof_path}。请先运行 07_meta_model.py。")
    if not os.path.exists(holdout_path):
        raise FileNotFoundError(f"缺少 Holdout 文件: {holdout_path}。请先运行 07_meta_model.py。")

    oof_df = pd.read_parquet(oof_path).copy()
    holdout_df = pd.read_parquet(holdout_path).copy()
    oof_df["meta_pred"] = (oof_df["y_prob"] >= threshold).astype(int)
    if "meta_pred" not in holdout_df.columns:
        holdout_df["meta_pred"] = (holdout_df["y_prob"] >= threshold).astype(int)

    stitched = pd.concat(
        [
            oof_df[["meta_pred", "y_prob"]].rename(columns={"y_prob": "meta_proba"}),
            holdout_df[["meta_pred", "y_prob"]].rename(columns={"y_prob": "meta_proba"}),
        ],
        axis=0,
    )
    stitched = stitched[~stitched.index.duplicated(keep="last")].sort_index()

    common_idx = tbm.index.intersection(stitched.index)
    result = tbm.loc[common_idx].copy()
    result[["meta_pred", "meta_proba"]] = stitched.loc[common_idx, ["meta_pred", "meta_proba"]]
    return result


def main():
    """主函数。"""
    print("=" * 70)
    print("  AL9999 单仓位滚动回测（修正版）")
    print("=" * 70)

    print("\n[Step 1] 加载数据...")
    tbm, bars = load_data()

    print("\n[Step 2] 获取 Meta Model 预测...")
    signals = get_meta_predictions(tbm)
    print(f"总信号数: {len(signals)}")

    print("\n[Step 3] 运行单仓位滚动回测...")

    print("\n--- Primary Model (无过滤) ---")
    primary_trades = rolling_backtest(signals, bars, use_meta_filter=False)
    primary_perf = calculate_performance(primary_trades)
    primary_trades = primary_perf["trades_df"]
    print(f"交易次数: {primary_perf['n_trades']}")
    print(f"总收益: {primary_perf['total_pnl']:.2f} 点")
    print(f"胜率: {primary_perf['win_rate']*100:.1f}%")
    print(f"盈亏比: {primary_perf['profit_factor']:.2f}")
    print(f"年化 Sharpe: {primary_perf['sharpe']:.2f}")
    print(f"最大回撤: {primary_perf['mdd']:.2f} 点")

    print("\n--- Combined Strategy (Meta 过滤) ---")
    filtered_count = int((signals["meta_pred"] == 1).sum())
    print(f"Meta 过滤后信号数: {filtered_count}")
    combined_trades = rolling_backtest(signals, bars, use_meta_filter=True)
    combined_perf = calculate_performance(combined_trades)
    combined_trades = combined_perf["trades_df"]
    print(f"交易次数: {combined_perf['n_trades']}")
    print(f"总收益: {combined_perf['total_pnl']:.2f} 点")
    print(f"胜率: {combined_perf['win_rate']*100:.1f}%")
    print(f"盈亏比: {combined_perf['profit_factor']:.2f}")
    print(f"年化 Sharpe: {combined_perf['sharpe']:.2f}")
    print(f"最大回撤: {combined_perf['mdd']:.2f} 点")

    print("\n" + "=" * 70)
    print("  对比原回测逻辑")
    print("=" * 70)
    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│                    原回测 vs 修正后回测                              │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  Primary Sharpe:     0.32 (原) → {primary_perf['sharpe']:.2f} (修正)              │")
    print(f"│  Combined Sharpe:   21.94 (原) → {combined_perf['sharpe']:.2f} (修正)              │")
    print(f"│  Primary 胜率:      48.3% (原) → {primary_perf['win_rate']*100:.1f}% (修正)              │")
    print(f"│  Combined 胜率:     75.3% (原) → {combined_perf['win_rate']*100:.1f}% (修正)              │")
    print("└─────────────────────────────────────────────────────────────────────┘")

    print("\n[Step 4] 保存结果...")
    primary_trades.to_parquet(os.path.join(FEATURES_DIR, "rolling_primary_trades.parquet"))
    combined_trades.to_parquet(os.path.join(FEATURES_DIR, "rolling_combined_trades.parquet"))
    print("✅ 已保存滚动回测结果")

    print("\n[Step 5] 生成对比图...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax1 = axes[0]
    if len(primary_trades) > 0:
        ax1.plot(
            primary_trades["exit_time"],
            primary_trades["net_pnl"].cumsum(),
            label="Primary Model",
            alpha=0.7,
            color="purple",
            lw=1.5,
        )
    if len(combined_trades) > 0:
        ax1.plot(
            combined_trades["exit_time"],
            combined_trades["net_pnl"].cumsum(),
            label="Combined Strategy",
            color="orange",
            lw=2,
        )
    ax1.set_title("Rolling Backtest: Cumulative PnL (Single Position)", fontsize=14)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Net PnL (Points)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    categories = ["Sharpe", "Win Rate (%)", "Profit Factor"]
    primary_vals = [
        primary_perf["sharpe"],
        primary_perf["win_rate"] * 100,
        min(primary_perf["profit_factor"], 5),
    ]
    combined_vals = [
        combined_perf["sharpe"],
        combined_perf["win_rate"] * 100,
        min(combined_perf["profit_factor"], 5),
    ]
    x = np.arange(len(categories))
    width = 0.35
    ax2.bar(x - width / 2, primary_vals, width, label="Primary", color="purple", alpha=0.7)
    ax2.bar(x + width / 2, combined_vals, width, label="Combined", color="orange")
    ax2.set_ylabel("Value")
    ax2.set_title("Performance Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "10_rolling_backtest.png"), dpi=150)
    print(f"✅ 图表已保存: {FIGURES_DIR}/10_rolling_backtest.png")
    plt.close()

    print("\n" + "=" * 70)
    print("  回测完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
