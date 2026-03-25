"""
Bet Sizing Pipeline — AFML Meta-Labeling
=========================================

该脚本将 Primary Model 的趋势预测与 Meta-Model 的置信度概率结合，计算最终持仓权重。
符合 AFML 第 10 章方法论：
  1. Base Size (m_t)：基于 Meta-Model 概率 P(bin=1) 映射到 [-1, 1]。
  2. Concurrent Sizes：处理重叠事件，计算任意时刻的平均持仓权重。
  3. Discretization：离散化持仓，减少微小调仓。

Inputs:
  - primary_oof_signals.csv : 趋势预测 (side)
  - meta_oof_signals.csv    : 元概率 (P(bin=1))
  - cusum_sampled_bars.csv  : 事件退出时间 (t1)
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from afmlkit.label.bet_size import (
    get_signal_size,
    get_concurrent_sizes,
    discretize_size,
    get_size_change_signals
)

# ── Paths ────────────────────────────────────────────────────────────
PRIMARY_OOF_PATH = Path("outputs/models/primary_model/primary_oof_signals.csv")
META_OOF_PATH    = Path("outputs/models/meta_model/meta_oof_signals.csv")
LABELS_PATH      = Path("outputs/dollar_bars/cusum_sampled_bars.csv")
OUTPUT_DIR       = Path("outputs/bet_sizing")

def run_bet_sizing():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[BetSizing] 加载输入数据...")
    df_p = pd.read_csv(PRIMARY_OOF_PATH, parse_dates=["timestamp"])
    df_m = pd.read_csv(META_OOF_PATH, parse_dates=["timestamp"])
    df_l = pd.read_csv(LABELS_PATH, parse_dates=["timestamp"])

    # ── 合并 ────────────────────────────────────────────────────────
    df = pd.merge(df_p[["timestamp", "pred_high_recall"]], 
                  df_m[["timestamp", "meta_prob"]], 
                  on="timestamp", how="inner")
    df = pd.merge(df, df_l[["timestamp", "t1"]], 
                  on="timestamp", how="left")
    
    df["t1"] = pd.to_datetime(df["t1"])
    df = df.dropna(subset=["t1", "meta_prob"]).sort_values("timestamp")

    print(f"[BetSizing] 计算 Base Size (使用 Sigmoid/CDF 映射)...")
    # side = pred_high_recall (+1/-1)
    # prob = meta_prob (P(bin=1))
    df["base_size"] = get_signal_size(df["meta_prob"], df["pred_high_recall"])

    # ── 并行仓位聚合 ────────────────────────────────────────────────
    print(f"[BetSizing] 聚合并行信号（计算平均持仓）...")
    t_events = pd.DatetimeIndex(df["timestamp"])
    t_exits = df["t1"]
    
    # 获取整个时间轴上的活跃仓位
    # get_concurrent_sizes 会生成一个覆盖所有 event 时刻和 exit 时刻的时间序列
    active_sizes = get_concurrent_sizes(df["base_size"], t_events, t_exits)
    
    # ── 离散化 ──────────────────────────────────────────────────────
    print(f"[BetSizing] 离散化持仓 (Step=0.1)...")
    disc_sizes = discretize_size(active_sizes, step_size=0.1)
    
    # ── 导出结果 ────────────────────────────────────────────────────
    result_df = pd.DataFrame({
        "avg_size": active_sizes,
        "discretized_size": disc_sizes
    }, index=active_sizes.index)
    
    out_file = OUTPUT_DIR / "final_bet_sizes.csv"
    result_df.to_csv(out_file)
    print(f"✓ 仓位序列已保存 → {out_file}")

    # ── 可视化 ──────────────────────────────────────────────────────
    print(f"[可视化] 生成仓位分析图...")
    plot_results(df, result_df, OUTPUT_DIR)
    
    print(f"\n[完成] 共有 {len(df)} 个信号，平均持仓强度: {active_sizes.abs().mean():.4f}")

def plot_results(events_df, timeseries_df, output_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    # 1. 原始信号强度分布 (CDF 映射后的 Base Size)
    sns.histplot(events_df["base_size"], bins=30, kde=True, ax=ax1, color="#3498db")
    ax1.set_title("Base Bet Size Distribution (Primary Side * Meta Conf)", fontweight="bold")
    ax1.set_xlabel("Size")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. 时间序列上的持仓变化 (取最近一段，如 500 个数据点以便观察详情)
    plot_ts = timeseries_df.tail(800)
    ax2.plot(plot_ts.index, plot_ts["avg_size"], label="Active Avg Size (Continuous)", 
             color="#95a5a6", alpha=0.8, lw=1)
    ax2.step(plot_ts.index, plot_ts["discretized_size"], label="Final Discretized Size (step=0.1)", 
             where="post", color="#e74c3c", lw=2)
    
    ax2.set_title("Target Position Size Over Time (Recent 800 samples)", fontweight="bold")
    ax2.set_ylabel("Exposure")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plot_path = output_dir / "bet_sizing_analysis.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[图表] 仓位图表已生成 → {plot_path}")

if __name__ == "__main__":
    run_bet_sizing()
