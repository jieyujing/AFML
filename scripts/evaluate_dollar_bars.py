"""
Dollar Bars 全量数据评估脚本（流式处理）
========================================

使用 afmlkit 的 DynamicDollarBarKit 逻辑，对全量 25GB+ 交易数据
按月流式生成 5 种频率的动态 Dollar Bars，通过 JB 检验和自相关分析评估 IID 特性。

两阶段流式处理:
  阶段 1: 逐月扫描 → 计算每日 dollar volume
  阶段 2: 逐月加载 trades → 5 频率同时生成 bars（carry-over cum_dollar）

用法:
    uv run python examples/evaluate_dollar_bars.py
"""

import os
import gc
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from numba import njit
from numba.typed import List as NumbaList

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
DATA_PATH = "data/h5/BTCUSDT_PERPum_2301-2601.h5"
TARGET_FREQS = [4, 6, 10, 20, 50]          # 每日目标 bar 数
EWMA_SPAN = 20                              # EWMA 平滑窗口
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dollar_bars_evaluation.png")
BARS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "dollar_bars")


# ──────────────────────────────────────────────
# Numba 核心: 带 carry-over 的动态阈值 indexer
# ──────────────────────────────────────────────
@njit(nogil=True)
def _stream_dynamic_dollar_bar_indexer(
    prices: np.ndarray,
    volumes: np.ndarray,
    thresholds: np.ndarray,
    carry_over: float,
) -> tuple:
    """
    动态阈值 Dollar Bar indexer，支持跨分块 carry-over。

    :param prices: 当前月 tick 价格
    :param volumes: 当前月 tick 成交量
    :param thresholds: 当前月每 tick 对应的阈值
    :param carry_over: 上月残余累积金额
    :returns: (close_indices_list, new_carry_over)
    """
    n = len(prices)
    indices = NumbaList()

    cum_dollar = carry_over
    for i in range(n):
        cum_dollar += prices[i] * volumes[i]
        if cum_dollar >= thresholds[i]:
            indices.append(i)
            cum_dollar = cum_dollar - thresholds[i]

    return indices, cum_dollar


def _build_ohlcv_from_indices(
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    close_indices: np.ndarray,
) -> pd.DataFrame:
    """从 close_indices 构建 OHLCV DataFrame（纯 numpy 实现，避免导入 Numba 基类）。"""
    if len(close_indices) < 2:
        return pd.DataFrame()

    from afmlkit.bar.base import comp_bar_ohlcv

    ohlcv_tuple = comp_bar_ohlcv(prices, volumes, close_indices)

    close_ts = timestamps[close_indices[1:]]  # bar close 对应的时间戳

    ohlcv_df = pd.DataFrame({
        "timestamp": close_ts,
        "open": ohlcv_tuple[0],
        "high": ohlcv_tuple[1],
        "low": ohlcv_tuple[2],
        "close": ohlcv_tuple[3],
        "volume": ohlcv_tuple[4],
        "vwap": ohlcv_tuple[5],
        "trades": ohlcv_tuple[6],
        "median_trade_size": ohlcv_tuple[7],
    })

    ohlcv_df["timestamp"] = pd.to_datetime(ohlcv_df["timestamp"], unit="ns")
    ohlcv_df.set_index("timestamp", inplace=True)

    return ohlcv_df


# ──────────────────────────────────────────────
# 阶段 1: 扫描每日 dollar volume
# ──────────────────────────────────────────────
def scan_daily_dollar_volumes(data_path: str) -> tuple[pd.Series, list[str]]:
    """逐月扫描全部数据，仅计算每日 dollar volume 聚合。"""
    print("=" * 60)
    print("  阶段 1: 扫描每日 Dollar Volume")
    print("=" * 60)

    with pd.HDFStore(data_path, mode="r") as store:
        month_keys = sorted([k for k in store.keys() if k.startswith("/trades/")])

    print(f"📂 发现 {len(month_keys)} 个月度分区")

    daily_parts: list[pd.Series] = []
    t0 = time.perf_counter()

    for i, key in enumerate(month_keys):
        mt0 = time.perf_counter()
        month_df = pd.read_hdf(data_path, key=key, columns=["timestamp", "price", "amount"])

        dt_index = pd.to_datetime(month_df["timestamp"], unit="ns")
        dollar_vol = (month_df["price"].values * month_df["amount"].values).astype(np.float64)

        daily = pd.Series(dollar_vol, index=dt_index).resample("D").sum()
        daily_parts.append(daily)

        elapsed = time.perf_counter() - mt0
        print(f"   [{i+1:>2}/{len(month_keys)}] {key}: {len(month_df):>12,} 行  ({elapsed:.1f}s)")

        del month_df, dt_index, dollar_vol, daily
        gc.collect()

    total_elapsed = time.perf_counter() - t0
    daily_dollar_vol = pd.concat(daily_parts).sort_index()
    daily_dollar_vol = daily_dollar_vol.groupby(daily_dollar_vol.index).sum()  # 合并重复日期
    daily_dollar_vol = daily_dollar_vol.replace(0, np.nan).ffill()

    print(f"\n📊 阶段 1 完成: {len(daily_dollar_vol)} 个交易日, 耗时 {total_elapsed:.1f}s")

    del daily_parts
    gc.collect()

    return daily_dollar_vol, month_keys


def compute_ewma_thresholds(
    daily_dollar_vol: pd.Series,
    target_freqs: list[int],
    ewma_span: int,
) -> dict[int, pd.Series]:
    """对 daily dollar volume 做 EWMA，为每个频率计算日度阈值。"""
    daily_ewma = daily_dollar_vol.ewm(span=ewma_span, min_periods=1).mean()
    thresholds = {}
    for freq in target_freqs:
        thresholds[freq] = daily_ewma / freq
    return thresholds


# ──────────────────────────────────────────────
# 阶段 2: 逐月流式生成 bars
# ──────────────────────────────────────────────
def stream_dollar_bars(
    data_path: str,
    month_keys: list[str],
    daily_thresholds: dict[int, pd.Series],
    target_freqs: list[int],
) -> dict[int, pd.DataFrame]:
    """逐月加载 trades，为所有频率同时生成 dollar bars。"""
    print("\n" + "=" * 60)
    print("  阶段 2: 逐月流式生成 Dollar Bars")
    print("=" * 60)

    # 每频率的 carry-over 和 bars 列表
    carry_over = {freq: 0.0 for freq in target_freqs}
    all_bars: dict[int, list[pd.DataFrame]] = {freq: [] for freq in target_freqs}

    t0 = time.perf_counter()
    total_bars_count = {freq: 0 for freq in target_freqs}

    for i, key in enumerate(month_keys):
        mt0 = time.perf_counter()
        month_df = pd.read_hdf(data_path, key=key, columns=["timestamp", "price", "amount"])

        timestamps = month_df["timestamp"].values.astype(np.int64)
        prices = month_df["price"].values.astype(np.float64)
        volumes = month_df["amount"].values.astype(np.float64)

        # 为每个 tick 映射日期（向量化：仅做一次，所有频率共享）
        dt_index = pd.to_datetime(timestamps, unit="ns")
        tick_dates_int = dt_index.normalize().view(np.int64)  # 纳秒 int64，用于快速匹配

        for freq in target_freqs:
            # 向量化阈值映射：构建日期→阈值 lookup，用 searchsorted 映射
            daily_thrs = daily_thresholds[freq]
            thrs_dates_int = daily_thrs.index.view(np.int64)
            thrs_values = daily_thrs.values.astype(np.float64)

            # searchsorted: 找到每个 tick_date 在 thrs_dates_int 中的位置
            idx = np.searchsorted(thrs_dates_int, tick_dates_int, side="right") - 1
            idx = np.clip(idx, 0, len(thrs_values) - 1)
            tick_thresholds = thrs_values[idx]

            # 运行 Numba indexer (带 carry-over)
            indices, new_carry = _stream_dynamic_dollar_bar_indexer(
                prices, volumes, tick_thresholds, carry_over[freq]
            )
            carry_over[freq] = new_carry

            close_indices = np.array(indices, dtype=np.int64)

            if len(close_indices) >= 2:
                bars_df = _build_ohlcv_from_indices(timestamps, prices, volumes, close_indices)
                if len(bars_df) > 0:
                    all_bars[freq].append(bars_df)
                    total_bars_count[freq] += len(bars_df)

        elapsed = time.perf_counter() - mt0
        bars_info = " | ".join([f"f{f}={total_bars_count[f]:,}" for f in target_freqs])
        print(f"   [{i+1:>2}/{len(month_keys)}] {key}  ({elapsed:.1f}s)  [{bars_info}]")

        del month_df, timestamps, prices, volumes, dt_index, tick_dates_int
        gc.collect()

    total_elapsed = time.perf_counter() - t0
    print(f"\n📊 阶段 2 完成: 耗时 {total_elapsed:.1f}s")

    # 合并各频率的 bars
    results = {}
    for freq in target_freqs:
        if all_bars[freq]:
            results[freq] = pd.concat(all_bars[freq], ignore_index=False)
            print(f"   freq={freq:>2d}: {len(results[freq]):>8,} bars")
        else:
            results[freq] = pd.DataFrame()
            print(f"   freq={freq:>2d}:        0 bars (empty)")

    del all_bars
    gc.collect()

    return results


# ──────────────────────────────────────────────
# 保存 Dollar Bars
# ──────────────────────────────────────────────
def save_dollar_bars(bars_dict: dict[int, pd.DataFrame], output_dir: str):
    """将各频率 Dollar Bars 保存为 CSV 文件（因 Parquet 引擎缺失）。"""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n💾 保存 Dollar Bars 到 {output_dir}/")
    for freq, bars_df in bars_dict.items():
        if len(bars_df) == 0:
            continue
        path = os.path.join(output_dir, f"dollar_bars_freq{freq}.csv")
        # 保存为 CSV，包含索引（时间戳）
        bars_df.to_csv(path)
        print(f"   freq={freq:>2d}: {path}  ({len(bars_df):,} bars)")


# ──────────────────────────────────────────────
# IID 评估
# ──────────────────────────────────────────────
def evaluate_iid(ohlcv: pd.DataFrame) -> dict:
    """对 OHLCV 的 close 计算对数收益率，执行 JB 检验和一阶自相关。"""
    log_ret = np.log(ohlcv["close"] / ohlcv["close"].shift(1)).dropna()
    log_ret = log_ret.replace([np.inf, -np.inf], np.nan).dropna()

    jb_stat, jb_pvalue = stats.jarque_bera(log_ret)
    autocorr_1 = log_ret.autocorr(lag=1)

    return {
        "n_bars": len(ohlcv),
        "n_returns": len(log_ret),
        "jb_stat": jb_stat,
        "jb_pvalue": jb_pvalue,
        "autocorr_1": autocorr_1,
        "mean_ret": log_ret.mean(),
        "std_ret": log_ret.std(),
        "skew": log_ret.skew(),
        "kurtosis": log_ret.kurtosis(),
        "log_ret": log_ret,
    }


def print_results_table(results: dict[int, dict]) -> int:
    """打印结果表格，返回最优频率。"""
    print("\n" + "=" * 90)
    print("  Dollar Bars IID 评估结果 (全量 37 个月)")
    print("=" * 90)
    print(f"{'频率':>6s} | {'Bar 数':>10s} | {'JB 统计量':>14s} | {'JB p-value':>12s} | {'一阶自相关':>10s} | {'偏度':>8s} | {'峰度':>8s}")
    print("-" * 90)

    for freq in sorted(results.keys()):
        r = results[freq]
        print(
            f"{freq:>6d} | {r['n_bars']:>10,d} | {r['jb_stat']:>14.2f} | {r['jb_pvalue']:>12.6f} | "
            f"{r['autocorr_1']:>10.4f} | {r['skew']:>8.4f} | {r['kurtosis']:>8.4f}"
        )

    # 综合排序: JB 归一化 + |自相关| 归一化，各 50%
    freqs_sorted = sorted(results.keys())
    jb_vals = np.array([results[f]["jb_stat"] for f in freqs_sorted])
    ac_vals = np.array([abs(results[f]["autocorr_1"]) for f in freqs_sorted])

    jb_norm = (jb_vals - jb_vals.min()) / (jb_vals.max() - jb_vals.min() + 1e-12)
    ac_norm = (ac_vals - ac_vals.min()) / (ac_vals.max() - ac_vals.min() + 1e-12)
    scores = 0.5 * jb_norm + 0.5 * ac_norm

    best_idx = int(np.argmin(scores))
    best_freq = freqs_sorted[best_idx]

    print("-" * 90)
    print(f"🏆 最优频率: {best_freq} bars/day  (JB={results[best_freq]['jb_stat']:.2f}, "
          f"AutoCorr={results[best_freq]['autocorr_1']:.4f})")
    print("=" * 90)

    return best_freq


# ──────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────
def plot_evaluation(
    results: dict[int, dict],
    best_freq: int,
    bars_dict: dict[int, pd.DataFrame],
    output_path: str,
):
    """生成 4 面板可视化图表。"""
    freqs = sorted(results.keys())
    jb_stats = [results[f]["jb_stat"] for f in freqs]
    autocorrs = [results[f]["autocorr_1"] for f in freqs]

    best_color = "#e15759"
    bar_colors = [best_color if f == best_freq else "#4e79a7" for f in freqs]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Dynamic Dollar Bars — IID Evaluation (Full 37-Month Dataset)",
        fontsize=16, fontweight="bold", y=0.98,
    )
    fig.patch.set_facecolor("#fafafa")

    # ── 子图 1: JB 统计量 ──
    ax1 = axes[0, 0]
    bars1 = ax1.bar([str(f) for f in freqs], jb_stats, color=bar_colors, edgecolor="white", linewidth=1.2)
    ax1.set_title("Jarque-Bera Statistic by Frequency", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Target Bars / Day")
    ax1.set_ylabel("JB Statistic (lower = more normal)")
    ax1.set_yscale("log")
    for bar_rect, val in zip(bars1, jb_stats):
        ax1.text(bar_rect.get_x() + bar_rect.get_width() / 2, bar_rect.get_height(),
                 f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)

    # ── 子图 2: 自相关 ──
    ax2 = axes[0, 1]
    bars2 = ax2.bar([str(f) for f in freqs], autocorrs, color=bar_colors, edgecolor="white", linewidth=1.2)
    ax2.set_title("First-Order Autocorrelation by Frequency", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Target Bars / Day")
    ax2.set_ylabel("Autocorrelation (closer to 0 = more IID)")
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    for bar_rect, val in zip(bars2, autocorrs):
        offset = 0.002 if val >= 0 else -0.002
        ax2.text(bar_rect.get_x() + bar_rect.get_width() / 2, val + offset,
                 f"{val:.4f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

    # ── 子图 3: 最优频率的对数收益率分布 ──
    ax3 = axes[1, 0]
    best_ret = results[best_freq]["log_ret"]
    n_bins = min(200, max(50, len(best_ret) // 100))
    ax3.hist(best_ret, bins=n_bins, density=True, alpha=0.7, color="#4e79a7",
             edgecolor="white", linewidth=0.3, label="Log Returns")
    x_range = np.linspace(best_ret.min(), best_ret.max(), 300)
    mu, sigma = best_ret.mean(), best_ret.std()
    ax3.plot(x_range, stats.norm.pdf(x_range, mu, sigma), color=best_color, linewidth=2,
             label=f"Normal Fit (μ={mu:.6f}, σ={sigma:.6f})")
    ax3.set_title(f"Log Return Distribution — Best Freq: {best_freq} bars/day", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Log Return")
    ax3.set_ylabel("Density")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.set_axisbelow(True)

    # ── 子图 4: 最优频率的收盘价时间序列 ──
    ax4 = axes[1, 1]
    best_ohlcv = bars_dict.get(best_freq)
    if best_ohlcv is not None and len(best_ohlcv) > 0:
        plot_data = best_ohlcv["close"]
        if len(plot_data) > 5000:
            step = len(plot_data) // 5000
            plot_data = plot_data.iloc[::step]
        ax4.plot(plot_data.index, plot_data.values, color="#4e79a7", linewidth=0.5, alpha=0.9)
        ax4.set_title(f"Close Price — {best_freq} bars/day ({len(best_ohlcv):,} bars)", fontsize=13, fontweight="bold")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Price (USDT)")
        ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax4.grid(alpha=0.3)
        ax4.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n📈 可视化已保存: {output_path}")


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Dynamic Dollar Bars — 全量数据 IID 评估")
    print("  (两阶段流式处理)")
    print("=" * 60)

    # 环境检查
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BARS_OUTPUT_DIR, exist_ok=True)

    # 阶段 1: 扫描 daily dollar volume
    daily_dollar_vol, month_keys = scan_daily_dollar_volumes(DATA_PATH)

    # 计算各频率 EWMA 阈值
    daily_thresholds = compute_ewma_thresholds(daily_dollar_vol, TARGET_FREQS, EWMA_SPAN)
    del daily_dollar_vol
    gc.collect()

    # 阶段 2: 逐月流式生成 bars
    bars_dict = stream_dollar_bars(DATA_PATH, month_keys, daily_thresholds, TARGET_FREQS)

    # 保存 Dollar Bars
    save_dollar_bars(bars_dict, BARS_OUTPUT_DIR)

    # IID 评估
    print("\n🔬 执行 IID 评估...")
    eval_results: dict[int, dict] = {}
    for freq in TARGET_FREQS:
        if len(bars_dict[freq]) > 0:
            eval_results[freq] = evaluate_iid(bars_dict[freq])

    if not eval_results:
        print("❌ 所有频率均未生成 bars，请检查数据。")
        return

    # 打印结果
    best_freq = print_results_table(eval_results)

    # 可视化
    plot_evaluation(eval_results, best_freq, bars_dict, OUTPUT_FILE)

    # 释放
    del bars_dict
    gc.collect()

    print("\n✅ 全量评估完成！")


if __name__ == "__main__":
    main()
