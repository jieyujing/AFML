"""
03_feature_engineering.py - IF9999 Phase 2 特征工程

流程:
1. 加载 Dollar Bars
2. FracDiff 参数优化（自动搜索最优 d）
3. 应用 FracDiff 生成平稳序列
4. 计算动态 CUSUM 阈值
5. 应用 CUSUM Filter 采样事件点
6. 可视化验证
7. 输出保存
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.IF9999.config import (
    BARS_DIR, FIGURES_DIR, FEATURES_DIR,
    FRACDIFF_THRES, FRACDIFF_D_STEP, FRACDIFF_MAX_D,
    CUSUM_WINDOW
)

from afmlkit.feature.core.frac_diff import optimize_d, frac_diff_ffd
from afmlkit.feature.core.structural_break.adf import adf_test
from afmlkit.sampling.filters import cusum_filter_with_state

sns.set_theme(style="whitegrid", context="paper")


# ============================================================
# 数据加载
# ============================================================

def load_dollar_bars(bars_path: str) -> pd.DataFrame:
    """
    加载 Dollar Bars 数据。

    :param bars_path: parquet 文件路径
    :returns: DataFrame with timestamp index
    """
    bars = pd.read_parquet(bars_path)
    print(f"✅ 加载 Dollar Bars: {len(bars)} bars")
    print(f"   时间范围: {bars.index.min()} ~ {bars.index.max()}")
    return bars


# ============================================================
# FracDiff 模块
# ============================================================

def run_fracdiff_optimization(prices: pd.Series) -> tuple:
    """
    运行 FracDiff 参数优化。

    :param prices: 价格序列
    :returns: (optimal_d, fracdiff_series, adf_result)
    """
    print("\n[FracDiff] 搜索最优 d 值...")

    # 搜索最优 d
    optimal_d = optimize_d(
        prices,
        thres=FRACDIFF_THRES,
        d_step=FRACDIFF_D_STEP,
        max_d=FRACDIFF_MAX_D,
        min_corr=0.0
    )
    print(f"  最优 d = {optimal_d:.4f}")

    # 应用 FracDiff
    fracdiff_series = frac_diff_ffd(prices, d=optimal_d, thres=FRACDIFF_THRES)
    print(f"  FracDiff 序列长度: {len(fracdiff_series)} (截断 {len(prices) - len(fracdiff_series)} 个)")

    # ADF 验证
    valid_series = fracdiff_series.dropna()
    t_stat, p_value = None, None
    if len(valid_series) > 10:
        t_stat, p_value, _ = adf_test(valid_series)
        print(f"  ADF 检验: t={t_stat:.4f}, p={p_value:.4f}")
        if p_value < 0.05:
            print(f"  ✅ 序列平稳 (p < 0.05)")
        else:
            print(f"  ⚠️ 序列非平稳 (p >= 0.05)")

    return optimal_d, fracdiff_series, (t_stat, p_value)


# ============================================================
# CUSUM Filter 模块
# ============================================================

def compute_dynamic_cusum_threshold(fracdiff_series: pd.Series, window: int) -> pd.Series:
    """
    计算动态 CUSUM 阈值。

    基于滚动波动率，与 Dollar Bars 动态阈值风格一致。

    :param fracdiff_series: FracDiff 序列
    :param window: 滚动窗口
    :returns: 阈值序列
    """
    # 滚动标准差
    threshold_series = fracdiff_series.rolling(window).std()

    # 前 window 个点用全局 std 填充
    global_std = fracdiff_series.std()
    threshold_series = threshold_series.fillna(global_std)

    print(f"\n[CUSUM] 动态阈值计算完成")
    print(f"  窗口: {window}")
    print(f"  平均阈值: {threshold_series.mean():.6f}")
    print(f"  阈值范围: {threshold_series.min():.6f} ~ {threshold_series.max():.6f}")

    return threshold_series


def run_cusum_filter(fracdiff_series: pd.Series, threshold_series: pd.Series) -> tuple:
    """
    应用 CUSUM Filter 采样事件点。

    使用 FracDiff 序列的变化量作为 CUSUM 输入，
    因为 CUSUM Filter 期望输入是差值序列（如收益率），
    而 FracDiff 输出是价格量级的值。

    :param fracdiff_series: FracDiff 序列
    :param threshold_series: 阈值序列
    :returns: (event_indices, s_pos, s_neg, n_events)
    """
    # 计算 FracDiff 的变化量（类似收益率）
    fracdiff_diff = fracdiff_series.diff()
    valid_idx = fracdiff_diff.dropna().index

    # 准备输入数组
    diff_arr = fracdiff_diff.dropna().values.astype(np.float64)
    threshold_arr = threshold_series.loc[valid_idx].values.astype(np.float64)

    # 应用 CUSUM Filter
    event_indices, s_pos, s_neg, thr = cusum_filter_with_state(diff_arr, threshold_arr)

    n_events = len(event_indices)
    print(f"\n[CUSUM] Filter 应用完成")
    print(f"  输入: FracDiff 变化量序列")
    print(f"  事件点数量: {n_events}")
    print(f"  事件率: {n_events / len(diff_arr) * 100:.2f}%")

    return event_indices, s_pos, s_neg, n_events


# ============================================================
# 可视化模块
# ============================================================

def plot_fracdiff_comparison(
    prices: pd.Series,
    fracdiff_series: pd.Series,
    optimal_d: float,
    save_path: str
):
    """
    绘制价格 vs FracDiff 对比图。

    :param prices: 原始价格序列
    :param fracdiff_series: FracDiff 序列
    :param optimal_d: 最优 d 值
    :param save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # 价格序列
    ax1.plot(prices.index, prices.values, color='steelblue', linewidth=0.8)
    ax1.set_title('Original Price Series', fontsize=12)
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)

    # FracDiff 序列
    ax2.plot(fracdiff_series.index, fracdiff_series.values, color='darkorange', linewidth=0.8)
    ax2.set_title(f'FracDiff Series (d={optimal_d:.4f})', fontsize=12)
    ax2.set_ylabel('FracDiff')
    ax2.set_xlabel('Time')
    ax2.grid(True, alpha=0.3)

    # 添加零线
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ FracDiff 对比图已保存: {save_path}")


def plot_cusum_state(
    fracdiff_series: pd.Series,
    s_pos: np.ndarray,
    s_neg: np.ndarray,
    threshold_series: pd.Series,
    event_indices: np.ndarray,
    save_path: str
):
    """
    绘制 CUSUM 状态曲线图。

    :param fracdiff_series: FracDiff 序列（用于获取 index）
    :param s_pos: 正向累积数组
    :param s_neg: 负向累积数组
    :param threshold_series: 阈值序列
    :param event_indices: 事件点索引
    :param save_path: 保存路径
    """
    valid_idx = fracdiff_series.dropna().index

    fig, ax = plt.subplots(figsize=(14, 6))

    # CUSUM 状态曲线
    ax.plot(valid_idx, s_pos, color='green', linewidth=0.8, label='S+ (positive)')
    ax.plot(valid_idx, np.abs(s_neg), color='red', linewidth=0.8, label='S- (negative)')

    # 阈值线（取平均阈值）
    mean_threshold = threshold_series.mean()
    ax.axhline(y=mean_threshold, color='gray', linestyle='--', linewidth=1.5, label=f'Threshold (mean={mean_threshold:.4f})')

    # 事件点标记
    event_timestamps = valid_idx[event_indices]
    ax.scatter(event_timestamps, [mean_threshold] * len(event_indices),
               color='black', s=20, marker='^', label=f'Events (n={len(event_indices)})', zorder=5)

    ax.set_title('CUSUM Filter State Curves', fontsize=12)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Sum')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ CUSUM 状态图已保存: {save_path}")


def plot_event_distribution(
    bars: pd.DataFrame,
    event_indices: np.ndarray,
    fracdiff_series: pd.Series,
    save_path: str
):
    """
    绘制事件点分布图（价格序列上标记事件点）。

    :param bars: Dollar Bars DataFrame
    :param event_indices: 事件点索引（对应 fracdiff_series）
    :param fracdiff_series: FracDiff 序列（用于映射索引）
    :param save_path: 保存路径
    """
    valid_idx = fracdiff_series.dropna().index
    event_timestamps = valid_idx[event_indices]

    # 事件点对应的价格
    event_prices = bars['close'].loc[event_timestamps]

    fig, ax = plt.subplots(figsize=(14, 6))

    # 价格序列
    ax.plot(bars.index, bars['close'].values, color='steelblue', linewidth=0.8, label='Price')

    # 事件点标记
    ax.scatter(event_timestamps, event_prices.values,
               color='red', s=30, marker='o', label=f'CUSUM Events (n={len(event_indices)})', zorder=5)

    ax.set_title('CUSUM Event Distribution on Price Series', fontsize=12)
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 事件分布图已保存: {save_path}")


def main():
    """测试完整 FracDiff + CUSUM 流程"""
    bars_path = os.path.join(BARS_DIR, 'dollar_bars_target6.parquet')

    # 加载数据
    bars = load_dollar_bars(bars_path)
    prices = bars['close']

    # FracDiff
    optimal_d, fracdiff_series, adf_result = run_fracdiff_optimization(prices)

    # CUSUM 阈值
    threshold_series = compute_dynamic_cusum_threshold(fracdiff_series, CUSUM_WINDOW)

    # CUSUM Filter
    event_indices, s_pos, s_neg, n_events = run_cusum_filter(fracdiff_series, threshold_series)

    print(f"\n完整流程测试完成")
    print(f"  optimal_d = {optimal_d:.4f}")
    print(f"  n_events = {n_events}")


if __name__ == "__main__":
    main()