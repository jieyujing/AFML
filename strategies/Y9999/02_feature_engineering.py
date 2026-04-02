"""
03_feature_engineering.py - Y9999 Phase 2 特征工程

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
from statsmodels.tsa.stattools import adfuller
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.Y9999.config import (
    BARS_DIR, FIGURES_DIR, FEATURES_DIR,
    FRACDIFF_THRES, FRACDIFF_D_STEP, FRACDIFF_MAX_D,
    CUSUM_WINDOW, CUSUM_MULTIPLIER, FEATURE_CONFIG
)

from afmlkit.feature.core.frac_diff import optimize_d, frac_diff_ffd
from afmlkit.sampling.filters import cusum_filter_with_state
from strategies.Y9999.feature_compute import (
    compute_all_features,
    compute_event_features,
    compute_trend_scan_labels,
)

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

    # ADF 验证（使用 statsmodels 避免 Numba 问题）
    valid_series = fracdiff_series.dropna()
    t_stat, p_value = None, None
    if len(valid_series) > 10:
        adf_result = adfuller(valid_series.values, regression='c')
        t_stat, p_value = adf_result[0], adf_result[1]
        print(f"  ADF 检验: t={t_stat:.4f}, p={p_value:.4f}")
        if p_value < 0.05:
            print(f"  ✅ 序列平稳 (p < 0.05)")
        else:
            print(f"  ⚠️ 序列非平稳 (p >= 0.05)")

    return optimal_d, fracdiff_series, (t_stat, p_value)


# ============================================================
# CUSUM Filter 模块
# ============================================================

def compute_dynamic_cusum_threshold(fracdiff_series: pd.Series, window: int, multiplier: float = 1.0) -> pd.Series:
    """
    计算动态 CUSUM 阈值。

    基于 CUSUM 输入序列（FracDiff 变化量）的滚动波动率，
    确保阈值与实际输入匹配。

    :param fracdiff_series: FracDiff 序列
    :param window: 滚动窗口
    :param multiplier: 阈值乘数，用于控制事件率
    :returns: 阈值序列
    """
    # 计算 FracDiff 的变化量（CUSUM 的实际输入）
    fracdiff_diff = fracdiff_series.diff()

    # 阈值基于变化量的标准差，与 CUSUM 输入一致
    threshold_series = fracdiff_diff.rolling(window).std() * multiplier

    # 前 window 个点用全局 std 填充
    global_std = fracdiff_diff.std() * multiplier
    threshold_series = threshold_series.fillna(global_std)

    print(f"\n[CUSUM] 动态阈值计算完成")
    print(f"  窗口: {window}")
    print(f"  乘数: {multiplier}")
    print(f"  输入: FracDiff 变化量序列")
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
    # CUSUM Filter 使用 FracDiff 的变化量，索引需从变化量序列获取
    fracdiff_diff = fracdiff_series.diff()
    valid_idx = fracdiff_diff.dropna().index

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
    :param event_indices: 事件点索引（对应 fracdiff_diff）
    :param fracdiff_series: FracDiff 序列（用于映射索引）
    :param save_path: 保存路径
    """
    # CUSUM Filter 使用 FracDiff 的变化量，索引需从变化量序列获取
    fracdiff_diff = fracdiff_series.diff()
    valid_idx = fracdiff_diff.dropna().index
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


# ============================================================
# 输出保存
# ============================================================

def save_outputs(
    bars: pd.DataFrame,
    fracdiff_series: pd.Series,
    optimal_d: float,
    event_indices: np.ndarray,
    adf_p_value: float  # ADF 检验 p 值
):
    """
    保存 FracDiff 序列和 CUSUM 事件点。

    :param bars: Dollar Bars DataFrame
    :param fracdiff_series: FracDiff 序列（含 NaN）
    :param optimal_d: 最优 d 值
    :param event_indices: 事件点索引（对应 fracdiff_diff）
    :param adf_p_value: ADF 检验 p 值
    """
    os.makedirs(FEATURES_DIR, exist_ok=True)

    # 1. FracDiff 序列
    fracdiff_df = pd.DataFrame({
        'fracdiff': fracdiff_series,
    })
    fracdiff_path = os.path.join(FEATURES_DIR, 'fracdiff_series.parquet')
    fracdiff_df.to_parquet(fracdiff_path)
    print(f"\n✅ FracDiff 序列已保存: {fracdiff_path}")
    print(f"   长度: {len(fracdiff_df)}")

    # 保存参数信息
    params_df = pd.DataFrame({
        'optimal_d': [optimal_d],
        'n_events': [len(event_indices)],
        'adf_p_value': [adf_p_value]
    })
    params_path = os.path.join(FEATURES_DIR, 'fracdiff_params.parquet')
    params_df.to_parquet(params_path)
    print(f"✅ 参数信息已保存: {params_path}")

    # 2. CUSUM 事件点（使用变化量序列的索引）
    fracdiff_diff = fracdiff_series.diff()
    valid_idx = fracdiff_diff.dropna().index
    event_timestamps = valid_idx[event_indices]

    events_df = pd.DataFrame({
        'timestamp': event_timestamps,
        'price': bars['close'].loc[event_timestamps].values,
        'fracdiff': fracdiff_series.loc[event_timestamps].values,
    })
    events_path = os.path.join(FEATURES_DIR, 'cusum_events.parquet')
    events_df.to_parquet(events_path)
    print(f"✅ CUSUM 事件点已保存: {events_path}")
    print(f"   事件数量: {len(events_df)}")


def main():
    """
    Y9999 Phase 2 特征工程主流程。
    """
    print("=" * 70)
    print("  Y9999 Phase 2 Feature Engineering")
    print("=" * 70)

    # 确保输出目录存在
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: 加载 Dollar Bars
    print("\n[Step 1] 加载 Dollar Bars...")
    bars_path = os.path.join(BARS_DIR, 'dollar_bars_target8.parquet')
    bars = load_dollar_bars(bars_path)
    prices = bars['close']

    # Step 2: FracDiff 优化
    print("\n[Step 2] FracDiff 参数优化...")
    optimal_d, fracdiff_series, adf_result = run_fracdiff_optimization(prices)

    # Step 3: CUSUM 阈值
    print("\n[Step 3] 计算 CUSUM 动态阈值...")
    threshold_series = compute_dynamic_cusum_threshold(fracdiff_series, CUSUM_WINDOW, CUSUM_MULTIPLIER)

    # Step 4: CUSUM Filter
    print("\n[Step 4] 应用 CUSUM Filter...")
    event_indices, s_pos, s_neg, n_events = run_cusum_filter(fracdiff_series, threshold_series)

    # Step 5: 可视化
    print("\n[Step 5] 生成可视化图表...")
    plot_fracdiff_comparison(
        prices, fracdiff_series, optimal_d,
        os.path.join(FIGURES_DIR, '02_fracdiff_comparison.png')
    )

    plot_cusum_state(
        fracdiff_series, s_pos, s_neg, threshold_series, event_indices,
        os.path.join(FIGURES_DIR, '02_cusum_state.png')
    )

    plot_event_distribution(
        bars, event_indices, fracdiff_series,
        os.path.join(FIGURES_DIR, '02_event_distribution.png')
    )

    # Step 6: 保存基础输出
    print("\n[Step 6] 保存基础输出文件...")
    save_outputs(
        bars, fracdiff_series, optimal_d, event_indices,
        adf_result[1]  # ADF p-value
    )

    # Step 7: 计算全量特征
    print("\n[Step 7] 计算全量特征...")
    features_df = compute_all_features(bars, FEATURE_CONFIG)
    features_path = os.path.join(FEATURES_DIR, 'bars_features.parquet')
    features_df.to_parquet(features_path)
    print(f"✅ 全量特征已保存: {features_path}")
    print(f"   特征数量: {len(features_df.columns)}")
    print(f"   样本数量: {len(features_df)}")

    # Step 8: 提取事件点特征
    print("\n[Step 8] 提取事件点特征...")
    # 转换 event_indices 为实际 bars 索引
    fracdiff_diff = fracdiff_series.diff()
    valid_idx = fracdiff_diff.dropna().index
    # event_indices 是相对于 valid_idx 的索引，需要转换为 bars 的整数索引
    bars_event_indices = np.array([bars.index.get_loc(ts) for ts in valid_idx[event_indices]])

    event_features_df = compute_event_features(bars, features_df, bars_event_indices)
    event_features_path = os.path.join(FEATURES_DIR, 'events_features.parquet')
    event_features_df.to_parquet(event_features_path)
    print(f"✅ 事件点特征已保存: {event_features_path}")
    print(f"   事件数量: {len(event_features_df)}")
    print(f"   特征数量: {len(event_features_df.columns)}")

    # Step 9: Trend Scan 标签（含未来信息）
    if FEATURE_CONFIG.get('trend_scan', {}).get('enabled', False):
        print("\n[Step 9] 计算 Trend Scan 标签...")
        trend_scan_df = compute_trend_scan_labels(bars, bars_event_indices, FEATURE_CONFIG['trend_scan'])
        trend_scan_path = os.path.join(FEATURES_DIR, 'trend_scan_labels.parquet')
        trend_scan_df.to_parquet(trend_scan_path)
        print(f"✅ Trend Scan 标签已保存: {trend_scan_path}")
        print(f"   标签数量: {len(trend_scan_df)}")
        # 统计多空分布
        long_count = (trend_scan_df['side'] == 1).sum()
        short_count = (trend_scan_df['side'] == -1).sum()
        print(f"   多头标签: {long_count}, 空头标签: {short_count}")

    # 完成
    print("\n" + "=" * 70)
    print("  Phase 2 Feature Engineering 完成")
    print("=" * 70)
    print(f"输出目录: {FEATURES_DIR}")
    print(f"  - fracdiff_series.parquet")
    print(f"  - fracdiff_params.parquet")
    print(f"  - cusum_events.parquet")
    print(f"  - bars_features.parquet (全量特征)")
    print(f"  - events_features.parquet (事件点特征)")
    if FEATURE_CONFIG.get('trend_scan', {}).get('enabled', False):
        print(f"  - trend_scan_labels.parquet (Trend Scan 标签)")
    print(f"图表目录: {FIGURES_DIR}")
    print(f"  - 02_fracdiff_comparison.png")
    print(f"  - 02_cusum_state.png")
    print(f"  - 02_event_distribution.png")


if __name__ == "__main__":
    main()