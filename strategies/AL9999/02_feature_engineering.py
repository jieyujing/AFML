"""
02_feature_engineering.py - AL9999 Phase 2 特征工程

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.AL9999.config import (
    BARS_DIR, FIGURES_DIR, FEATURES_DIR,
    FRACDIFF_THRES, FRACDIFF_D_STEP, FRACDIFF_MAX_D,
    CUSUM_WINDOW, CUSUM_MULTIPLIER, FEATURE_CONFIG,
    TARGET_DAILY_BARS
)

from afmlkit.feature.core.frac_diff import optimize_d, frac_diff_ffd
from afmlkit.sampling.filters import cusum_filter_with_state
from strategies.AL9999.feature_compute import (
    compute_all_features,
    compute_event_features,
    compute_trend_scan_labels,
)

sns.set_theme(style="whitegrid", context="paper")


def load_dollar_bars(bars_path: str) -> pd.DataFrame:
    """加载 Dollar Bars 数据。"""
    bars = pd.read_parquet(bars_path)
    print(f"✅ 加载 Dollar Bars: {len(bars)} bars")
    print(f"   时间范围: {bars.index.min()} ~ {bars.index.max()}")
    return bars


def run_fracdiff_optimization(prices: pd.Series) -> tuple:
    """运行 FracDiff 参数优化。"""
    print("\n[FracDiff] 搜索最优 d 值...")

    optimal_d = optimize_d(
        prices,
        thres=FRACDIFF_THRES,
        d_step=FRACDIFF_D_STEP,
        max_d=FRACDIFF_MAX_D,
        min_corr=0.0
    )
    print(f"  最优 d = {optimal_d:.4f}")

    fracdiff_series = frac_diff_ffd(prices, d=optimal_d, thres=FRACDIFF_THRES)
    print(f"  FracDiff 序列长度: {len(fracdiff_series)} (截断 {len(prices) - len(fracdiff_series)} 个)")

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


def compute_dynamic_cusum_threshold(fracdiff_series: pd.Series, window: int, multiplier: float = 1.0) -> pd.Series:
    """计算动态 CUSUM 阈值。"""
    fracdiff_diff = fracdiff_series.diff()
    threshold_series = fracdiff_diff.rolling(window).std() * multiplier
    global_std = fracdiff_diff.std() * multiplier
    threshold_series = threshold_series.fillna(global_std)

    print(f"\n[CUSUM] 动态阈值计算完成")
    print(f"  窗口: {window}")
    print(f"  乘数: {multiplier}")
    print(f"  平均阈值: {threshold_series.mean():.6f}")

    return threshold_series


def run_cusum_filter(fracdiff_series: pd.Series, threshold_series: pd.Series) -> tuple:
    """应用 CUSUM Filter 采样事件点。"""
    fracdiff_diff = fracdiff_series.diff()
    valid_idx = fracdiff_diff.dropna().index

    diff_arr = fracdiff_diff.dropna().values.astype(np.float64)
    threshold_arr = threshold_series.loc[valid_idx].values.astype(np.float64)

    event_indices, s_pos, s_neg, thr = cusum_filter_with_state(diff_arr, threshold_arr)

    n_events = len(event_indices)
    print(f"\n[CUSUM] Filter 应用完成")
    print(f"  事件点数量: {n_events}")
    print(f"  事件率: {n_events / len(diff_arr) * 100:.2f}%")

    return event_indices, s_pos, s_neg, n_events


def plot_fracdiff_comparison(prices, fracdiff_series, optimal_d, save_path):
    """绘制价格 vs FracDiff 对比图。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(prices.index, prices.values, color='steelblue', linewidth=0.8)
    ax1.set_title('Original Price Series (AL9999)', fontsize=12)
    ax1.set_ylabel('Price (元/吨)')
    ax1.grid(True, alpha=0.3)

    ax2.plot(fracdiff_series.index, fracdiff_series.values, color='darkorange', linewidth=0.8)
    ax2.set_title(f'FracDiff Series (d={optimal_d:.4f})', fontsize=12)
    ax2.set_ylabel('FracDiff')
    ax2.set_xlabel('Time')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ FracDiff 对比图已保存: {save_path}")


def plot_cusum_state(fracdiff_series, s_pos, s_neg, threshold_series, event_indices, save_path):
    """绘制 CUSUM 状态曲线图。"""
    fracdiff_diff = fracdiff_series.diff()
    valid_idx = fracdiff_diff.dropna().index

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(valid_idx, s_pos, color='green', linewidth=0.8, label='S+ (positive)')
    ax.plot(valid_idx, np.abs(s_neg), color='red', linewidth=0.8, label='S- (negative)')

    mean_threshold = threshold_series.mean()
    ax.axhline(y=mean_threshold, color='gray', linestyle='--', linewidth=1.5, label=f'Threshold (mean={mean_threshold:.4f})')

    event_timestamps = valid_idx[event_indices]
    ax.scatter(event_timestamps, [mean_threshold] * len(event_indices),
               color='black', s=20, marker='^', label=f'Events (n={len(event_indices)})', zorder=5)

    ax.set_title('CUSUM Filter State Curves (AL9999)', fontsize=12)
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Sum')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ CUSUM 状态图已保存: {save_path}")


def plot_event_distribution(bars, event_indices, fracdiff_series, save_path):
    """绘制事件点分布图。"""
    fracdiff_diff = fracdiff_series.diff()
    valid_idx = fracdiff_diff.dropna().index
    event_timestamps = valid_idx[event_indices]
    event_prices = bars['close'].loc[event_timestamps]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(bars.index, bars['close'].values, color='steelblue', linewidth=0.8, label='Price')

    ax.scatter(event_timestamps, event_prices.values,
               color='red', s=30, marker='o', label=f'CUSUM Events (n={len(event_indices)})', zorder=5)

    ax.set_title('CUSUM Event Distribution on Price Series (AL9999)', fontsize=12)
    ax.set_xlabel('Time')
    ax.set_ylabel('Price (元/吨)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 事件分布图已保存: {save_path}")


def save_outputs(bars, fracdiff_series, optimal_d, event_indices, adf_p_value,
                 s_pos=None, s_neg=None):
    """保存 FracDiff 序列和 CUSUM 事件点。"""
    os.makedirs(FEATURES_DIR, exist_ok=True)

    fracdiff_df = pd.DataFrame({'fracdiff': fracdiff_series})
    fracdiff_path = os.path.join(FEATURES_DIR, 'fracdiff_series.parquet')
    fracdiff_df.to_parquet(fracdiff_path)
    print(f"\n✅ FracDiff 序列已保存: {fracdiff_path}")

    params_df = pd.DataFrame({
        'optimal_d': [optimal_d],
        'n_events': [len(event_indices)],
        'adf_p_value': [adf_p_value]
    })
    params_path = os.path.join(FEATURES_DIR, 'fracdiff_params.parquet')
    params_df.to_parquet(params_path)
    print(f"✅ 参数信息已保存: {params_path}")

    fracdiff_diff = fracdiff_series.diff()
    valid_idx = fracdiff_diff.dropna().index
    event_timestamps = valid_idx[event_indices]

    events_data = {
        'timestamp': event_timestamps,
        'price': bars['close'].loc[event_timestamps].values,
        'fracdiff': fracdiff_series.loc[event_timestamps].values,
    }
    if s_pos is not None and len(s_pos) == len(event_indices):
        events_data['g_up'] = s_pos
    if s_neg is not None and len(s_neg) == len(event_indices):
        events_data['g_down'] = s_neg

    events_df = pd.DataFrame(events_data)
    events_path = os.path.join(FEATURES_DIR, 'cusum_events.parquet')
    events_df.to_parquet(events_path)
    n_directional = sum(1 for k in ['g_up', 'g_down'] if k in events_data)
    print(f"✅ CUSUM 事件点已保存: {events_path}")
    print(f"   事件数: {len(events_df)}, 方向列: {n_directional}")


def main():
    """AL9999 Phase 2 特征工程主流程。"""
    print("=" * 70)
    print("  AL9999 Phase 2 Feature Engineering")
    print("=" * 70)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Step 1: 加载 Dollar Bars
    print("\n[Step 1] 加载 Dollar Bars...")
    bars_path = os.path.join(BARS_DIR, f'dollar_bars_target{TARGET_DAILY_BARS}.parquet')
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

    # Extract g_up/g_down for each event.
    # s_pos_history[i] / s_neg_history[i] record the state AFTER the update
    # but BEFORE the reset (lines 115-116 vs 118-125 in filters.py).
    # So s_pos/s_neg **at** the event index has the triggering value.
    g_up_vals = s_pos[event_indices].copy()
    g_down_vals = s_neg[event_indices].copy()

    # Step 5: 可视化
    print("\n[Step 5] 生成可视化图表...")
    plot_fracdiff_comparison(prices, fracdiff_series, optimal_d,
                              os.path.join(FIGURES_DIR, '02_fracdiff_comparison.png'))
    plot_cusum_state(fracdiff_series, s_pos, s_neg, threshold_series, event_indices,
                     os.path.join(FIGURES_DIR, '02_cusum_state.png'))
    plot_event_distribution(bars, event_indices, fracdiff_series,
                            os.path.join(FIGURES_DIR, '02_event_distribution.png'))

    # Step 6: 保存基础输出
    print("\n[Step 6] 保存基础输出文件...")
    save_outputs(bars, fracdiff_series, optimal_d, event_indices, adf_result[1],
                 s_pos=g_up_vals, s_neg=g_down_vals)

    # Step 7: 计算全量特征
    print("\n[Step 7] 计算全量特征...")
    features_df = compute_all_features(bars, FEATURE_CONFIG)
    features_path = os.path.join(FEATURES_DIR, 'bars_features.parquet')
    features_df.to_parquet(features_path)
    print(f"✅ 全量特征已保存: {features_path}")
    print(f"   特征数量: {len(features_df.columns)}")

    # Step 8: 提取事件点特征
    print("\n[Step 8] 提取事件点特征...")
    fracdiff_diff = fracdiff_series.diff()
    valid_idx = fracdiff_diff.dropna().index
    bars_event_indices = np.array([bars.index.get_loc(ts) for ts in valid_idx[event_indices]])

    event_features_df = compute_event_features(bars, features_df, bars_event_indices)
    event_features_path = os.path.join(FEATURES_DIR, 'events_features.parquet')
    event_features_df.to_parquet(event_features_path)
    print(f"✅ 事件点特征已保存: {event_features_path}")
    print(f"   事件数量: {len(event_features_df)}")

    # Step 9: Trend Scan 标签
    if FEATURE_CONFIG.get('trend_scan', {}).get('enabled', False):
        print("\n[Step 9] 计算 Trend Scan 标签...")
        trend_scan_df = compute_trend_scan_labels(bars, bars_event_indices, FEATURE_CONFIG['trend_scan'])
        trend_scan_path = os.path.join(FEATURES_DIR, 'trend_scan_labels.parquet')
        trend_scan_df.to_parquet(trend_scan_path)
        print(f"✅ Trend Scan 标签已保存: {trend_scan_path}")
        long_count = (trend_scan_df['side'] == 1).sum()
        short_count = (trend_scan_df['side'] == -1).sum()
        print(f"   多头标签: {long_count}, 空头标签: {short_count}")

    print("\n" + "=" * 70)
    print("  Phase 2 Feature Engineering 完成")
    print("=" * 70)


if __name__ == "__main__":
    main()