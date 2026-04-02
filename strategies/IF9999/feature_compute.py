"""
feature_compute.py - 特征计算函数集合

按 AFML 方法论分层组织特征计算：
- L1: 基础特征（动量、波动率、趋势、均值回复、时间）
- L2: 高级特征（微观结构、均线交叉、熵）
- L3: 结构特征（序列相关、结构突变）
- Trend Scan: 标签特征（含未来信息）
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Dict, Any

# afmlkit 核心函数导入
from afmlkit.feature.core.momentum import roc, rsi_wilder, stoch_k
from afmlkit.feature.core.volatility import ewms
from afmlkit.feature.core.trend import adx_core
from afmlkit.feature.core.reversion import vwap_distance
from afmlkit.feature.core.time import time_cues
from afmlkit.feature.core.microstructure import (
    amihud_illiquidity,
    rolling_corwin_schultz_spread,
    high_low_volatility,
)
from afmlkit.feature.core.cross_ma import cross_ma_ratio, cross_ma_signal
from afmlkit.feature.core.entropy import rolling_entropy
from afmlkit.feature.core.serial_corr import (
    rolling_serial_correlation,
    ljung_box_statistic,
)
from afmlkit.feature.core.structural_break import (
    adf_test_rolling,
    sadf_test,
)
from afmlkit.feature.core.trend_scan import trend_scan_labels
from afmlkit.feature.core.theil_imbalance import (
    clv_split,
    bvc_split,
    direction_split,
    rolling_theil_imbalance,
    rolling_theil_decomposed,
)


# ============================================================
# L1: 基础特征
# ============================================================

def compute_momentum_features(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    计算动量特征：RSI, ROC, Stoch_K

    :param bars: OHLCV DataFrame
    :param features: 特征 DataFrame（已存在）
    :param config: momentum 配置
    :returns: 更新后的特征 DataFrame
    """
    close = bars['close'].values.astype(np.float64)
    low = bars['low'].values.astype(np.float64)
    high = bars['high'].values.astype(np.float64)

    # RSI
    for window in config.get('rsi_windows', [14]):
        rsi_vals = rsi_wilder(close, window)
        features[f'feat_rsi_{window}'] = pd.Series(rsi_vals, index=bars.index)

    # ROC
    for period in config.get('roc_periods', [10]):
        roc_vals = roc(close, period)
        features[f'feat_roc_{period}'] = pd.Series(roc_vals, index=bars.index)

    # Stoch K
    stoch_k_len = config.get('stoch_k_length', 14)
    stoch_k_vals = stoch_k(close, low, high, stoch_k_len)
    features[f'feat_stoch_k_{stoch_k_len}'] = pd.Series(stoch_k_vals, index=bars.index)

    return features


def compute_volatility_features(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    计算波动率特征：EWM_STD, High-Low Vol

    :param bars: OHLCV DataFrame
    :param features: 特征 DataFrame
    :param config: volatility 配置
    :returns: 更新后的特征 DataFrame
    """
    close = bars['close'].values.astype(np.float64)
    low = bars['low'].values.astype(np.float64)
    high = bars['high'].values.astype(np.float64)

    # 计算收益率用于波动率
    returns = np.empty_like(close)
    returns[0] = np.nan
    returns[1:] = np.log(close[1:] / close[:-1])

    # EWM STD
    for span in config.get('ewm_spans', [20]):
        ewm_std_vals = ewms(returns, span)
        features[f'feat_ewm_vol_{span}'] = pd.Series(ewm_std_vals, index=bars.index)

    # High-Low Volatility (Parkinson)
    hl_windows = config.get('hl_windows', [20])
    for window in hl_windows:
        hl_vol_vals = high_low_volatility(high, low, window)
        features[f'feat_hl_vol_{window}'] = pd.Series(hl_vol_vals, index=bars.index)

    return features


def compute_trend_features(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    计算趋势特征：ADX

    :param bars: OHLCV DataFrame
    :param features: 特征 DataFrame
    :param config: trend 配置
    :returns: 更新后的特征 DataFrame
    """
    close = bars['close'].values.astype(np.float64)
    low = bars['low'].values.astype(np.float64)
    high = bars['high'].values.astype(np.float64)

    length = config.get('adx_length', 14)

    # ADX 只返回单个数组
    adx_vals = adx_core(high, low, close, length)
    features[f'feat_adx_{length}'] = pd.Series(adx_vals, index=bars.index)

    return features


def compute_reversion_features(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    计算均值回复特征：VWAP Distance

    :param bars: OHLCV DataFrame
    :param features: 特征 DataFrame
    :param config: reversion 配置
    :returns: 更新后的特征 DataFrame
    """
    close = bars['close'].values.astype(np.float64)
    volume = bars['volume'].values.astype(np.float64)

    for window in config.get('vwap_windows', [20]):
        vwap_dist_vals = vwap_distance(close, volume, window, is_log=False)
        features[f'feat_vwap_dist_{window}'] = pd.Series(vwap_dist_vals, index=bars.index)

    return features


def compute_time_features(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    计算时间特征：Time Cues

    :param bars: OHLCV DataFrame（必须有 datetime index）
    :param features: 特征 DataFrame
    :param config: time_cues 配置
    :returns: 更新后的特征 DataFrame
    """
    if not config.get('enabled', True):
        return features

    # 提取纳秒时间戳
    timestamps = bars.index.values.astype(np.int64)

    # time_cues 返回多个数组
    sin_td, cos_td, sin_dw, cos_dw, asia, eu, us, trans, top_hr = time_cues(timestamps)

    features['feat_sin_time'] = pd.Series(sin_td, index=bars.index)
    features['feat_cos_time'] = pd.Series(cos_td, index=bars.index)
    features['feat_sin_dow'] = pd.Series(sin_dw, index=bars.index)
    features['feat_cos_dow'] = pd.Series(cos_dw, index=bars.index)
    features['feat_asia_sess'] = pd.Series(asia, index=bars.index).astype(int)
    features['feat_eu_sess'] = pd.Series(eu, index=bars.index).astype(int)
    features['feat_us_sess'] = pd.Series(us, index=bars.index).astype(int)

    return features


# ============================================================
# L2: 高级特征
# ============================================================

def compute_theil_imbalance_features(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    计算 Theil Imbalance 特征：买卖集中度不对称性

    Theil Index 用 KL 散度度量成交量在时间 bar 上的集中度：
    - T_buy: 买方成交量集中度
    - T_sell: 卖方成交量集中度
    - T_ratio = T_buy / (T_buy + T_sell)

    解释:
        T_ratio > 0.5: 买方更集中 (机构买入特征)
        T_ratio < 0.5: 卖方更集中 (机构卖出特征)
        T_ratio ≈ 0.5: 对称

    :param bars: OHLCV DataFrame
    :param features: 特征 DataFrame
    :param config: theil_imbalance 配置
    :returns: 更新后的特征 DataFrame
    """
    close = bars['close'].values.astype(np.float64)
    low = bars['low'].values.astype(np.float64)
    high = bars['high'].values.astype(np.float64)
    open_ = bars['open'].values.astype(np.float64)
    volume = bars['volume'].values.astype(np.float64)

    split_method = config.get('split_method', 'clv')
    decomposed = config.get('decomposed', False)

    # 分拆成交量
    if split_method == 'clv':
        buy_vol, sell_vol = clv_split(close, low, high, volume)
    elif split_method == 'bvc':
        buy_vol, sell_vol = bvc_split(close, volume)
    elif split_method == 'direction':
        buy_vol, sell_vol = direction_split(close, open_, volume)
    else:
        raise ValueError(f"Unknown split_method: {split_method}")

    for window in config.get('windows', [20]):
        if decomposed:
            # 完整分解
            t_buy, t_sell, t_ratio, t_diff = rolling_theil_decomposed(
                buy_vol, sell_vol, window
            )
            features[f'feat_theil_buy_{window}'] = pd.Series(t_buy, index=bars.index)
            features[f'feat_theil_sell_{window}'] = pd.Series(t_sell, index=bars.index)
            features[f'feat_theil_ratio_{window}'] = pd.Series(t_ratio, index=bars.index)
            features[f'feat_theil_diff_{window}'] = pd.Series(t_diff, index=bars.index)
        else:
            # 仅计算 T_ratio
            t_ratio = rolling_theil_imbalance(buy_vol, sell_vol, window)
            features[f'feat_theil_imb_{window}'] = pd.Series(t_ratio, index=bars.index)

    return features


def compute_microstructure_features(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    计算微观结构特征：Amihud, Corwin-Schultz, High-Low Vol

    :param bars: OHLCV DataFrame
    :param features: 特征 DataFrame
    :param config: microstructure 配置
    :returns: 更新后的特征 DataFrame
    """
    close = bars['close'].values.astype(np.float64)
    low = bars['low'].values.astype(np.float64)
    high = bars['high'].values.astype(np.float64)
    volume = bars['volume'].values.astype(np.float64)

    # 计算收益率
    returns = np.empty_like(close)
    returns[0] = np.nan
    returns[1:] = (close[1:] - close[:-1]) / close[:-1]

    # Dollar Volume
    dollar_volume = close * volume

    for window in config.get('windows', [20]):
        # Amihud Illiquidity
        amihud_vals = amihud_illiquidity(returns, dollar_volume, window)
        features[f'feat_amihud_{window}'] = pd.Series(amihud_vals, index=bars.index)

        # Corwin-Schultz Spread
        cs_spread_vals = rolling_corwin_schultz_spread(high, low, window)
        features[f'feat_cs_spread_{window}'] = pd.Series(cs_spread_vals, index=bars.index)

    return features


def compute_cross_ma_features(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    计算均线交叉特征：Cross MA Ratio, Signal

    :param bars: OHLCV DataFrame
    :param features: 特征 DataFrame
    :param config: cross_ma 配置
    :returns: 更新后的特征 DataFrame
    """
    close = bars['close'].values.astype(np.float64)

    for fast, slow in config.get('fast_slow_pairs', [(5, 20)]):
        # Cross MA Ratio
        ratio_vals = cross_ma_ratio(close, fast, slow)
        features[f'feat_cross_ma_ratio_{fast}_{slow}'] = pd.Series(ratio_vals, index=bars.index)

        # Cross MA Signal
        signal_vals = cross_ma_signal(close, fast, slow)
        features[f'feat_cross_ma_sig_{fast}_{slow}'] = pd.Series(signal_vals, index=bars.index)

    return features


def compute_entropy_features(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    计算熵特征：Shannon Entropy, LZ Entropy

    :param bars: OHLCV DataFrame
    :param features: 特征 DataFrame
    :param config: entropy 配置
    :returns: 更新后的特征 DataFrame
    """
    close = bars['close'].values.astype(np.float64)

    # 计算收益率
    returns = np.empty_like(close)
    returns[0] = np.nan
    returns[1:] = np.log(close[1:] / close[:-1])
    returns = returns[1:]  # 去掉第一个 NaN

    n_bins = config.get('n_bins', 3)

    for window in config.get('windows', [50]):
        # 使用 rolling_entropy 函数
        # method: 0=Shannon, 1=LZ, 2=Kontoyiannis
        shannon_vals = rolling_entropy(returns, window, n_bins=n_bins, method=0)
        lz_vals = rolling_entropy(returns, window, n_bins=n_bins, method=1)

        features[f'feat_shannon_{window}'] = pd.Series(shannon_vals, index=bars.index[1:])
        features[f'feat_lz_entropy_{window}'] = pd.Series(lz_vals, index=bars.index[1:])

    return features


# ============================================================
# L3: 结构特征
# ============================================================

def compute_serial_corr_features(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    计算序列相关特征：Rolling Serial Corr, Ljung-Box

    :param bars: OHLCV DataFrame
    :param features: 特征 DataFrame
    :param config: serial_corr 配置
    :returns: 更新后的特征 DataFrame
    """
    close = bars['close'].values.astype(np.float64)

    # 计算收益率
    returns = np.empty_like(close)
    returns[0] = np.nan
    returns[1:] = np.log(close[1:] / close[:-1])
    returns = returns[1:]

    window = config.get('ljung_box_window', 20)
    lags_arr = np.array(config.get('lags', [1, 5, 10]), dtype=np.int64)

    # Rolling Serial Correlation - 返回 2D array (n, len(lags))
    serial_corr_vals = rolling_serial_correlation(returns, window, lags_arr)

    for i, lag in enumerate(config.get('lags', [1, 5])):
        features[f'feat_serial_corr_lag{lag}_{window}'] = pd.Series(
            serial_corr_vals[:, i], index=bars.index[1:]
        )

    # Ljung-Box Statistic - 需要滚动计算
    n = len(returns)
    lb_vals = np.full(n, np.nan)
    for i in range(window, n):
        window_data = returns[i - window + 1 : i + 1]
        lb_vals[i] = ljung_box_statistic(window_data, lags_arr)

    features[f'feat_ljung_box_{window}'] = pd.Series(lb_vals, index=bars.index[1:])

    return features


def compute_structural_break_features(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    计算结构突变特征：ADF Rolling, SADF

    :param bars: OHLCV DataFrame
    :param features: 特征 DataFrame
    :param config: structural_break 配置
    :returns: 更新后的特征 DataFrame
    """
    close = bars['close'].values.astype(np.float64)

    # ADF Rolling Test
    adf_window = config.get('adf_window', 100)
    adf_vals = adf_test_rolling(close, adf_window)
    features[f'feat_adf_test_{adf_window}'] = pd.Series(adf_vals, index=bars.index)

    # SADF Test
    min_window = config.get('sadf_min_window', 10)
    max_window = config.get('sadf_max_window', 60)
    sadf_vals = sadf_test(close, min_window=min_window, max_window=max_window)
    features[f'feat_sadf_{min_window}_{max_window}'] = pd.Series(sadf_vals, index=bars.index)

    return features


# ============================================================
# Trend Scan 标签（含未来信息）
# ============================================================

def compute_trend_scan_labels(
    bars: pd.DataFrame,
    event_indices: NDArray[np.int64],
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    计算 Trend Scan 标签（含未来信息）

    注意：此函数包含未来信息，仅用于标签生成，
    不可用于预测特征。

    :param bars: OHLCV DataFrame
    :param event_indices: CUSUM 事件点索引
    :param config: trend_scan 配置
    :returns: Trend Scan 标签 DataFrame
    """
    # trend_scan_labels 需要 pd.Series 和 pd.DatetimeIndex
    price_series = pd.Series(bars['close'].values, index=bars.index, name='close')
    event_timestamps = bars.index[event_indices]
    t_events = pd.DatetimeIndex(event_timestamps)

    windows = config.get('windows', [5, 10, 20, 30, 50])

    # 调用 trend_scan_labels
    result_df = trend_scan_labels(price_series, t_events, windows)

    return result_df


# ============================================================
# 综合特征计算
# ============================================================

def compute_all_features(
    bars: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    计算所有特征（不含 Trend Scan）

    :param bars: OHLCV DataFrame
    :param config: FEATURE_CONFIG
    :returns: 特征 DataFrame（宽表）
    """
    features = pd.DataFrame(index=bars.index)

    # L1: 基础特征
    if config.get('momentum', {}).get('enabled', False):
        features = compute_momentum_features(bars, features, config['momentum'])

    if config.get('volatility', {}).get('enabled', False):
        features = compute_volatility_features(bars, features, config['volatility'])

    if config.get('trend', {}).get('enabled', False):
        features = compute_trend_features(bars, features, config['trend'])

    if config.get('reversion', {}).get('enabled', False):
        features = compute_reversion_features(bars, features, config['reversion'])

    if config.get('time_cues', {}).get('enabled', False):
        features = compute_time_features(bars, features, config['time_cues'])

    # L2: 高级特征
    if config.get('theil_imbalance', {}).get('enabled', False):
        features = compute_theil_imbalance_features(bars, features, config['theil_imbalance'])

    if config.get('microstructure', {}).get('enabled', False):
        features = compute_microstructure_features(bars, features, config['microstructure'])

    if config.get('cross_ma', {}).get('enabled', False):
        features = compute_cross_ma_features(bars, features, config['cross_ma'])

    if config.get('entropy', {}).get('enabled', False):
        features = compute_entropy_features(bars, features, config['entropy'])

    # L3: 结构特征
    if config.get('serial_corr', {}).get('enabled', False):
        features = compute_serial_corr_features(bars, features, config['serial_corr'])

    if config.get('structural_break', {}).get('enabled', False):
        features = compute_structural_break_features(bars, features, config['structural_break'])

    return features


def compute_event_features(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    event_indices: NDArray[np.int64]
) -> pd.DataFrame:
    """
    从全量特征中提取事件点特征

    :param bars: OHLCV DataFrame
    :param features: 全量特征 DataFrame
    :param event_indices: CUSUM 事件点索引
    :returns: 事件点特征 DataFrame（宽表）
    """
    event_timestamps = bars.index[event_indices]
    event_features = features.loc[event_timestamps].copy()

    # 添加元信息
    event_features['event_idx'] = event_indices
    event_features['price'] = bars['close'].iloc[event_indices].values

    return event_features