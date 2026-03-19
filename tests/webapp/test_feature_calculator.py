# tests/webapp/test_feature_calculator.py
import numpy as np
import pandas as pd
import pytest
from webapp.utils.feature_calculator import (
    compute_log_returns,
    compute_volatility_features,
    compute_momentum_features,
    compute_fracdiff_features,
)


def test_compute_log_returns():
    """测试对数收益率计算"""
    df = pd.DataFrame({'close': [100.0, 101.0, 102.0, 103.0]})
    result = compute_log_returns(df)
    assert 'log_return' in result.columns
    assert pd.isna(result['log_return'].iloc[0])
    assert not pd.isna(result['log_return'].iloc[1:]).all()


def test_compute_volatility_features():
    """测试波动率特征计算"""
    df = pd.DataFrame({
        'close': np.linspace(100, 110, 100),
        'high': np.linspace(101, 111, 100),
        'low': np.linspace(99, 109, 100),
        'log_return': np.random.randn(100) * 0.01
    })
    result = compute_volatility_features(df, spans=[10, 50])
    assert 'vol_ewm_10' in result.columns
    assert 'vol_ewm_50' in result.columns
    assert 'vol_parkinson' in result.columns
    assert 'vol_atr_14' in result.columns
    assert 'vol_bb_pct_b_20' in result.columns
    assert 'trend_variance_ratio_20' in result.columns


def test_compute_momentum_features():
    """测试动量特征计算"""
    df = pd.DataFrame({
        'close': np.linspace(100, 110, 100),
        'high': np.linspace(101, 111, 100),
        'low': np.linspace(99, 109, 100),
        'volume': np.random.randint(1000, 2000, 100)
    })
    # 先计算对数收益率，以便 liq_amihud 可以使用
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    result = compute_momentum_features(df)
    assert 'ema_short' in result.columns
    assert 'ema_long' in result.columns
    assert 'rsi_14' in result.columns
    assert 'mom_roc_10' in result.columns
    assert 'mom_stoch_k_14' in result.columns
    assert 'corr_pv_10' in result.columns
    assert 'liq_amihud' in result.columns
    assert 'vol_rel_20' in result.columns


def test_compute_fracdiff_features():
    """测试分数阶差分特征计算"""
    # 生成更长的稳定随机游走序列以避免 ADF 测试中的 NaN 问题
    np.random.seed(42)
    returns = np.random.randn(500) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({'close': prices})
    result, optimal_d = compute_fracdiff_features(df, thres=1e-4, d_step=0.05)
    assert 'ffd_log_price' in result.columns
    assert 0.0 <= optimal_d <= 1.0
