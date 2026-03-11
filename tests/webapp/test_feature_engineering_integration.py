# tests/webapp/test_feature_engineering_integration.py
"""特征工程集成测试 - 验证完整流程"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from webapp.utils.feature_calculator import compute_all_features


@pytest.fixture
def sample_dollar_bars():
    """生成模拟 Dollar Bars 数据"""
    np.random.seed(42)
    n = 500

    dates = pd.date_range('2023-01-01', periods=n, freq='5min')

    # 随机游走生成价格
    returns = np.random.randn(n) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))

    # 生成 OHLCV
    data = {
        'timestamp': dates,
        'open': prices * (1 + np.random.randn(n) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n) * 0.002)),
        'low': prices * (1 - np.abs(np.random.randn(n) * 0.002)),
        'close': prices,
        'volume': np.random.exponential(1000, n)
    }

    df = pd.DataFrame(data)
    df = df.set_index('timestamp')
    return df


@pytest.fixture
def sample_cusum_labels(sample_dollar_bars):
    """生成模拟 CUSUM 标签

    注意：CUSUM 标签需要以 timestamp 作为索引列（第一列），
    因为 align_features_with_cusum 使用 index_col=0 读取 CSV。
    """
    # 选择较晚的时间戳作为事件（避免被 EWM/FFD 等滤波器丢弃）
    # 从索引 100 开始，确保有足够的历史数据计算滞后特征
    event_idx = np.arange(100, len(sample_dollar_bars) - 50, 5)
    events = sample_dollar_bars.iloc[event_idx].copy()

    # 创建以 timestamp 为索引的 DataFrame
    labels = pd.DataFrame({
        'bin': np.random.choice([-1, 0, 1], len(events)),
        't1': events.index + pd.Timedelta(minutes=30),
        'avg_uniqueness': np.random.randn(len(events)),
        'return_attribution': np.random.randn(len(events)) * 0.01
    }, index=events.index)
    labels.index.name = 'timestamp'

    return labels


def test_compute_features_without_alignment(sample_dollar_bars):
    """测试不与其他对齐的特征计算"""
    config = {
        'volatility': {'spans': [10, 50]},
        'momentum': {'rsi_window': 14, 'roc_period': 10},
        'fractional_diff': {'enabled': True, 'threshold': 1e-4, 'd_step': 0.05}
    }

    result, metadata = compute_all_features(
        df=sample_dollar_bars,
        config=config,
        align_to_cusum=False
    )

    # 验证特征列存在
    assert 'vol_ewm_10' in result.columns
    assert 'vol_ewm_50' in result.columns
    assert 'rsi_14' in result.columns
    assert 'ffd_log_price' in result.columns

    # 验证元数据
    assert 'optimal_d' in metadata
    assert metadata['rows_after_clean'] <= metadata['rows_before_clean']
    assert not metadata['aligned_to_cusum']


def test_compute_features_with_cusum_alignment(
    sample_dollar_bars,
    tmp_path
):
    """测试与 CUSUM 事件对齐的特征计算"""
    # 先计算特征，获取有效的时间戳索引
    config = {
        'volatility': {'spans': [10]},
        'fractional_diff': {'enabled': True, 'threshold': 1e-4, 'd_step': 0.05}
    }

    features_result, _ = compute_all_features(
        df=sample_dollar_bars,
        config=config,
        align_to_cusum=False
    )

    # 从有效索引中选择部分作为 CUSUM 事件（跳过前面的滞后行）
    valid_indices = features_result.index[50:-10]  # 跳过开头和结尾
    event_timestamps = valid_indices[np.arange(0, len(valid_indices), 5)]

    # 创建 CUSUM 标签，只包含特征计算后保留的时间戳
    sample_cusum_labels = pd.DataFrame({
        'bin': np.random.choice([-1, 0, 1], len(event_timestamps)),
        't1': event_timestamps + pd.Timedelta(minutes=30),
        'avg_uniqueness': np.random.randn(len(event_timestamps)),
        'return_attribution': np.random.randn(len(event_timestamps)) * 0.01
    }, index=event_timestamps)
    sample_cusum_labels.index.name = 'timestamp'

    # 保存 CUSUM 标签到临时文件
    # 注意：必须保存索引，因为 align_features_with_cusum 使用 index_col=0 读取
    cusum_path = tmp_path / "cusum_labels.csv"
    sample_cusum_labels.to_csv(cusum_path, index=True)

    config = {
        'volatility': {'spans': [10]},
        'momentum': {'rsi_window': 14, 'roc_period': 10},
        'fractional_diff': {'enabled': True, 'threshold': 1e-4, 'd_step': 0.05},
        'cusum': {'align_enabled': True, 'path': str(cusum_path)}
    }

    result, metadata = compute_all_features(
        df=sample_dollar_bars,
        config=config,
        cusum_path=str(cusum_path),
        align_to_cusum=True
    )

    # 验证对齐后的行数与 CUSUM 事件数一致
    assert len(result) == len(sample_cusum_labels)

    # 验证标签列存在
    assert 'bin' in result.columns
    assert 'avg_uniqueness' in result.columns

    # 验证元数据
    assert metadata['aligned_to_cusum']


def test_feature_calculator_with_missing_columns(sample_dollar_bars):
    """测试缺失某些列时的容错处理"""
    # 移除 high/low 列
    df = sample_dollar_bars.drop(columns=['high', 'low'])

    config = {}

    # 应该不抛出异常，只是跳过需要 high/low 的特征
    result, metadata = compute_all_features(df=df, config=config, align_to_cusum=False)

    # 验证不依赖 high/low 的特征存在
    assert 'vol_ewm_10' in result.columns
    assert 'rsi_14' in result.columns
    assert 'ffd_log_price' in result.columns

    # 验证依赖 high/low 的特征不存在或为 NaN
    assert 'vol_parkinson' not in result.columns
    assert 'mom_stoch_k_14' not in result.columns


def test_compute_all_features_with_alpha158(sample_dollar_bars):
    """Test compute_all_features with Alpha158 enabled."""
    config = {
        'volatility': {'spans': [10]},
        'alpha158': {
            'enabled': True,
            'volatility': {'spans': [5, 10]},
            'ma': {'windows': [5, 10]},
            'rank': {'enabled': True, 'window': 20}
        }
    }

    result, metadata = compute_all_features(
        df=sample_dollar_bars,
        config=config,
        align_to_cusum=False
    )

    # Check existing features
    assert 'vol_ewm_10' in result.columns

    # Check Alpha158 features
    assert 'ffd_ma_5' in result.columns
    assert 'ffd_vol_std_5' in result.columns
    assert 'ffd_rank_ffd_ma_5_20' in result.columns

    # Check metadata
    assert metadata.get('alpha158_enabled') is True
    assert 'alpha158_optimal_d' in metadata
