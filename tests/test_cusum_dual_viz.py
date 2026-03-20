"""CUSUM 双层可视化测试"""
import sys
from pathlib import Path

# Add project root to path for scripts import
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import pandas as pd
from scripts.cusum_filtering import compute_dynamic_cusum_filter


@pytest.fixture
def sample_df():
    """创建测试数据"""
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2024-01-01', periods=n, freq='h')
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        'close': prices,
        'volume': np.random.randint(100, 1000, n)
    }, index=dates)


class TestReturnState:
    """测试 return_state 参数"""

    def test_return_state_false_returns_two_values(self, sample_df):
        """return_state=False 返回两个值"""
        result = compute_dynamic_cusum_filter(
            sample_df, return_state=False
        )
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], pd.DatetimeIndex)

    def test_return_state_true_returns_three_values(self, sample_df):
        """return_state=True 返回三个值"""
        result = compute_dynamic_cusum_filter(
            sample_df, return_state=True
        )
        assert len(result) == 3
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], pd.DatetimeIndex)
        assert isinstance(result[2], dict)

    def test_cusum_state_structure(self, sample_df):
        """测试 cusum_state 结构"""
        _, _, cusum_state = compute_dynamic_cusum_filter(
            sample_df, return_state=True
        )
        required_keys = ['s_pos', 's_neg', 'threshold', 'time_index']
        for key in required_keys:
            assert key in cusum_state, f"Missing key: {key}"

    def test_s_pos_initialized(self, sample_df):
        """测试 s_pos[0] 已初始化为 0"""
        _, _, cusum_state = compute_dynamic_cusum_filter(
            sample_df, return_state=True
        )
        assert cusum_state['s_pos'][0] == 0.0
        assert cusum_state['s_neg'][0] == 0.0

    def test_threshold_is_positive(self, sample_df):
        """测试阈值为正数"""
        _, _, cusum_state = compute_dynamic_cusum_filter(
            sample_df, return_state=True
        )
        assert cusum_state['threshold'] > 0

    def test_time_index_alignment(self, sample_df):
        """测试时间索引与状态数组对齐"""
        _, _, cusum_state = compute_dynamic_cusum_filter(
            sample_df, return_state=True
        )
        assert len(cusum_state['s_pos']) == len(cusum_state['time_index'])
        assert len(cusum_state['s_neg']) == len(cusum_state['time_index'])