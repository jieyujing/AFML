# tests/webapp/test_optimizer.py

import pandas as pd
import numpy as np
import pytest


def make_test_data(n_points: int = 200) -> pd.DataFrame:
    """创建测试数据"""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(n_points) * 0.5)
    dates = pd.date_range('2024-01-01', periods=n_points, freq='h')
    return pd.DataFrame({'price': prices}, index=dates)


def test_get_splits():
    """测试分割索引生成"""
    from webapp.utils.primary_model.optimizer import WalkForwardOptimizer

    optimizer = WalkForwardOptimizer(
        train_size=100,
        test_size=30,
        embargo=5
    )

    splits = optimizer.get_splits(200)

    # 应该生成至少一个分割
    assert len(splits) >= 1

    # 检查分割结构
    train_idx, test_idx = splits[0]
    assert len(train_idx) == 100
    assert len(test_idx) == 30

    # 检查 embargo
    assert test_idx[0] - train_idx[-1] == 6  # 5 + 1


def test_get_splits_insufficient_data():
    """测试数据不足时的行为"""
    from webapp.utils.primary_model.optimizer import WalkForwardOptimizer

    optimizer = WalkForwardOptimizer(
        train_size=100,
        test_size=30,
        embargo=5
    )

    splits = optimizer.get_splits(50)  # 数据不足

    assert len(splits) == 0


def test_optimize():
    """测试优化流程"""
    from webapp.utils.primary_model.optimizer import WalkForwardOptimizer
    from webapp.utils.primary_model.dual_ma import DualMAStrategy

    data = make_test_data(200)

    strategy = DualMAStrategy(
        short_range=(3, 5),
        long_range=(8, 12),
        step=1,
        tp_ratio=2.0,
        sl_ratio=1.0,
        vol_window=10
    )

    optimizer = WalkForwardOptimizer(
        train_size=50,
        test_size=20,
        embargo=3
    )

    result = optimizer.optimize(data, strategy)

    # 检查返回类型
    assert hasattr(result, 'best_params')
    assert hasattr(result, 'best_score')
    assert hasattr(result, 'all_results')

    # 检查最优参数
    assert 'short_window' in result.best_params
    assert 'long_window' in result.best_params
    assert result.best_params['long_window'] > result.best_params['short_window']

    # 检查分数范围
    assert 0.0 <= result.best_score <= 1.0

    # 检查结果 DataFrame
    assert len(result.all_results) > 0
    assert 'fold' in result.all_results.columns
    assert 'test_recall' in result.all_results.columns


def test_optimize_insufficient_data_raises():
    """测试数据不足时抛出异常"""
    from webapp.utils.primary_model.optimizer import WalkForwardOptimizer
    from webapp.utils.primary_model.dual_ma import DualMAStrategy

    data = make_test_data(50)  # 数据不足

    strategy = DualMAStrategy(
        short_range=(3, 5),
        long_range=(8, 12),
        step=1
    )

    optimizer = WalkForwardOptimizer(
        train_size=100,
        test_size=30,
        embargo=5
    )

    with pytest.raises(ValueError, match="数据量不足"):
        optimizer.optimize(data, strategy)


def test_optimize_empty_param_grid_raises():
    """测试空参数网格时抛出异常"""
    from webapp.utils.primary_model.optimizer import WalkForwardOptimizer
    from webapp.utils.primary_model.base import PrimaryModelBase

    data = make_test_data(200)

    # 创建返回空参数网格的策略
    class EmptyGridStrategy(PrimaryModelBase):
        @property
        def name(self):
            return "Empty"

        @property
        def param_grid(self):
            return {}

        def generate_signals(self, data, **params):
            pass

    strategy = EmptyGridStrategy()
    optimizer = WalkForwardOptimizer(train_size=50, test_size=20, embargo=3)

    with pytest.raises(ValueError, match="参数网格为空"):
        optimizer.optimize(data, strategy)