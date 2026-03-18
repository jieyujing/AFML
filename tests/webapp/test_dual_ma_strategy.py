import pandas as pd
import numpy as np
import pytest


def make_test_data(n_points: int = 50) -> pd.DataFrame:
    """创建测试数据"""
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(n_points) * 0.5)
    dates = pd.date_range('2024-01-01', periods=n_points, freq='h')
    return pd.DataFrame({'price': prices}, index=dates)


def test_dual_ma_strategy_name():
    """测试策略名称"""
    from webapp.utils.primary_model.dual_ma import DualMAStrategy

    strategy = DualMAStrategy()
    assert strategy.name == "双均线策略"


def test_dual_ma_param_grid():
    """测试参数网格生成"""
    from webapp.utils.primary_model.dual_ma import DualMAStrategy

    strategy = DualMAStrategy(
        short_range=(3, 5),
        long_range=(6, 10),
        step=1
    )

    grid = strategy.param_grid

    # 确保所有组合 long > short
    for s, l in zip(grid['short_window'], grid['long_window']):
        assert l > s


def test_dual_ma_generate_signals():
    """测试信号生成"""
    from webapp.utils.primary_model.dual_ma import DualMAStrategy

    data = make_test_data(50)
    strategy = DualMAStrategy(
        tp_ratio=2.0,
        sl_ratio=1.0,
        vol_window=5
    )

    result = strategy.generate_signals(data, short_window=3, long_window=10)

    # 检查返回类型
    assert hasattr(result, 'signals')
    assert hasattr(result, 'positions')
    assert hasattr(result, 'events_with_labels')

    # 检查信号范围
    assert set(result.signals.unique()).issubset({-1, 0, 1})

    # 检查位置范围
    assert set(result.positions.unique()).issubset({-1, 1})

    # 检查 TBM 标签
    assert 'label' in result.events_with_labels.columns
    assert set(result.events_with_labels['label'].unique()).issubset({-1, 0, 1})


def test_dual_ma_tbm_labels():
    """测试 TBM 标签计算"""
    from webapp.utils.primary_model.dual_ma import DualMAStrategy

    # 创建趋势数据（持续上涨）
    dates = pd.date_range('2024-01-01', periods=30, freq='h')
    prices = 100 + np.arange(30) * 0.5  # 持续上涨
    data = pd.DataFrame({'price': prices}, index=dates)

    strategy = DualMAStrategy(tp_ratio=2.0, sl_ratio=1.0, vol_window=5)
    result = strategy.generate_signals(data, short_window=3, long_window=10)

    # 在持续上涨的趋势中，大部分标签应该是 1（止盈）
    positive_labels = (result.events_with_labels['label'] == 1).sum()
    assert positive_labels > len(result.events_with_labels) * 0.5


def test_dual_ma_recall_calculation():
    """测试 Recall 计算（集成测试）"""
    from webapp.utils.primary_model.dual_ma import DualMAStrategy

    data = make_test_data(50)
    strategy = DualMAStrategy(tp_ratio=2.0, sl_ratio=1.0, vol_window=5)

    result = strategy.generate_signals(data, short_window=3, long_window=10)

    # 计算 recall
    recall = strategy.evaluate(
        result.signals.loc[result.events_with_labels.index],
        result.events_with_labels['label']
    )

    # Recall 应该在 0-1 之间
    assert 0.0 <= recall <= 1.0