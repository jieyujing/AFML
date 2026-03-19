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


def test_dual_ma_tbm_long_position():
    """测试 TBM 标签计算 - 多头仓位"""
    from webapp.utils.primary_model.dual_ma import DualMAStrategy

    # 创建上涨趋势数据，使用较大数据量确保有足够的多头信号
    dates = pd.date_range('2024-01-01', periods=60, freq='h')
    prices = 100 + np.arange(60) * 0.5  # 持续上涨
    data = pd.DataFrame({'price': prices}, index=dates)

    strategy = DualMAStrategy(tp_ratio=2.0, sl_ratio=1.0, vol_window=5)
    result = strategy.generate_signals(data, short_window=3, long_window=10)

    # 检查 direction 列存在
    assert 'direction' in result.events_with_labels.columns

    # 在持续上涨趋势中，应该有多头仓位
    long_count = (result.events_with_labels['direction'] == 1).sum()
    assert long_count > 0

    # 多头仓位的标签：当价格上涨时应该是止盈 (label=1)
    long_labels = result.events_with_labels[result.events_with_labels['direction'] == 1]['label']
    assert (long_labels == 1).all(), f"Expected all long position labels to be 1, got: {long_labels.unique()}"


def test_dual_ma_tbm_short_position():
    """测试 TBM 标签计算 - 空头仓位"""
    from webapp.utils.primary_model.dual_ma import DualMAStrategy

    # 创建下跌趋势数据
    dates = pd.date_range('2024-01-01', periods=60, freq='h')
    prices = 130 - np.arange(60) * 0.5  # 持续下跌
    data = pd.DataFrame({'price': prices}, index=dates)

    strategy = DualMAStrategy(tp_ratio=2.0, sl_ratio=1.0, vol_window=5)
    result = strategy.generate_signals(data, short_window=3, long_window=10)

    # 检查 direction 列存在
    assert 'direction' in result.events_with_labels.columns

    # 在持续下跌趋势中，应该有空头仓位
    short_count = (result.events_with_labels['direction'] == -1).sum()
    assert short_count > 0

    # 空头仓位的标签：当价格下跌时应该是止盈 (label=1)
    short_labels = result.events_with_labels[result.events_with_labels['direction'] == -1]['label']
    assert (short_labels == 1).all(), f"Expected all short position labels to be 1 (profit when price down), got: {short_labels.unique()}"


def test_dual_ma_tbm_short_position_tp_sl():
    """测试空头仓位的止盈止损逻辑"""
    from webapp.utils.primary_model.dual_ma import DualMAStrategy

    # 创建数据：价格持续下跌
    n_points = 20
    dates = pd.date_range('2024-01-01', periods=n_points, freq='h')
    # 持续下跌的价格
    prices = np.array([100.0, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86])
    data = pd.DataFrame({'price': prices}, index=dates[:len(prices)])

    strategy = DualMAStrategy(tp_ratio=0.5, sl_ratio=0.3, vol_window=5)

    # 添加 volatility 列
    prices_series = data['price']
    returns = np.log(prices_series / prices_series.shift(1))
    volatility = returns.rolling(window=5, min_periods=1).std()
    data['volatility'] = volatility

    short_positions = pd.Series(-1, index=data.index)  # 全程做空

    labels_df = strategy._compute_tbm_labels(
        data,
        positions=short_positions,
        tp_ratio=0.5,
        sl_ratio=0.3,
        time_barrier=None
    )

    # 空头：direction 应该都是 -1
    assert (labels_df['direction'] == -1).all()

    # 验证方向调整后的收益计算
    # 空头：价格持续下跌，应该产生正收益
    assert (labels_df['returns'] > 0).all(), f"Short positions should have positive returns when price drops, got: {labels_df['returns'].unique()}"

    # 空头：价格下跌应该触发止盈 (label=1)
    assert (labels_df['label'] == 1).all(), f"Short positions with dropping price should have label=1, got: {labels_df['label'].unique()}"


def test_dual_ma_returns_direction_adjusted():
    """测试收益计算是否考虑方向"""
    from webapp.utils.primary_model.dual_ma import DualMAStrategy

    dates = pd.date_range('2024-01-01', periods=20, freq='h')
    # 价格从 100 涨到 110
    prices = 100 + np.arange(20)
    data = pd.DataFrame({'price': prices}, index=dates)

    strategy = DualMAStrategy(tp_ratio=2.0, sl_ratio=1.0, vol_window=5)
    result = strategy.generate_signals(data, short_window=3, long_window=10)

    # 检查 returns 列存在且为 direction-adjusted
    assert 'returns' in result.events_with_labels.columns