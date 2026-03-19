import pandas as pd
import numpy as np
import pytest


def test_optimization_result_dataclass():
    """测试 OptimizationResult 数据类"""
    from webapp.utils.primary_model.base import OptimizationResult

    result = OptimizationResult(
        best_params={'short_window': 5, 'long_window': 20},
        best_score=0.75,
        all_results=pd.DataFrame({'fold': [0, 1]}),
        cv_results={'n_folds': 2}
    )

    assert result.best_params['short_window'] == 5
    assert result.best_score == 0.75
    assert len(result.all_results) == 2


def test_signal_result_dataclass():
    """测试 SignalResult 数据类"""
    from webapp.utils.primary_model.base import SignalResult

    signals = pd.Series([1, 0, -1, 0, 1])
    positions = pd.Series([1, 1, -1, -1, 1])
    events = pd.DataFrame({'label': [1, -1, 0, 1, -1]})

    result = SignalResult(
        signals=signals,
        positions=positions,
        events_with_labels=events
    )

    assert len(result.signals) == 5
    assert result.positions.iloc[0] == 1


@pytest.fixture
def test_strategy():
    """创建测试用 TestStrategy 类的 fixture"""
    from webapp.utils.primary_model.base import PrimaryModelBase

    class TestStrategy(PrimaryModelBase):
        @property
        def name(self):
            return "Test"

        @property
        def param_grid(self):
            return {}

        def generate_signals(self, data, **params):
            ...

    return TestStrategy


def test_evaluate_recall(test_strategy):
    """测试 Recall 计算"""
    # 创建测试数据
    signals = pd.Series([1, 0, 1, 0, 1], index=range(5))
    labels = pd.Series([1, 1, -1, 1, 0], index=range(5))

    # 正标签: indices 0, 1, 3
    # 信号在正标签处: index 0 有信号 (1), index 1 无信号 (0), index 3 无信号 (0)
    # TP = 1 (index 0), FN = 2 (indices 1, 3)
    # Recall = 1 / (1 + 2) = 0.333...

    strategy = test_strategy()
    recall = strategy.evaluate(signals, labels)

    assert abs(recall - 1/3) < 0.001


def test_evaluate_recall_empty_positive_labels(test_strategy):
    """测试无正标签时的 Recall"""
    signals = pd.Series([1, 0, -1], index=range(3))
    labels = pd.Series([0, 0, -1], index=range(3))  # 无正标签

    strategy = test_strategy()
    recall = strategy.evaluate(signals, labels)

    assert recall == 0.0


def test_evaluate_recall_different_indices(test_strategy):
    """测试 signals 和 labels 有完全不同的索引"""
    signals = pd.Series([1, 0, 1], index=[0, 1, 2])
    labels = pd.Series([1, 1, 1], index=[10, 11, 12])  # 完全不同的索引

    strategy = test_strategy()
    recall = strategy.evaluate(signals, labels)

    assert recall == 0.0


def test_evaluate_recall_partial_overlap(test_strategy):
    """测试 signals 和 labels 有部分索引重叠"""
    # signals: indices 0,1,2,3,4 - 有正信号在 0,2,4
    # labels: indices 2,3,4,5,6 - 正标签在 2,3,4
    signals = pd.Series([1, 0, 1, 0, 1, 0], index=range(6))
    labels = pd.Series([0, 0, 1, 1, 1, 0], index=range(6))

    # 正标签 indices: 2, 3, 4
    # 有信号的正标签: indices 2, 4
    # TP = 2, FN = 1 (index 3)
    # Recall = 2 / 3 = 0.666...

    strategy = test_strategy()
    recall = strategy.evaluate(signals, labels)

    assert abs(recall - 2/3) < 0.001


def test_evaluate_recall_all_zeros(test_strategy):
    """测试 signals 全为 0 时的 Recall"""
    signals = pd.Series([0, 0, 0, 0, 0], index=range(5))
    labels = pd.Series([1, 1, -1, 1, 0], index=range(5))

    # 正标签: indices 0, 1, 3
    # 所有正标签处都没有信号
    # TP = 0, FN = 3
    # Recall = 0 / 3 = 0

    strategy = test_strategy()
    recall = strategy.evaluate(signals, labels)

    assert recall == 0.0


def test_evaluate_recall_no_common_indices(test_strategy):
    """测试无共同索引时的 Recall（边界情况）"""
    signals = pd.Series([1, 1], index=[0, 1])
    labels = pd.Series([1, 1], index=[5, 6])  # 无共同索引

    strategy = test_strategy()
    recall = strategy.evaluate(signals, labels)

    assert recall == 0.0


def test_evaluate_recall_empty_signals(test_strategy):
    """测试空 signals 时的 Recall"""
    signals = pd.Series([], dtype=int)
    labels = pd.Series([1, 1, 1], index=range(3))

    strategy = test_strategy()
    recall = strategy.evaluate(signals, labels)

    assert recall == 0.0