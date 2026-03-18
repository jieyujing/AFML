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


def test_evaluate_recall():
    """测试 Recall 计算"""
    from webapp.utils.primary_model.base import PrimaryModelBase

    # 创建测试数据
    signals = pd.Series([1, 0, 1, 0, 1], index=range(5))
    labels = pd.Series([1, 1, -1, 1, 0], index=range(5))

    # 正标签: indices 0, 1, 3
    # 信号在正标签处: index 0 有信号 (1), index 1 无信号 (0), index 3 无信号 (0)
    # TP = 1 (index 0), FN = 2 (indices 1, 3)
    # Recall = 1 / (1 + 2) = 0.333...

    # 创建一个匿名类来测试 evaluate 方法
    class TestStrategy(PrimaryModelBase):
        @property
        def name(self):
            return "Test"

        @property
        def param_grid(self):
            return {}

        def generate_signals(self, data, **params):
            pass

    strategy = TestStrategy()
    recall = strategy.evaluate(signals, labels)

    assert abs(recall - 1/3) < 0.001


def test_evaluate_recall_empty_positive_labels():
    """测试无正标签时的 Recall"""
    from webapp.utils.primary_model.base import PrimaryModelBase

    signals = pd.Series([1, 0, -1], index=range(3))
    labels = pd.Series([0, 0, -1], index=range(3))  # 无正标签

    class TestStrategy(PrimaryModelBase):
        @property
        def name(self):
            return "Test"

        @property
        def param_grid(self):
            return {}

        def generate_signals(self, data, **params):
            pass

    strategy = TestStrategy()
    recall = strategy.evaluate(signals, labels)

    assert recall == 0.0