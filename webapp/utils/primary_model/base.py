from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List
import pandas as pd


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, Any]
    best_score: float
    all_results: pd.DataFrame
    cv_results: Dict[str, Any]


@dataclass
class SignalResult:
    """信号生成结果"""
    signals: pd.Series              # 1=做多, -1=做空, 0=无信号
    positions: pd.Series            # 持仓状态 (连续)
    events_with_labels: pd.DataFrame  # 带TBM标签的事件


class PrimaryModelBase(ABC):
    """Primary Model 抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        ...

    @property
    @abstractmethod
    def param_grid(self) -> Dict[str, List]:
        """参数搜索空间"""
        ...

    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame,
        **params
    ) -> SignalResult:
        """
        生成交易信号

        :param data: CUSUM采样后的数据，必须包含 'price' 列
        :param params: 策略参数
        :returns: SignalResult
        """
        ...

    def evaluate(
        self,
        signals: pd.Series,
        labels: pd.Series
    ) -> float:
        """
        计算 Recall

        信号与标签按相同索引对齐。
        - labels > 0: 真实盈利机会
        - signals > 0: 策略发出做多信号

        Recall = TP / (TP + FN)
        - TP: labels > 0 且 signals > 0
        - FN: labels > 0 且 signals == 0

        :param signals: 策略信号 (±1/0)，索引与 labels 相同
        :param labels: TBM标签 (±1/0)，索引与 signals 相同
        :returns: Recall 分数
        """
        # 确保索引对齐
        common_idx = signals.index.intersection(labels[labels > 0].index)

        if len(common_idx) == 0:
            return 0.0

        aligned_signals = signals.loc[common_idx]
        tp = (aligned_signals > 0).sum()
        fn = (aligned_signals == 0).sum()

        if tp + fn == 0:
            return 0.0

        return float(tp / (tp + fn))