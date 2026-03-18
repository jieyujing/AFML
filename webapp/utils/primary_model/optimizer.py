# webapp/utils/primary_model/optimizer.py

import pandas as pd
import numpy as np
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Any

from .base import OptimizationResult


@dataclass
class FoldResult:
    """单折验证结果"""
    fold_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    best_params: Dict[str, Any]
    train_score: float
    test_score: float


class WalkForwardOptimizer:
    """Walk-Forward 优化器"""

    def __init__(
        self,
        train_size: int = 100,
        test_size: int = 30,
        embargo: int = 5
    ):
        """
        :param train_size: 训练窗口大小（事件数）
        :param test_size: 测试窗口大小（事件数）
        :param embargo: 训练和测试之间的隔离期
        """
        self.train_size = train_size
        self.test_size = test_size
        self.embargo = embargo

    def get_splits(self, n_samples: int) -> List[tuple]:
        """
        生成 Walk-Forward 分割索引

        :param n_samples: 总样本数
        :returns: [(train_idx, test_idx), ...]
        """
        splits = []
        start = 0

        while start + self.train_size + self.embargo + self.test_size <= n_samples:
            train_end = start + self.train_size
            test_start = train_end + self.embargo
            test_end = test_start + self.test_size

            train_idx = np.arange(start, train_end)
            test_idx = np.arange(test_start, test_end)

            splits.append((train_idx, test_idx))
            start = test_end  # 滚动窗口

        return splits

    def optimize(
        self,
        data: pd.DataFrame,
        strategy,
        metric: str = 'recall'  # noqa: ARG002 - 预留扩展接口
    ) -> OptimizationResult:
        """
        执行 Walk-Forward 优化

        :param data: CUSUM 采样数据
        :param strategy: 策略实例 (PrimaryModelBase)
        :param metric: 优化目标（目前仅支持 recall）
        :returns: OptimizationResult
        """
        # 获取参数网格并生成通用参数组合
        param_grid = strategy.param_grid
        if not param_grid:
            raise ValueError("参数网格为空，无法执行优化")

        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        param_combinations = [dict(zip(keys, combo)) for combo in product(*values)]

        if not param_combinations:
            raise ValueError("无有效参数组合，无法执行优化")

        splits = self.get_splits(len(data))

        if not splits:
            raise ValueError(
                f"数据量不足：需要至少 "
                f"{self.train_size + self.embargo + self.test_size} 个事件，"
                f"当前仅 {len(data)} 个"
            )

        fold_results = []
        all_test_scores = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]

            # 在训练集上网格搜索
            best_train_score = -np.inf
            best_params = None

            for params in param_combinations:
                result = strategy.generate_signals(train_data, **params)

                # 信号与标签对齐
                valid_idx = result.events_with_labels.index
                aligned_signals = result.signals.loc[valid_idx]
                labels = result.events_with_labels['label']

                score = strategy.evaluate(aligned_signals, labels)

                if score > best_train_score:
                    best_train_score = score
                    best_params = params.copy()

            # 在测试集上验证
            test_result = strategy.generate_signals(
                test_data,
                **best_params
            )

            valid_idx = test_result.events_with_labels.index
            aligned_signals = test_result.signals.loc[valid_idx]
            test_labels = test_result.events_with_labels['label']
            test_score = strategy.evaluate(aligned_signals, test_labels)

            fold_results.append(FoldResult(
                fold_idx=fold_idx,
                train_start=int(train_idx[0]),
                train_end=int(train_idx[-1]),
                test_start=int(test_idx[0]),
                test_end=int(test_idx[-1]),
                best_params=best_params,
                train_score=best_train_score,
                test_score=test_score
            ))

            all_test_scores.append(test_score)

        # 汇总结果
        avg_score = np.mean(all_test_scores)
        std_score = np.std(all_test_scores)

        # 选择最常出现的最优参数
        param_counts = {}
        for fr in fold_results:
            # 使用 tuple 作为可哈希的 key
            key = tuple(sorted(fr.best_params.items()))
            param_counts[key] = param_counts.get(key, 0) + 1

        most_common = max(param_counts.items(), key=lambda x: x[1])
        final_best_params = dict(most_common[0])

        # 构建结果 DataFrame
        results_df = pd.DataFrame([
            {
                'fold': fr.fold_idx,
                'train_range': f"{fr.train_start}-{fr.train_end}",
                'test_range': f"{fr.test_start}-{fr.test_end}",
                **fr.best_params,  # 动态包含所有参数列
                'train_recall': fr.train_score,
                'test_recall': fr.test_score
            }
            for fr in fold_results
        ])

        return OptimizationResult(
            best_params=final_best_params,
            best_score=avg_score,
            all_results=results_df,
            cv_results={
                'fold_results': fold_results,
                'n_folds': len(fold_results),
                'avg_test_score': avg_score,
                'std_test_score': std_score
            }
        )