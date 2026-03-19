# Primary Model (双均线策略) 实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 AFMLKit Webapp 中实现 Primary Model 页面，提供双均线策略的信号生成和 Walk-Forward 优化功能。

**Architecture:** 创建 `webapp/utils/primary_model/` 模块，包含抽象基类、双均线策略实现和优化器。更新 session.py 和 sidebar.py 以支持新的状态键和导航。新增可视化组件和主页面。

**Tech Stack:** Python, Streamlit, Pandas, NumPy, Plotly

---

## File Structure

```
webapp/
├── utils/primary_model/
│   ├── __init__.py          # 模块入口，导出主要类
│   ├── base.py              # 数据类和抽象基类
│   ├── dual_ma.py           # 双均线策略实现
│   └── optimizer.py         # Walk-Forward 优化器
├── components/
│   └── primary_model_viz.py # 可视化组件
├── pages/
│   └── 04_primary_model.py  # Primary Model 页面
├── session.py               # 修改：添加状态键
└── components/sidebar.py    # 修改：更新导航
```

---

### Task 1: 创建基础数据类和抽象基类

**Files:**
- Create: `webapp/utils/primary_model/__init__.py`
- Create: `webapp/utils/primary_model/base.py`
- Create: `tests/webapp/test_primary_model_base.py`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p webapp/utils/primary_model
mkdir -p tests/webapp
```

- [ ] **Step 2: 编写 base.py 测试**

```python
# tests/webapp/test_primary_model_base.py

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
```

- [ ] **Step 3: 运行测试确认失败**

```bash
cd /Users/link/Documents/AFMLKIT && NUMBA_DISABLE_JIT=1 uv run pytest tests/webapp/test_primary_model_base.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 4: 实现 base.py**

```python
# webapp/utils/primary_model/base.py

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
        pass

    @property
    @abstractmethod
    def param_grid(self) -> Dict[str, List]:
        """参数搜索空间"""
        pass

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
        pass

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
```

- [ ] **Step 5: 创建 __init__.py**

```python
# webapp/utils/primary_model/__init__.py

from .base import OptimizationResult, SignalResult, PrimaryModelBase
from .dual_ma import DualMAStrategy
from .optimizer import WalkForwardOptimizer

__all__ = [
    'OptimizationResult',
    'SignalResult',
    'PrimaryModelBase',
    'DualMAStrategy',
    'WalkForwardOptimizer'
]
```

- [ ] **Step 6: 运行测试确认通过**

```bash
cd /Users/link/Documents/AFMLKIT && NUMBA_DISABLE_JIT=1 uv run pytest tests/webapp/test_primary_model_base.py -v
```

Expected: PASS

- [ ] **Step 7: 提交**

```bash
git add webapp/utils/primary_model/ tests/webapp/test_primary_model_base.py
git commit -m "feat(webapp): add primary model base classes and data models

- Add OptimizationResult and SignalResult dataclasses
- Add PrimaryModelBase abstract class with evaluate() method
- Add unit tests for base classes"
```

---

### Task 2: 实现 DualMAStrategy 双均线策略

**Files:**
- Create: `webapp/utils/primary_model/dual_ma.py`
- Create: `tests/webapp/test_dual_ma_strategy.py`

- [ ] **Step 1: 编写测试**

```python
# tests/webapp/test_dual_ma_strategy.py

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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /Users/link/Documents/AFMLKIT && NUMBA_DISABLE_JIT=1 uv run pytest tests/webapp/test_dual_ma_strategy.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 3: 实现 dual_ma.py**

```python
# webapp/utils/primary_model/dual_ma.py

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List
from itertools import product

from .base import PrimaryModelBase, SignalResult


class DualMAStrategy(PrimaryModelBase):
    """双均线策略"""

    def __init__(
        self,
        short_range: Tuple[int, int] = (3, 10),
        long_range: Tuple[int, int] = (15, 50),
        step: int = 2,
        tp_ratio: float = 2.0,
        sl_ratio: float = 1.0,
        time_barrier: Optional[int] = None,
        vol_window: int = 20
    ):
        """
        :param short_range: 短期均线周期范围 (min, max)
        :param long_range: 长期均线周期范围 (min, max)
        :param step: 参数搜索步长
        :param tp_ratio: 止盈倍数（相对波动率）
        :param sl_ratio: 止损倍数（相对波动率）
        :param time_barrier: 时间屏障（事件数），None 表示不使用
        :param vol_window: 波动率计算窗口
        """
        self.short_range = short_range
        self.long_range = long_range
        self.step = step
        self.tp_ratio = tp_ratio
        self.sl_ratio = sl_ratio
        self.time_barrier = time_barrier
        self.vol_window = vol_window

    @property
    def name(self) -> str:
        return "双均线策略"

    @property
    def param_grid(self) -> Dict[str, List[int]]:
        """生成参数网格，确保 long > short"""
        short_vals = list(range(
            self.short_range[0],
            self.short_range[1] + 1,
            self.step
        ))
        long_vals = list(range(
            self.long_range[0],
            self.long_range[1] + 1,
            self.step
        ))

        # 过滤无效组合
        valid_combos = [
            (s, l) for s, l in product(short_vals, long_vals)
            if l > s
        ]

        return {
            'short_window': [c[0] for c in valid_combos],
            'long_window': [c[1] for c in valid_combos]
        }

    def generate_signals(
        self,
        data: pd.DataFrame,
        short_window: int = 5,
        long_window: int = 20,
        **kwargs
    ) -> SignalResult:
        """
        生成双均线交叉信号

        :param data: CUSUM采样数据，必须包含 'price' 列
        :param short_window: 短期均线周期
        :param long_window: 长期均线周期
        """
        prices = data['price']

        # 计算均线
        ma_short = prices.rolling(window=short_window, min_periods=1).mean()
        ma_long = prices.rolling(window=long_window, min_periods=1).mean()

        # 持仓状态：短期均线在上则为多头
        position = pd.Series(
            np.where(ma_short > ma_long, 1, -1),
            index=prices.index
        )

        # 信号：持仓变化点
        signal_change = position.diff()
        signals = pd.Series(0, index=prices.index, dtype=int)
        signals[signal_change > 0] = 1   # 金叉做多
        signals[signal_change < 0] = -1  # 死叉做空

        # 计算波动率（用于 TBM 动态止盈止损）
        returns = np.log(prices / prices.shift(1))
        volatility = returns.rolling(
            window=self.vol_window,
            min_periods=1
        ).std()

        # 构建 TBM 输入数据
        events_df = data.copy()
        events_df['volatility'] = volatility

        # 计算 TBM 标签
        events_with_labels = self._compute_tbm_labels(
            events_df,
            tp_ratio=self.tp_ratio,
            sl_ratio=self.sl_ratio,
            time_barrier=self.time_barrier
        )

        return SignalResult(
            signals=signals,
            positions=position,
            events_with_labels=events_with_labels
        )

    def _compute_tbm_labels(
        self,
        data: pd.DataFrame,
        tp_ratio: float,
        sl_ratio: float,
        time_barrier: Optional[int]
    ) -> pd.DataFrame:
        """计算三重屏障法标签"""
        prices = data['price'].values
        volatility = data['volatility'].values

        results = []
        n = len(prices)

        for i in range(n - 1):
            entry_price = prices[i]
            entry_vol = volatility[i] if not np.isnan(volatility[i]) else 0.02

            # 动态止盈止损
            tp = entry_price * (1 + tp_ratio * entry_vol)
            sl = entry_price * (1 - sl_ratio * entry_vol)

            label = 0
            exit_idx = n - 1

            # 搜索退出点
            max_j = min(i + time_barrier, n) if time_barrier else n
            for j in range(i + 1, max_j):
                if prices[j] >= tp:
                    label = 1   # 止盈
                    exit_idx = j
                    break
                elif prices[j] <= sl:
                    label = -1  # 止损
                    exit_idx = j
                    break
            else:
                # 时间屏障或数据末尾
                if time_barrier and i + time_barrier < n:
                    exit_idx = i + time_barrier
                    label = 1 if prices[exit_idx] > entry_price else -1
                else:
                    exit_idx = n - 1
                    label = 1 if prices[exit_idx] > entry_price else -1

            results.append({
                'entry_idx': i,
                'exit_idx': exit_idx,
                'entry_price': entry_price,
                'exit_price': prices[exit_idx],
                'volatility': entry_vol,
                'label': label,
                'returns': np.log(prices[exit_idx] / entry_price)
            })

        df = pd.DataFrame(results)
        df.index = data.index[:len(results)]

        return df
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /Users/link/Documents/AFMLKIT && NUMBA_DISABLE_JIT=1 uv run pytest tests/webapp/test_dual_ma_strategy.py -v
```

Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add webapp/utils/primary_model/dual_ma.py tests/webapp/test_dual_ma_strategy.py
git commit -m "feat(webapp): implement DualMAStrategy for primary model

- Add dual moving average signal generation
- Add TBM label computation with dynamic tp/sl
- Add comprehensive unit tests"
```

---

### Task 3: 实现 WalkForwardOptimizer

**Files:**
- Create: `webapp/utils/primary_model/optimizer.py`
- Create: `tests/webapp/test_optimizer.py`

- [ ] **Step 1: 编写测试**

```python
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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /Users/link/Documents/AFMLKIT && NUMBA_DISABLE_JIT=1 uv run pytest tests/webapp/test_optimizer.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 3: 实现 optimizer.py**

```python
# webapp/utils/primary_model/optimizer.py

import pandas as pd
import numpy as np
from dataclasses import dataclass
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
        embargo: int = 5,
        min_train_size: int = 50
    ):
        """
        :param train_size: 训练窗口大小（事件数）
        :param test_size: 测试窗口大小（事件数）
        :param embargo: 训练和测试之间的隔离期
        :param min_train_size: 最小训练样本数
        """
        self.train_size = train_size
        self.test_size = test_size
        self.embargo = embargo
        self.min_train_size = min_train_size

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
        metric: str = 'recall'
    ) -> OptimizationResult:
        """
        执行 Walk-Forward 优化

        :param data: CUSUM 采样数据
        :param strategy: 策略实例 (PrimaryModelBase)
        :param metric: 优化目标（目前仅支持 recall）
        :returns: OptimizationResult
        """
        # 获取参数网格
        param_grid = strategy.param_grid
        param_combinations = list(zip(
            param_grid['short_window'],
            param_grid['long_window']
        ))

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

            for short_w, long_w in param_combinations:
                result = strategy.generate_signals(
                    train_data,
                    short_window=short_w,
                    long_window=long_w
                )

                # 信号与标签对齐
                valid_idx = result.events_with_labels.index
                aligned_signals = result.signals.loc[valid_idx]
                labels = result.events_with_labels['label']

                score = strategy.evaluate(aligned_signals, labels)

                if score > best_train_score:
                    best_train_score = score
                    best_params = {
                        'short_window': short_w,
                        'long_window': long_w
                    }

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
            key = (fr.best_params['short_window'], fr.best_params['long_window'])
            param_counts[key] = param_counts.get(key, 0) + 1

        most_common = max(param_counts.items(), key=lambda x: x[1])
        final_best_params = {
            'short_window': most_common[0][0],
            'long_window': most_common[0][1]
        }

        # 构建结果 DataFrame
        results_df = pd.DataFrame([
            {
                'fold': fr.fold_idx,
                'train_range': f"{fr.train_start}-{fr.train_end}",
                'test_range': f"{fr.test_start}-{fr.test_end}",
                'short_window': fr.best_params['short_window'],
                'long_window': fr.best_params['long_window'],
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
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /Users/link/Documents/AFMLKIT && NUMBA_DISABLE_JIT=1 uv run pytest tests/webapp/test_optimizer.py -v
```

Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add webapp/utils/primary_model/optimizer.py tests/webapp/test_optimizer.py
git commit -m "feat(webapp): implement WalkForwardOptimizer

- Add walk-forward cross-validation splits
- Add grid search optimization with recall metric
- Add comprehensive unit tests"
```

---

### Task 4: 更新 Session 状态

**Files:**
- Modify: `webapp/session.py`

- [ ] **Step 1: 在 KEYS 列表中添加新键**

在 `webapp/session.py` 中找到 KEYS 列表（第 50-59 行），在 `'cusum_state'` 后添加新键：

```python
KEYS = [
    'raw_data', 'bar_data', 'dollar_bars', 'features', 'labels', 'sample_weights',
    'bar_config', 'feature_config', 'label_config', 'model_config', 'backtest_config',
    'model', 'model_results', 'feature_importance', 'feature_metadata',
    'backtest_results', 'plots',
    'current_step', 'is_processing', 'last_updated',
    'experiment_name', 'experiment_notes',
    'iid_results', 'iid_score_df', 'best_freq', 'generation_time',
    'cusum_config', 'cusum_sampled_data', 'cusum_events', 'cusum_state',
    # Primary Model 相关（新增）
    'primary_model_config',
    'primary_model_result',
    'primary_model_signals',
    'primary_model_labels'
]
```

- [ ] **Step 2: 在 init_session 中添加初始化**

在 `webapp/session.py` 的 `init_session` 方法中（约第 68-71 行），将 `primary_model_config` 添加到配置键列表：

```python
            elif key in ['bar_config', 'feature_config', 'label_config',
                        'model_config', 'backtest_config', 'model_results',
                        'backtest_results', 'feature_metadata', 'cusum_config',
                        'primary_model_config']:
```

- [ ] **Step 3: 在 reset_all 中添加重置逻辑**

在 `webapp/session.py` 的 `reset_all` 方法中（约第 128-131 行），将 `primary_model_config` 添加到配置键列表：

```python
            elif key in ['bar_config', 'feature_config', 'label_config',
                        'model_config', 'backtest_config', 'model_results',
                        'backtest_results', 'feature_metadata', 'cusum_config',
                        'primary_model_config']:
```

- [ ] **Step 4: 运行现有测试确认无回归**

```bash
cd /Users/link/Documents/AFMLKIT && NUMBA_DISABLE_JIT=1 uv run pytest tests/ -q
```

Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add webapp/session.py
git commit -m "feat(webapp): add primary model state keys to SessionManager"
```

---

### Task 5: 重命名现有页面文件

**Files:**
- Rename: `webapp/pages/04_feature_engineering.py` → `webapp/pages/05_feature_engineering.py`
- Rename: `webapp/pages/05_labeling.py` → `webapp/pages/06_labeling.py`
- Rename: `webapp/pages/06_feature_analysis.py` → `webapp/pages/07_feature_analysis.py`
- Rename: `webapp/pages/07_model_training.py` → `webapp/pages/08_model_training.py`
- Rename: `webapp/pages/08_backtest.py` → `webapp/pages/09_backtest.py`
- Rename: `webapp/pages/09_visualization.py` → `webapp/pages/10_visualization.py`
- Rename: `webapp/pages/10_experiment.py` → `webapp/pages/11_experiment.py`

- [ ] **Step 1: 重命名页面文件**

```bash
cd /Users/link/Documents/AFMLKIT/webapp/pages
mv 04_feature_engineering.py 05_feature_engineering.py
mv 05_labeling.py 06_labeling.py
mv 06_feature_analysis.py 07_feature_analysis.py
mv 07_model_training.py 08_model_training.py
mv 08_backtest.py 09_backtest.py
mv 09_visualization.py 10_visualization.py
mv 10_experiment.py 11_experiment.py
```

- [ ] **Step 2: 提交**

```bash
git add webapp/pages/
git commit -m "refactor(webapp): renumber pages for Primary Model insertion"
```

---

### Task 6: 创建可视化组件

**Files:**
- Create: `webapp/components/primary_model_viz.py`

- [ ] **Step 1: 创建可视化组件**

```python
# webapp/components/primary_model_viz.py

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional


def plot_optimization_results(result) -> go.Figure:
    """参数分布热力图

    :param result: OptimizationResult 实例
    :returns: Plotly Figure
    """
    df = result.all_results

    # 统计每个参数组合的出现频率
    param_counts = df.groupby(['short_window', 'long_window']).size().reset_index(name='count')

    fig = px.density_heatmap(
        param_counts,
        x='short_window',
        y='long_window',
        z='count',
        title='最优参数分布',
        labels={
            'short_window': '短期均线',
            'long_window': '长期均线',
            'count': '频次'
        }
    )

    return fig


def plot_fold_performance(result) -> go.Figure:
    """各 Fold Recall 趋势

    :param result: OptimizationResult 实例
    :returns: Plotly Figure
    """
    df = result.all_results

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['fold'],
        y=df['train_recall'],
        mode='lines+markers',
        name='训练集 Recall',
        line=dict(color='#1f77b4')
    ))

    fig.add_trace(go.Scatter(
        x=df['fold'],
        y=df['test_recall'],
        mode='lines+markers',
        name='测试集 Recall',
        line=dict(color='#2ca02c')
    ))

    fig.add_hline(
        y=result.best_score,
        line_dash='dash',
        line_color='red',
        annotation_text=f'平均: {result.best_score:.2%}'
    )

    fig.update_layout(
        title='Walk-Forward Recall 趋势',
        xaxis_title='Fold',
        yaxis_title='Recall',
        hovermode='x unified'
    )

    return fig


def plot_signals_overview(
    data: pd.DataFrame,
    signals: pd.Series,
    labels: Optional[pd.DataFrame] = None
) -> go.Figure:
    """信号概览图

    :param data: 价格数据，必须包含 'price' 列
    :param signals: 信号序列 (±1/0)
    :param labels: 可选的 TBM 标签 DataFrame
    :returns: Plotly Figure
    """
    fig = go.Figure()

    # 价格线
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['price'],
        mode='lines',
        name='价格',
        line=dict(color='gray', width=1)
    ))

    # 做多信号
    long_idx = signals[signals == 1].index
    if len(long_idx) > 0:
        fig.add_trace(go.Scatter(
            x=long_idx,
            y=data.loc[long_idx, 'price'],
            mode='markers',
            name='做多信号',
            marker=dict(symbol='triangle-up', size=10, color='green')
        ))

    # 做空信号
    short_idx = signals[signals == -1].index
    if len(short_idx) > 0:
        fig.add_trace(go.Scatter(
            x=short_idx,
            y=data.loc[short_idx, 'price'],
            mode='markers',
            name='做空信号',
            marker=dict(symbol='triangle-down', size=10, color='red')
        ))

    fig.update_layout(
        title='交易信号概览',
        xaxis_title='时间',
        yaxis_title='价格',
        hovermode='x unified'
    )

    return fig
```

- [ ] **Step 2: 提交**

```bash
git add webapp/components/primary_model_viz.py
git commit -m "feat(webapp): add primary model visualization components

- Add parameter distribution heatmap
- Add fold recall trend chart
- Add signals overview chart"
```

---

### Task 7: 创建 Primary Model 页面

**Files:**
- Create: `webapp/pages/04_primary_model.py`

- [ ] **Step 1: 创建页面文件**

```python
# webapp/pages/04_primary_model.py

"""Primary Model 页面 - 双均线策略信号生成与优化"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from webapp.session import SessionManager
from webapp.utils.primary_model import DualMAStrategy, WalkForwardOptimizer
from webapp.components.primary_model_viz import (
    plot_optimization_results,
    plot_fold_performance,
    plot_signals_overview
)


def main():
    st.title("📈 Primary Model - 双均线策略")
    st.markdown("""
    基于 **Meta-Labeling** 框架的 Primary Model。
    目标：**最大化 Recall**，捕获尽可能多的盈利机会。
    """)

    sm = SessionManager()

    # ========== Step 1: 数据源确认 ==========
    st.header("Step 1: 数据源确认")

    cusum_data = sm.get('cusum_sampled_data')
    if cusum_data is None:
        st.warning("⚠️ 请先在「CUSUM 采样」页面生成采样数据")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("事件数", len(cusum_data))
    with col2:
        if isinstance(cusum_data.index, pd.DatetimeIndex):
            st.metric("时间范围",
                      f"{cusum_data.index[0].date()} ~ {cusum_data.index[-1].date()}")
        else:
            st.metric("数据点数", len(cusum_data))
    with col3:
        cusum_config = sm.get('cusum_config', {})
        sample_rate = cusum_config.get('sample_rate', 'N/A')
        st.metric("采样率", f"{sample_rate}" if sample_rate else "N/A")

    with st.expander("📋 数据预览"):
        st.dataframe(cusum_data.head(10))

    # ========== Step 2: TBM 参数配置 ==========
    st.header("Step 2: 三重屏障参数")

    col1, col2, col3 = st.columns(3)
    with col1:
        tp_ratio = st.number_input(
            "止盈倍数",
            min_value=0.5, max_value=5.0,
            value=2.0, step=0.1,
            help="止盈阈值 = 波动率 × 此倍数"
        )
    with col2:
        sl_ratio = st.number_input(
            "止损倍数",
            min_value=0.5, max_value=5.0,
            value=1.0, step=0.1,
            help="止损阈值 = 波动率 × 此倍数"
        )
    with col3:
        time_barrier = st.number_input(
            "时间屏障（事件数）",
            min_value=0, max_value=100,
            value=0, step=5,
            help="0 表示不使用时间屏障"
        )

    st.info(f"📌 止盈/止损比 = {tp_ratio}:{sl_ratio}")

    # ========== Step 3: 策略参数范围 ==========
    st.header("Step 3: 双均线参数范围")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("短期均线")
        short_min = st.number_input("最小周期", 2, 20, 3, key="short_min")
        short_max = st.number_input("最大周期", 5, 50, 10, key="short_max")
        short_step = st.number_input("步长", 1, 10, 2, key="short_step")

    with col2:
        st.subheader("长期均线")
        long_min = st.number_input("最小周期", 10, 100, 15, key="long_min")
        long_max = st.number_input("最大周期", 20, 200, 50, key="long_max")
        long_step = st.number_input("步长", 1, 20, 5, key="long_step")

    # 参数组合预览
    short_vals = list(range(short_min, short_max + 1, short_step))
    long_vals = list(range(long_min, long_max + 1, long_step))
    n_combinations = len([1 for s in short_vals for l in long_vals if l > s])

    st.caption(f"参数组合数: {n_combinations}")

    # ========== Step 4: Walk-Forward 配置 ==========
    st.header("Step 4: Walk-Forward 配置")

    col1, col2, col3 = st.columns(3)
    with col1:
        train_size = st.number_input(
            "训练窗口（事件数）",
            min_value=20, max_value=500, value=100
        )
    with col2:
        test_size = st.number_input(
            "测试窗口（事件数）",
            min_value=10, max_value=100, value=30
        )
    with col3:
        embargo = st.number_input(
            "隔离期（事件数）",
            min_value=0, max_value=20, value=5
        )

    # 预估 Fold 数
    required = train_size + embargo + test_size
    n_folds = max(0, (len(cusum_data) - required) // test_size) if test_size > 0 else 0
    st.caption(f"需要至少 {required} 个事件，当前 {len(cusum_data)} 个，预计 Fold 数: ~{n_folds}")

    # ========== 执行优化 ==========
    st.header("Step 5: 执行优化")

    if st.button("🚀 开始 Walk-Forward 优化", type="primary"):
        if n_folds <= 0:
            st.error(f"❌ 数据量不足！需要至少 {required} 个事件，当前仅 {len(cusum_data)} 个")
        else:
            with st.spinner("优化中..."):
                # 创建策略实例
                strategy = DualMAStrategy(
                    short_range=(short_min, short_max),
                    long_range=(long_min, long_max),
                    step=max(short_step, long_step),
                    tp_ratio=tp_ratio,
                    sl_ratio=sl_ratio,
                    time_barrier=time_barrier if time_barrier > 0 else None
                )

                # 创建优化器
                optimizer = WalkForwardOptimizer(
                    train_size=train_size,
                    test_size=test_size,
                    embargo=embargo
                )

                # 执行优化
                result = optimizer.optimize(cusum_data, strategy, metric='recall')

                # 保存到 session
                sm.update('primary_model_result', result)
                sm.update('primary_model_config', {
                    'tp_ratio': tp_ratio,
                    'sl_ratio': sl_ratio,
                    'time_barrier': time_barrier,
                    'short_range': (short_min, short_max),
                    'long_range': (long_min, long_max),
                    'train_size': train_size,
                    'test_size': test_size,
                    'embargo': embargo
                })

                st.success("✅ 优化完成!")
                st.rerun()

    # ========== 结果展示 ==========
    result = sm.get('primary_model_result')
    if result:
        st.header("Step 6: 结果展示")

        # 汇总指标
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "最优短期均线",
                result.best_params['short_window']
            )
        with col2:
            st.metric(
                "最优长期均线",
                result.best_params['long_window']
            )
        with col3:
            st.metric(
                "平均测试 Recall",
                f"{result.best_score:.2%}"
            )

        # Fold 结果表
        st.subheader("📋 各 Fold 详情")
        st.dataframe(
            result.all_results.style.format({
                'train_recall': '{:.2%}',
                'test_recall': '{:.2%}'
            }),
            use_container_width=True
        )

        # 可视化
        st.subheader("📊 可视化")
        tab1, tab2 = st.tabs(["参数分布", "Recall 趋势"])

        with tab1:
            fig = plot_optimization_results(result)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = plot_fold_performance(result)
            st.plotly_chart(fig, use_container_width=True)

        # 导出
        st.subheader("💾 导出信号")
        if st.button("生成最终信号并导出"):
            # 用最优参数生成全量信号
            config = sm.get('primary_model_config', {})
            strategy = DualMAStrategy(
                tp_ratio=config.get('tp_ratio', 2.0),
                sl_ratio=config.get('sl_ratio', 1.0),
                time_barrier=config.get('time_barrier') if config.get('time_barrier', 0) > 0 else None
            )
            final_result = strategy.generate_signals(
                cusum_data,
                **result.best_params
            )

            # 保存
            sm.update('primary_model_signals', final_result.signals)
            sm.update('primary_model_labels', final_result.events_with_labels)

            st.success("✅ 信号已生成并保存到会话状态")

            # 导出 CSV
            csv = final_result.events_with_labels.to_csv()
            st.download_button(
                "📥 下载信号 CSV",
                csv,
                "primary_model_signals.csv",
                "text/csv"
            )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add webapp/pages/04_primary_model.py
git commit -m "feat(webapp): add Primary Model page

- Add 6-step workflow for dual MA strategy
- Add TBM parameter configuration
- Add walk-forward optimization UI
- Add result visualization and export"
```

---

### Task 8: 更新 Sidebar 导航

**Files:**
- Modify: `webapp/components/sidebar.py`

- [ ] **Step 1: 更新 PAGES 字典**

在 `webapp/components/sidebar.py` 中，找到 `PAGES` 字典（第 8-59 行），替换为：

```python
PAGES: Dict[str, dict] = {
    "首页": {
        "icon": "🏠",
        "file": "app.py",
        "description": "AFMLKit Web UI 首页"
    },
    "1️⃣ 数据导入": {
        "icon": "📥",
        "file": "pages/01_data_import.py",
        "description": "导入交易数据并构建 K 线"
    },
    "💵 Dollar Bar": {
        "icon": "💵",
        "file": "pages/02_dollar_bar.py",
        "description": "生成和评估 Dollar Bars"
    },
    "🔬 CUSUM 采样": {
        "icon": "🔬",
        "file": "pages/03_cusum_sampling.py",
        "description": "CUSUM 事件采样与可视化"
    },
    "📈 Primary Model": {
        "icon": "📈",
        "file": "pages/04_primary_model.py",
        "description": "双均线策略信号生成与优化"
    },
    "2️⃣ 特征工程": {
        "icon": "🔧",
        "file": "pages/05_feature_engineering.py",
        "description": "构建和变换特征"
    },
    "3️⃣ 标签生成": {
        "icon": "🏷️",
        "file": "pages/06_labeling.py",
        "description": "生成 TBM 标签和样本权重"
    },
    "4️⃣ 特征分析": {
        "icon": "📊",
        "file": "pages/07_feature_analysis.py",
        "description": "特征重要性和聚类分析"
    },
    "5️⃣ 模型训练": {
        "icon": "🤖",
        "file": "pages/08_model_training.py",
        "description": "训练和评估模型"
    },
    "6️⃣ 回测评估": {
        "icon": "📈",
        "file": "pages/09_backtest.py",
        "description": "回测策略和绩效评估"
    },
    "🎨 可视化中心": {
        "icon": "🎨",
        "file": "pages/10_visualization.py",
        "description": "查看所有可视化结果"
    }
}
```

- [ ] **Step 2: 更新 steps 列表**

在 `render_sidebar()` 函数中找到 `steps` 列表（约第 103-112 行），添加 "Primary Model"：

```python
        steps = [
            "数据导入",
            "Dollar Bar",
            "CUSUM 采样",
            "Primary Model",
            "特征工程",
            "标签生成",
            "特征分析",
            "模型训练",
            "回测评估"
        ]
```

- [ ] **Step 3: 更新 step_mapping**

在 `navigate_to()` 函数中找到 `step_mapping`（约第 181-192 行），添加 Primary Model：

```python
    step_mapping = {
        '首页': 0,
        '1️⃣ 数据导入': 0,
        '💵 Dollar Bar': 1,
        '🔬 CUSUM 采样': 2,
        '📈 Primary Model': 3,
        '2️⃣ 特征工程': 4,
        '3️⃣ 标签生成': 5,
        '4️⃣ 特征分析': 6,
        '5️⃣ 模型训练': 7,
        '6️⃣ 回测评估': 8,
        '🎨 可视化中心': 9
    }
```

- [ ] **Step 4: 添加数据状态显示**

在数据状态区域（约第 137-145 行），在 CUSUM 状态后添加 Primary Model 状态：

```python
        has_cusum = st.session_state.get('cusum_sampled_data') is not None
        status_icon = "✅" if has_cusum else "❌"
        st.markdown(f"{status_icon} CUSUM 采样")

        has_primary_signals = st.session_state.get('primary_model_signals') is not None
        status_icon = "✅" if has_primary_signals else "❌"
        st.markdown(f"{status_icon} Primary Model 信号")

        status_icon = "✅" if has_features else "❌"
        st.markdown(f"{status_icon} 特征")
```

- [ ] **Step 5: 提交**

```bash
git add webapp/components/sidebar.py
git commit -m "feat(webapp): add Primary Model to sidebar navigation"
```

---

### Task 9: 运行完整测试

- [ ] **Step 1: 运行所有测试**

```bash
cd /Users/link/Documents/AFMLKIT && NUMBA_DISABLE_JIT=1 uv run pytest tests/webapp/ -v
```

Expected: PASS

- [ ] **Step 2: 启动 Webapp 验证**

```bash
cd /Users/link/Documents/AFMLKIT && uv run streamlit run webapp/app.py
```

手动验证：
1. 导航栏显示 "📈 Primary Model"
2. 页面可正常访问
3. 数据流正常（需要先完成 CUSUM 采样）

- [ ] **Step 3: 最终提交**

```bash
git add -A
git commit -m "feat(webapp): complete Primary Model implementation

- Add DualMAStrategy with TBM label computation
- Add WalkForwardOptimizer for parameter optimization
- Add Primary Model page with 6-step workflow
- Update session state and sidebar navigation
- Add visualization components

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Summary

| Task | Description | Files Created | Files Modified |
|------|-------------|--------------|----------------|
| 1 | Base classes | 2 | 0 |
| 2 | DualMAStrategy | 1 | 0 |
| 3 | Optimizer | 1 | 0 |
| 4 | Session state | 0 | 1 |
| 5 | Rename pages | 0 | 7 |
| 6 | Visualization | 1 | 0 |
| 7 | Primary Model page | 1 | 0 |
| 8 | Sidebar | 0 | 1 |
| 9 | Tests | 3 | 0 |

**Total: 9 new files, 9 modified files**