# Primary Model (双均线策略) 设计文档

**日期**: 2025-03-18
**状态**: 待审核

## 概述

在 AFMLKit Webapp 的数据管道中新增 Primary Model 页面，实现基于双均线的交易信号生成器，用于 AFML Meta-Labeling 框架的第一阶段。

## 需求总结

| 维度 | 选择 |
|------|------|
| 用途 | Meta-Labeling Primary Model |
| 计算基础 | CUSUM 采样点 |
| 参数方式 | 网格搜索优化 |
| 优化目标 | 高 Recall |
| 信号逻辑 | 经典双向交叉（多/空） |
| 页面位置 | CUSUM 采样后、特征工程前 |
| 数据划分 | Walk-Forward |
| 交易成本 | 不考虑 |
| TBM 参数 | 止盈/止损比 = 2/1（可配置） |

## 架构设计

### 目录结构

```
webapp/
├── pages/
│   ├── 04_primary_model.py         # 新增：Primary Model 页面
│   ├── 05_feature_engineering.py   # 重命名：原 04_feature_engineering.py
│   ├── 06_labeling.py              # 重命名：原 05_labeling.py
│   ├── 07_feature_analysis.py      # 重命名：原 06_feature_analysis.py
│   ├── 08_model_training.py        # 重命名：原 07_model_training.py
│   ├── 09_backtest.py              # 重命名：原 08_backtest.py
│   ├── 10_visualization.py         # 重命名：原 09_visualization.py
│   └── 11_experiment.py            # 重命名：原 10_experiment.py
├── components/
│   └── primary_model_viz.py        # 新增：可视化组件
├── utils/
│   └── primary_model/              # 新增：策略模块目录
│       ├── __init__.py
│       ├── base.py                 # PrimaryModelBase 抽象基类
│       ├── dual_ma.py              # DualMAStrategy 实现
│       └── optimizer.py            # Walk-Forward 优化器
├── session.py                      # 修改：新增状态键
└── components/sidebar.py           # 修改：更新导航
```

### 核心类关系

```
PrimaryModelBase (抽象类)
├── generate_signals()      # 生成信号
├── optimize()              # 参数优化
└── evaluate()              # 评估指标

DualMAStrategy (继承 PrimaryModelBase)
├── short_window, long_window
└── 实现 generate_signals()、evaluate()

WalkForwardOptimizer
├── train_size, test_size, embargo
└── optimize()              # 返回最优参数和验证结果
```

### 页面流程

```
Step 1: 数据源确认（来自 CUSUM 采样结果）
Step 2: TBM 参数配置（止盈/止损比、时间屏障）
Step 3: 策略参数范围（短期/长期均线周期范围）
Step 4: Walk-Forward 配置（窗口大小、隔离期）
Step 5: 执行优化（网格搜索 + Walk-Forward 验证）
Step 6: 结果展示与信号导出
```

### 数据流

**输入数据格式**：来自 CUSUM 采样的 DataFrame，必须包含：
- `price` (float): 价格序列
- `timestamp` (datetime): 时间索引
- 其他可选列：`volume`, `returns` 等

**波动率计算**：在策略内部使用 `price` 列计算对数收益率的滚动标准差。

```
CUSUM 事件 DataFrame
columns: [price, timestamp, ...]
         ↓
    TBM 计算（使用 afmlkit.label.kit.TBMLabel）
    确定每个事件点的真实交易机会
         ↓
    双均线信号生成
    金叉=做多(1), 死叉=做空(-1), 无变化=持有(0)
         ↓
    信号与标签对齐（按事件索引）
         ↓
    events_with_labels DataFrame
    columns: [timestamp, price, signal, label, returns, ...]
```

## 核心组件设计

### 1. PrimaryModelBase 抽象基类

```python
# webapp/utils/primary_model/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

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
    def param_grid(self) -> Dict[str, list]:
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
        positive_labels = labels.loc[common_idx]

        if len(positive_labels) == 0:
            return 0.0

        aligned_signals = signals.loc[common_idx]
        tp = (aligned_signals > 0).sum()
        fn = (aligned_signals == 0).sum()

        return float(tp / (tp + fn))
```

### 2. DualMAStrategy 实现

```python
# webapp/utils/primary_model/dual_ma.py

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List
from itertools import product

from afmlkit.label.kit import TBMLabel
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
        position = pd.Series(np.where(ma_short > ma_long, 1, -1), index=prices.index)

        # 信号：持仓变化点
        signal_change = position.diff()
        signals = pd.Series(0, index=prices.index, dtype=int)
        signals[signal_change > 0] = 1   # 金叉做多
        signals[signal_change < 0] = -1  # 死叉做空

        # 计算波动率（用于 TBM 动态止盈止损）
        returns = np.log(prices / prices.shift(1))
        volatility = returns.rolling(window=self.vol_window, min_periods=1).std()

        # 构建 TBM 输入数据
        events_df = data.copy()
        events_df['volatility'] = volatility

        # 使用 afmlkit 的 TBMLabel 计算标签
        # 简化实现：直接计算标签
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
                'returns': np.log(prices[exit_idx] / entry_price) if exit_idx < n else 0
            })

        df = pd.DataFrame(results)
        df.index = data.index[:len(results)]

        return df
```

### 3. Walk-Forward 优化器

```python
# webapp/utils/primary_model/optimizer.py

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from itertools import product

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
        strategy,  # PrimaryModelBase 实例
        metric: str = 'recall'
    ) -> OptimizationResult:
        """
        执行 Walk-Forward 优化

        :param data: CUSUM 采样数据
        :param strategy: 策略实例
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

                # 信号与标签对齐（只在有标签的点评估）
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

### 4. Session 状态扩展

在 `webapp/session.py` 中进行以下更新：

```python
# 1. 在 SessionManager.KEYS 列表末尾添加以下新键：
# Primary Model 相关（新增）
'primary_model_config',      # Dict: TBM + 策略配置
'primary_model_result',      # OptimizationResult
'primary_model_signals',     # pd.Series: 最终信号
'primary_model_labels',      # pd.DataFrame: TBM 标签

# 2. 注意：SessionManager 的方法都使用 @staticmethod 装饰器

# 3. 在 reset_all() 方法的配置键列表中添加（约第127行）：
elif key in ['bar_config', 'feature_config', 'label_config',
            'model_config', 'backtest_config', 'model_results',
            'backtest_results', 'feature_metadata', 'cusum_config',
            'primary_model_config']:  # 新增
    st.session_state[key] = {}

# 4. 添加独立的重置方法（可选）
@staticmethod
def reset_primary_model():
    """重置 Primary Model 相关状态"""
    SessionManager.update('primary_model_config', {})
    SessionManager.update('primary_model_result', None)
    SessionManager.update('primary_model_signals', None)
    SessionManager.update('primary_model_labels', None)
```

### 5. Sidebar 导航更新

在 `webapp/components/sidebar.py` 中更新：

```python
# 更新 PAGES 字典（注意：需要包含 icon 和 description 字段）
PAGES = {
    "首页": {"icon": "🏠", "file": "app.py", "description": "AFMLKit 金融机器学习平台"},
    "1️⃣ 数据导入": {"icon": "📥", "file": "pages/01_data_import.py", "description": "导入交易数据并构建 K 线"},
    "💵 Dollar Bar": {"icon": "💵", "file": "pages/02_dollar_bar.py", "description": "构建 Dollar Bars"},
    "🔬 CUSUM 采样": {"icon": "🔬", "file": "pages/03_cusum_sampling.py", "description": "CUSUM 事件采样"},
    "📈 Primary Model": {"icon": "📈", "file": "pages/04_primary_model.py", "description": "双均线策略信号生成与优化"},  # 新增
    "2️⃣ 特征工程": {"icon": "🔧", "file": "pages/05_feature_engineering.py", "description": "计算金融特征"},  # 编号后移
    "3️⃣ 标签生成": {"icon": "🏷️", "file": "pages/06_labeling.py", "description": "三重屏障法标签"},
    "4️⃣ 特征分析": {"icon": "📊", "file": "pages/07_feature_analysis.py", "description": "特征重要性分析"},
    "5️⃣ 模型训练": {"icon": "🤖", "file": "pages/08_model_training.py", "description": "训练 ML 模型"},
    "6️⃣ 回测评估": {"icon": "📈", "file": "pages/09_backtest.py", "description": "策略回测评估"},
    "🎨 可视化中心": {"icon": "🎨", "file": "pages/10_visualization.py", "description": "数据可视化"},
}

# 在 navigate_to() 函数内更新 step_mapping 字典
def navigate_to(page: str):
    step_mapping = {
        '首页': 0,
        '1️⃣ 数据导入': 0,
        '💵 Dollar Bar': 1,
        '🔬 CUSUM 采样': 2,
        '📈 Primary Model': 3,  # 新增
        '2️⃣ 特征工程': 4,       # 后移
        '3️⃣ 标签生成': 5,
        '4️⃣ 特征分析': 6,
        '5️⃣ 模型训练': 7,
        '6️⃣ 回测评估': 8,
        '🎨 可视化中心': 9,
    }
    # ... 其余代码不变

# 更新 steps 列表（用于进度指示器）
steps = [
    "数据导入",
    "Dollar Bar",
    "CUSUM 采样",
    "Primary Model",  # 新增
    "特征工程",
    "标签生成",
    "特征分析",
    "模型训练",
    "回测评估",
]

# 在数据状态显示区域添加 Primary Model 状态
# 插入位置：在 CUSUM 状态显示后、特征状态显示前（约第139-141行之间）
# 现有结构：
#   第 137-139 行：CUSUM 采样状态
#   [在此插入 Primary Model 状态]
#   第 141-142 行：特征状态
has_primary_signals = st.session_state.get('primary_model_signals') is not None
status_icon = "✅" if has_primary_signals else "❌"
st.markdown(f"{status_icon} Primary Model 信号")
```

### 6. 可视化组件

```python
# webapp/components/primary_model_viz.py

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot_optimization_results(result) -> go.Figure:
    """参数分布热力图"""
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
    """各 Fold Recall 趋势"""
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
    labels: pd.DataFrame
) -> go.Figure:
    """信号概览图"""
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
    fig.add_trace(go.Scatter(
        x=long_idx,
        y=data.loc[long_idx, 'price'],
        mode='markers',
        name='做多信号',
        marker=dict(symbol='triangle-up', size=10, color='green')
    ))

    # 做空信号
    short_idx = signals[signals == -1].index
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

## 文件清单

| 文件路径 | 操作 | 说明 |
|---------|------|------|
| `webapp/pages/04_primary_model.py` | 新增 | Primary Model 页面 |
| `webapp/pages/05_feature_engineering.py` | 重命名 | 原 04_feature_engineering.py |
| `webapp/pages/06_labeling.py` | 重命名 | 原 05_labeling.py |
| `webapp/pages/07_feature_analysis.py` | 重命名 | 原 06_feature_analysis.py |
| `webapp/pages/08_model_training.py` | 重命名 | 原 07_model_training.py |
| `webapp/pages/09_backtest.py` | 重命名 | 原 08_backtest.py |
| `webapp/pages/10_visualization.py` | 重命名 | 原 09_visualization.py |
| `webapp/pages/11_experiment.py` | 重命名 | 原 10_experiment.py（不在导航中） |
| `webapp/utils/primary_model/__init__.py` | 新增 | 模块入口 |
| `webapp/utils/primary_model/base.py` | 新增 | 抽象基类 |
| `webapp/utils/primary_model/dual_ma.py` | 新增 | 双均线策略 |
| `webapp/utils/primary_model/optimizer.py` | 新增 | Walk-Forward 优化器 |
| `webapp/components/primary_model_viz.py` | 新增 | 可视化组件 |
| `webapp/session.py` | 修改 | 扩展状态键、初始化方法 |
| `webapp/components/sidebar.py` | 修改 | 更新导航、数据状态显示 |
| `tests/webapp/test_primary_model.py` | 新增 | 单元测试 |

## 依赖

- 现有依赖：`pandas`, `numpy`, `plotly`, `streamlit`
- 可复用：`afmlkit.label.kit.TBMLabel`（可选，当前设计中使用简化实现）

**注意**：
- `vol_window=20` 默认值对于较短的 CUSUM 事件序列可能偏大，UI 中应允许用户调整
- 完整集成 `TBMLabel` 需要将 CUSUM DataFrame 转换为 `TradesData` 格式

## 测试计划

测试文件：`tests/webapp/test_primary_model.py`

1. **DualMAStrategy 测试**
   - 参数网格生成正确性
   - 信号生成逻辑验证
   - TBM 标签计算验证

2. **WalkForwardOptimizer 测试**
   - 分割索引生成正确性
   - 优化流程验证
   - 边界条件（数据量不足）

3. **evaluate 函数测试**
   - Recall 计算正确性
   - 索引对齐验证

## 后续扩展

1. 新增其他 Primary Model 策略（RSI、布林带等）
2. 将成熟策略迁移至 `afmlkit/strategy/` 核心库
3. 支持 Numba 加速大规模计算
4. 完整复用 `afmlkit.label.kit.TBMLabel`