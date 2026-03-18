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
│   └── 04_primary_model.py      # 新增：Primary Model 页面
├── components/
│   └── primary_model_viz.py     # 新增：可视化组件
├── utils/
│   └── primary_model/           # 新增：策略模块目录
│       ├── __init__.py
│       ├── base.py              # PrimaryModelBase 抽象基类
│       ├── dual_ma.py           # DualMAStrategy 实现
│       ├── tbm.py               # TBM 标签计算
│       └── optimizer.py         # Walk-Forward 优化器
└── session.py                   # 修改：新增状态键
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

```
CUSUM 事件 → TBM 计算（确定真实机会）→ 双均线优化（最大化 Recall）
                   ↓
            events_with_labels DataFrame
            columns: [timestamp, price, returns, label(±1/0), ...]
```

## 核心组件设计

### 1. PrimaryModelBase 抽象基类

```python
@dataclass
class OptimizationResult:
    best_params: Dict[str, Any]
    best_score: float
    all_results: pd.DataFrame
    cv_results: Dict[str, Any]

@dataclass
class SignalResult:
    signals: pd.Series           # 1=做多, -1=做空, 0=无信号
    positions: pd.Series         # 持仓状态
    events_with_labels: pd.DataFrame  # 带TBM标签的事件

class PrimaryModelBase(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def param_grid(self) -> Dict[str, list]: ...

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, **params) -> SignalResult: ...

    def evaluate(self, signals: pd.Series, labels: pd.Series, metric: str) -> float:
        # Recall 计算
        positive_labels = labels[labels > 0]
        tp = (signals[positive_labels.index] > 0).sum()
        fn = (signals[positive_labels.index] == 0).sum()
        return tp / (tp + fn) if len(positive_labels) > 0 else 0.0
```

### 2. DualMAStrategy 实现

```python
class DualMAStrategy(PrimaryModelBase):
    def __init__(
        self,
        short_range: Tuple[int, int] = (3, 10),
        long_range: Tuple[int, int] = (15, 50),
        step: int = 2,
        tbm_tp_ratio: float = 2.0,
        tbm_sl_ratio: float = 1.0,
        tbm_time_barrier: Optional[int] = None
    ): ...

    def generate_signals(self, data: pd.DataFrame, short_window: int, long_window: int) -> SignalResult:
        # 计算均线
        ma_short = prices.rolling(window=short_window).mean()
        ma_long = prices.rolling(window=long_window).mean()

        # 金叉做多、死叉做空
        signal_change = position.diff()
        signals[signal_change > 0] = 1   # 金叉
        signals[signal_change < 0] = -1  # 死叉

        # 计算 TBM 标签
        events_with_labels = compute_tbm_labels(data, ...)
        return SignalResult(signals, positions, events_with_labels)
```

### 3. TBM 标签计算

```python
def compute_tbm_labels(
    data: pd.DataFrame,
    tp_ratio: float = 2.0,
    sl_ratio: float = 1.0,
    time_barrier: Optional[int] = None,
    volatility_window: int = 20
) -> pd.DataFrame:
    # 计算动态波动率
    returns = np.log(prices[1:] / prices[:-1])
    vol = pd.Series(returns).rolling(volatility_window).std()

    for i in range(len(prices) - 1):
        # 动态止盈止损
        tp = entry_price * (1 + tp_ratio * entry_vol)
        sl = entry_price * (1 - sl_ratio * entry_vol)

        # 搜索退出点
        label = 0
        for j in range(i + 1, len(prices)):
            if prices[j] >= tp: label = 1; break
            elif prices[j] <= sl: label = -1; break
            elif time_barrier and (j - i) >= time_barrier:
                label = 1 if prices[j] > entry_price else -1
                break
```

### 4. Walk-Forward 优化器

```python
class WalkForwardOptimizer:
    def __init__(
        self,
        train_size: int = 100,
        test_size: int = 30,
        embargo: int = 5,
        min_train_size: int = 50
    ): ...

    def get_splits(self, n_samples: int) -> List[tuple]:
        # 生成滚动窗口分割
        splits = []
        start = 0
        while start + train_size + embargo + test_size <= n_samples:
            train_idx = np.arange(start, start + train_size)
            test_idx = np.arange(start + train_size + embargo,
                                 start + train_size + embargo + test_size)
            splits.append((train_idx, test_idx))
            start += train_size + embargo + test_size

    def optimize(self, data: pd.DataFrame, strategy, metric: str) -> OptimizationResult:
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            # 训练集上网格搜索最优参数
            for params in param_combinations:
                result = strategy.generate_signals(train_data, **params)
                score = strategy.evaluate(result.signals, labels, metric)

            # 测试集验证
            test_score = ...

        # 返回平均分数和最常出现的最优参数
        return OptimizationResult(best_params, avg_score, results_df, cv_results)
```

### 5. Session 状态扩展

新增状态键：
- `primary_model_config`: TBM + 策略配置
- `primary_model_result`: OptimizationResult
- `primary_model_signals`: 最终信号 Series
- `primary_model_labels`: TBM 标签 DataFrame

### 6. 可视化组件

- `plot_optimization_results()`: 参数分布热力图
- `plot_fold_performance()`: 各 Fold Recall 趋势图
- `plot_signals_overview()`: 信号概览图（价格 + 买卖点标记）

## 文件清单

| 文件路径 | 操作 | 说明 |
|---------|------|------|
| `webapp/pages/04_primary_model.py` | 新增 | Primary Model 页面 |
| `webapp/utils/primary_model/__init__.py` | 新增 | 模块入口 |
| `webapp/utils/primary_model/base.py` | 新增 | 抽象基类 |
| `webapp/utils/primary_model/dual_ma.py` | 新增 | 双均线策略 |
| `webapp/utils/primary_model/tbm.py` | 新增 | TBM 标签计算 |
| `webapp/utils/primary_model/optimizer.py` | 新增 | Walk-Forward 优化器 |
| `webapp/components/primary_model_viz.py` | 新增 | 可视化组件 |
| `webapp/session.py` | 修改 | 扩展状态键 |
| `webapp/components/sidebar.py` | 修改 | 更新导航 |

## 依赖

- 现有依赖：`pandas`, `numpy`, `plotly`, `streamlit`
- 可选复用：`afmlkit.label.kit.TBMLabel`（如需更完整的 TBM 实现）

## 后续扩展

1. 新增其他 Primary Model 策略（RSI、布林带等）
2. 将成熟策略迁移至 `afmlkit/strategy/` 核心库
3. 支持 Numba 加速大规模计算