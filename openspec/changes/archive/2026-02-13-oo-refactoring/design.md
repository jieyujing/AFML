## Context

AFML 项目当前采用过程式编程风格,核心算法以独立函数形式实现(如 `process_bars.py` 中的 `generate_fixed_dollar_bars`, `generate_dynamic_dollar_bars`)。随着项目复杂度增加,这种风格带来以下问题:

- **状态管理混乱**: DataFrame 在函数间传递,缺乏统一的上下文管理
- **可测试性差**: 难以对独立功能进行单元测试
- **耦合度高**: 核心逻辑与具体实现紧耦合
- **复用困难**: 相同逻辑难以在不同场景复用

## Goals / Non-Goals

**Goals:**
1. 将核心模块重构为具有清晰职责的类
2. 实现状态封装,类内部管理中间状态
3. 保持向后兼容,提供统一的接口
4. 提高代码的可测试性和可维护性

**Non-Goals:**
1. 不改变核心算法的数学逻辑
2. 不添加新的外部依赖
3. 不重构测试代码(测试在 tests/ 目录)
4. 暂不实现完整的管道调度系统

## Decisions

### 1. 类层次结构设计

采用**组合优于继承**的模式,每个处理器类独立,通过数据流组合:

```
AFMLPipeline (Orchestrator, 可选)
    ├── DollarBarsProcessor
    ├── TripleBarrierLabeler
    ├── FeatureEngineer
    ├── SampleWeightCalculator
    └── BetSizer
```

### 2. 各模块类设计

#### DollarBarsProcessor
```python
class DollarBarsProcessor:
    def __init__(self, daily_target: int = 4, ema_span: int = 20):
        self.daily_target = daily_target
        self.ema_span = ema_span
        self.threshold_ = None  # 拟合参数

    def fit(self, df: pd.DataFrame) -> "DollarBarsProcessor":
        # 计算阈值参数

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 生成 dollar bars

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Sklearn 兼容接口
```

#### TripleBarrierLabeler
```python
class TripleBarrierLabeler:
    def __init__(
        self,
        pt_sl: List[float] = [1.0, 1.0],
        vertical_barrier_bars: int = 12,
        min_ret: float = 0.001
    ):
        self.pt_sl = pt_sl
        self.vertical_barrier_bars = vertical_barrier_bars
        self.min_ret = min_ret

    def fit(self, close: pd.Series) -> "TripleBarrierLabeler":
        # 计算波动率等参数

    def label(self, events: pd.DatetimeIndex, t1: pd.Series) -> pd.DataFrame:
        # 生成标签
```

#### FeatureEngineer
```python
class FeatureEngineer:
    def __init__(self, windows: List[int] = [5, 10, 20, 30, 50]):
        self.windows = windows
        self.selected_features_ = None

    def fit(self, df: pd.DataFrame, labels: pd.Series) -> "FeatureEngineer":
        # 可选:基于重要性的特征选择

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 生成特征

    def fit_transform(self, df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
        # Sklearn 兼容
```

#### SampleWeightCalculator
```python
class SampleWeightCalculator:
    def __init__(self, decay: float = 0.9):
        self.decay = decay
        self.concurrency_ = None
        self.uniqueness_ = None

    def fit(self, events: pd.DataFrame) -> "SampleWeightCalculator":
        # 计算并发性和唯一性

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # 添加权重列
```

#### BetSizer
```python
class BetSizer:
    def __init__(self, step_size: float = 0.0):
        self.step_size = step_size

    def calculate(
        self,
        events: pd.DataFrame,
        prob_series: pd.Series,
        pred_series: pd.Series = None,
        average_active: bool = False
    ) -> pd.Series:
        # 计算仓位大小
```

### 3. 配置文件设计

使用 YAML 文件定义默认参数:

```yaml
# config/processor_defaults.yaml
dollar_bars:
  daily_target: 4
  ema_span: 20

labeling:
  pt_sl: [1.0, 1.0]
  vertical_barrier_bars: 12
  min_ret: 0.001

features:
  windows: [5, 10, 20, 30, 50]

sample_weights:
  decay: 0.9

cv:
  n_splits: 5
  embargo: 0.01

bet_sizing:
  step_size: 0.0
```

### 4. 文件结构

```
src/
├── afml/
│   ├── __init__.py
│   ├── base.py              # 基类 (ProcessorMixin)
│   ├── dollar_bars.py       # DollarBarsProcessor
│   ├── labeling.py          # TripleBarrierLabeler
│   ├── features.py          # FeatureEngineer
│   ├── sample_weights.py    # SampleWeightCalculator
│   ├── cv.py                # PurgedKFoldCV (包装)
│   └── __init__.py          # 导出所有类
├── process_bars.py          # 保留为 CLI 入口(兼容)
├── labeling.py              # 保留为 CLI 入口(兼容)
├── ...
```

### 5. 向后兼容策略

保留原有的 CLI 脚本,但内部调用新的类:

```python
# process_bars.py (新)
from afml import DollarBarsProcessor

def main():
    processor = DollarBarsProcessor(daily_target=4)
    df = processor.fit_transform(raw_df)
```

## Risks / Trade-offs

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 重构引入 bug | 核心功能失效 | 保留原有 CLI 入口作为回归测试 |
| 学习曲线陡峭 | 开发效率短期下降 | 提供使用示例和文档 |
| 类设计过度复杂 | 难以维护 | 保持类职责单一,遵循 SRP |

## Migration Plan

1. **第一阶段**: 创建 `src/afml/` 目录和基类
2. **第二阶段**: 逐个实现 Processor 类
3. **第三阶段**: 更新 CLI 脚本调用新类
4. **第四阶段**: 验证输出与原版本一致

## Open Questions

1. 是否需要实现完整的 sklearn Pipeline 兼容接口?
2. 配置文件是否需要版本控制?
