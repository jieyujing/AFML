# Technical Design

## Context
<!-- Background context for the technical design -->
在 AFML 框架下，基于时间或固定额度采样的 Bars 可能会残留大量微观结构噪音，导致金融特征过度拟合。为此需要引入 CUSUM 过滤器实现事件驱动的二次提取。项目目标是实现一个自动化的管道，读取 `outputs/dollar_bars/dollar_bars_freq20.csv` 文件，根据近期滚动标准差（波动率 $\sigma$）计算出具有自适应特性的动态过滤阈值（设定阈值 $h = 2 \times \sigma$），并只在价格偏差超过此阈值时生成有效事件触发点。

## Architecture
<!-- System architecture and component interactions -->
该工作流可以拆分为四个主要处理模块：
1. **Data Ingestion**：通过 Pandas 加载目标 Dollar Bars CSV 文件。为了性能加速，可以提前设定要映射的类型并且仅提取闭盘价（`close`）。
2. **Signal Generation ($\sigma$ Target)**：
   - 提取对数收益率序列（Log Returns）。
   - 借助 `afmlkit.feature.core.volatility.ewms` 的流式对数收益序列计算长度为 N (例如 N=100) 的滚动/指数特征标准差 $\sigma_t$。
3. **Event Detection (CUSUM Filter)**：
   - 生成动态阈值序列 $h = \sigma \times 2.0$。
   - 调用 `afmlkit.sampling.filters.cusum_filter` 函数执行对称的 CUSUM 过滤探测。
4. **Sub-sampling & Persistence**：
   - 依赖被标记检测出的行索引提取过滤后的 DataFrame 子集。
   - 分发或存储提纯后的数据集供三重屏障法进行数据标记并流转至后续管道。

## API Design
<!-- Interfaces, data structures, and endpoints -->
我们将在 `scripts/` 目录下创建一个名为 `cusum_sampling_workflow.py` 的主流程脚本。核心的函数抽象如下：

```python
import pandas as pd
import numpy as np
from typing import Optional

def compute_dynamic_cusum_filter(
    df: pd.DataFrame, 
    price_col: str = 'close', 
    vol_span: int = 50, 
    threshold_multiplier: float = 2.0
) -> pd.DataFrame:
    """
    通过 CUSUM 对时间序列的微观波动去噪
    """
    pass
```

## Implementation Details
<!-- Specific implementation decisions and algorithms -->
1. **NaN 初始化对齐问题**：
   在计算指数加权标准差（EWM_STD）时，序列前期的几组初始值为 `NaN`。必须首先用前推边界值进行 `bfill` 或替换为某个有意义的先验方差常数，避免将 NaN 值传递给底层的 JIT 代码并抛出错误。
2. **Log Return 微积分**：
   `cusum_filter` 在其内部使用了 Log Returns 计算累计和。因此对于外界阈值的生成逻辑中，波动率必须也是针对 Log Returns 而非简单收益率计算得出。
3. **Numba 执行环境**：
   需保证所有传入 `cusum_filter` 的数组参数确切为 Numpy 数组，并且以 Float64/Float32 类型加载，以防引起 Numba 编译兼容性故障（例如隐式传入 Pandas Series 将抛出异常）。

## Risks & Mitigations
<!-- Known risks and their mitigation strategies -->
- **过度采样缺失 (Starvation)**: 当倍数为 $k=2.0$ 相对较高时，可能导致极少数量的有效事件被采纳，进一步致使样本严重匮乏。
  **缓解方案**: 在运行后统计原始数据条数和过滤后条数的比率。建议比率介于预期的数据周期内（如一天能够获得 5-10 个事件点）。将该 $k$ 定义为一个可调超参，便于后续校准。
- **性能问题**：对庞大的 CSV 每行做全量数据滚屏操作。
  **缓解方案**：由于 `afmlkit` 内部大量使用了 `jnjit(nogil=True)` 的并行化或机器码加速函数。直接转到 Numpy 环境中进行数据喂送（Vectorization）将不会产生 CPU 瓶颈。

## Alternative Approaches
<!-- Other considered approaches and why they were rejected -->
- 采用基于固定绝对跌幅和涨幅的 CUSUM 模型：由于宏观经济周期切换中的市场波动率是变化的，一个固定的阈值在波动爆发时会过于灵敏造成信噪比下降，在低波荡期又会错过全部变异点。故坚决使用针对滚动波动率的自适应（动态）动态阈值设计。
