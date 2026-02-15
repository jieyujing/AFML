# Design: BTCUSDT Dollar Bars with JB Testing

## Context

当前项目已有:
- `DollarBarsProcessor`: 支持 `daily_target` 参数控制每日条柱数
- `_compute_jb_statistics()`: 计算 JB 统计量的函数
- `afml_polars_pipeline.py`: 已有完整 pipeline
- BTCUSDT parquet 数据: 2023-2026 年的分笔数据

## Goals / Non-Goals

**Goals:**
- 在 pipeline 中添加 `daily_target` 参数测试循环
- 测试 [4, 20, 50, 100] 不同值
- 收集并分析 JB 测试结果

**Non-Goals:**
- 不修改 DollarBarsProcessor 核心逻辑

## Decisions

### Decision 1: 扩展现有 pipeline

直接在 `afml_polars_pipeline.py` 中添加参数扫描功能，复用现有代码。

### Decision 2: 测试参数选择

选择 [4, 20, 50, 100] 代表:
- 4: 当前默认值
- 20: 较低
- 50: 用户指定
- 100: 较高

### Decision 3: 输出格式

生成对比表格，清晰展示:
| daily_target | bars_count | jb_stat | p_value | skewness | kurtosis | is_normal |

---

**Design 很简单**——现有组件已经完备，我们只需要编写胶水代码把它们串联起来。
