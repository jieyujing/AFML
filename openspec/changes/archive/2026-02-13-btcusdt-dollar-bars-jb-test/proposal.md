# Proposal: BTCUSDT Dollar Bars with Jarque-Bera Testing

## Why

我们需要验证 Dollar Bars 的 `daily_target` 参数对条柱统计特性的影响。根据 AFML 理论，合适的条柱参数应该产生接近正态分布的收益率（通过 Jarque-Bera 测试检验）。

## What Changes

- 使用 `daily_target=50` 生成 dollar bars
- 编写测试脚本加载 BTCUSDT parquet 数据
- 对收益率运行 Jarque-Bera 测试
- 根据结果分析参数是否需要调整

## Capabilities

### 新增 Capabilities
- `jb_test_runner.py`: 运行 JB 测试的脚本

### 修改 Capabilities
- `DollarBarsProcessor`: 使用 `daily_target=50` 参数

## Impact

- `data/BTCUSDT/parquet_db/`: 输入数据
- `src/afml/dollar_bars.py`: 处理器
- `src/afml/visualization.py`: JB 统计函数已有
