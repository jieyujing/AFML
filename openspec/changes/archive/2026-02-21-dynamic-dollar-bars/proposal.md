## Why

传统的 Dollar Bars 通常使用固定阈值，然而这无法适应市场波动率及交易量的周期性更迭。通过让基准阈值（如设定的日内提取频率倒推得出）跟随历史交易窗口平滑更新（EWMA 动态阈值），可以生成更纯净、更能恢复正态性和降低序列自相关性的高信息量 Dollar Bars 样本。结合严格的统计检验（JB Test + 自相关检验），这不仅让特征工程更可靠，也严格遵守了量价研究中的稳健性规则。

## What Changes

开发一个用于生成并评估最佳动态频率 Dollar Bars 的端到端分析管线（独立脚本），流程包括：
1. 日均总交易额提取与基础阈值推演。
2. EWMA 在时序上的动态映射机制应用。
3. 统计指标测算：JB 检验值越低（趋近正态）、一阶自相关度越低则越强。
4. 提供美观的量价时间序列数据可视化反馈。

## Capabilities

### New Capabilities
- `ewma-dynamic-bars`: 提供自动伸缩的动态阈值采样机制，以保证平稳性。
- `statistical-evaluation`: 引入 JB test 检验及自相关性等评估工具对时间序列进行定量评分。
- `bar-visualizer`: 基于数据聚合对最优阈值分布进行展示，与框架原生对象保持同构。

### Modified Capabilities

## Impact

- 主要增加 `scripts/dynamic_dollar_bars.py` 动态 Dollar Bar 生成与检验脚本。
- `afmlkit.bar.kit` 的 `DollarBarKit` 已支持一定的阈值（当前可能为固定），我们将通过外部传入一个能映射每个 tick 的动态 threshold 列表来实现，或者直接调用底层的 `_dollar_bar_indexer` 配合时变阈值（需检查底层是否支持动态）。
