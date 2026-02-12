## Why

Dollar Bars 是 AFML 方法论的核心组件，用于生成统计上更可靠的金融数据。然而，当前可视化模块只显示 bar 统计、收益率分布和 K 线图，没有提供 Jarque-Bera (JB) 统计量来验证 dollar bars 是否真正符合 AFML 标准——即使收益率分布更接近正态分布。缺乏 JB 统计量会导致用户无法量化验证 dollar bars 的有效性。

## What Changes

1. **新增 JB 统计量可视化功能**：
   - 为 dollar bars 计算并展示 Jarque-Bera 统计量
   - 计算偏度(Skewness)和峰度(Kurtosis)
   - 与时间序列 bars 进行对比
   - 显示 JB p-value 以验证正态性假设

2. **修改 `plot_bar_stats` 方法**：
   - 添加 JB 统计信息面板
   - 添加统计矩对比表格
   - 添加正态性验证结论

3. **更新 AFMLVisualizer 类**：
   - 新增 `plot_jb_statistics()` 方法
   - 支持与时间序列 bars 对比模式

## Capabilities

### New Capabilities
- `jb-statistics-visualization`: 新增 JB 统计量可视化能力，包括 JB 统计量、偏度、峰度计算和正态性验证结果展示

### Modified Capabilities
- `visual-analysis`: 修改现有可视化能力，在 `plot_bar_stats` 方法中集成 JB 统计量显示

## Impact

- **代码影响**：`src/afml/visualization.py` - 修改 `AFMLVisualizer.plot_bar_stats()` 方法
- **Pipeline 影响**：`src/afml_polars_pipeline.py` - 调用新增的 JB 统计可视化
- **测试影响**：需添加 JB 统计可视化单元测试
- **依赖**：`scipy.stats.jarque_bera` 用于 JB 检验计算
