## Context

Dollar Bars 是 AFML 方法论的核心组件，通过将交易聚合到固定美元金额阈值来生成统计上更可靠的金融数据。AFML 方法论要求 dollar bars 的收益率分布应该比时间序列 bars 更接近正态分布，这通过 Jarque-Bera (JB) 检验来验证。当前 `src/afml/visualization.py` 中的 `plot_bar_stats()` 方法虽然显示收益率分布直方图，但缺少 JB 统计量、偏度和峰度的数值显示，导致用户无法量化验证 dollar bars 的有效性。

## Goals / Non-Goals

**Goals:**
- 在 `plot_bar_stats()` 方法中集成 JB 统计量显示
- 计算并显示 Jarque-Bera 统计量、p-value、偏度(Skewness)和峰度(Kurtosis)
- 提供正态性验证结论（是否通过 JB 检验）
- 与时间序列 bars 的统计指标进行对比（如有传入）

**Non-Goals:**
- 不修改 dollar bars 生成逻辑（`DollarBarsProcessor`）
- 不添加新的交互式可视化组件
- 不修改 pipeline 的核心执行流程

## Decisions

1. **JB 统计集成方式：修改现有 `plot_bar_stats()` 方法**
   - 理由：保持可视化模块的单一职责原则，避免创建过多小型方法
   - 备选方案：创建独立的 `plot_jb_statistics()` 方法 → 被否决，增加方法数量但功能单一

2. **新增统计面板**
   - 在图表下方添加 JB 统计信息表格
   - 包含：JB Statistic、p-value、Skewness、Kurtosis
   - 备注：正态分布的 Skewness=0，Kurtosis=3

3. **JB 检验实现**
   - 使用 `scipy.stats.jarque_bera()` 计算 JB 统计量和 p-value
   - 使用 `scipy.stats.skew()` 和 `scipy.stats.kurtosis()` 计算矩统计量
   - 理由：与 `src/legacy/process_bars.py` 中的 `analyze_normality()` 保持一致

4. **返回值设计**
   - 方法返回包含 JB 统计量的字典
   - 包含：jb_stat、p_value、skewness、kurtosis、is_normal
   - 理由：便于 pipeline 日志记录和自动化验证

## Risks / Trade-offs

- **[风险] 计算性能**: JB 检验在大数据集上可能稍慢 → **缓解**：只对已聚合的 dollar bars 计算，数据量通常较小（< 10k bars）
- **[风险] 与现有代码重复**: `src/legacy/process_bars.py` 中已有 `analyze_normality()` 函数 → **缓解**：新实现使用 Polars/NumPy 保持一致，短期内共存，长期可考虑重构
- **[风险] 可读性**: 增加统计面板可能使图表拥挤 → **缓解**：使用紧凑的表格布局，在图表下方而非占用主要绘图空间
