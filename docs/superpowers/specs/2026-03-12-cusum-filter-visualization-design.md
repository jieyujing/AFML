# CUSUM Filter 可视化设计文档

## 1. 概述

为 WebUI 数据导入页面的 CUSUM filter 功能添加可视化图表，用于查看采样效果和采样率统计。

## 2. 目标

- 提供 CUSUM 采样效果的直观可视化
- 显示采样率统计数据
- 支持简单/高级模式切换

## 3. 设计方案

### 3.1 页面位置
**数据导入页面** (`webapp/pages/01_data_import.py`) - 在"构建 CUSUM K 线"按钮点击后显示

### 3.2 可视化仪表板（4 个图表）

1. **价格序列 + CUSUM 事件标记**
   - 主图显示价格走势
   - 用标记点/箭头标出 CUSUM 检测到的事件位置

2. **CUSUM 累积和曲线**
   - 显示 `s_pos` 和 `s_neg` 累积和随时间变化
   - 绘制阈值线作为参考

3. **动态阈值与波动率**
   - 显示滚动波动率变化
   - 显示动态阈值的變化

4. **采样率可视化**
   - 进度条显示采样比例
   - 指标面板显示关键数字

### 3.3 采样率显示

**指标面板内容：**
- 采样率百分比
- 原始行数 vs 采样后行数
- 平均事件频率（每天/每小时）
- 数据压缩比

**可视化元素：**
- 进度条直观显示采样比例

### 3.4 参数调节交互

**默认模式（简化界面）：**
- 仅显示核心参数
- 执行按钮

**高级模式（可展开）：**
- CUSUM 阈值 (`threshold`)
- 波动率窗口 (`vol_span`)
- 阈值乘数 (`threshold_multiplier`)
- 是否启用分数阶差分 (`use_frac_diff`)

## 4. 技术实现

### 4.1 依赖组件
- Streamlit (现有 WebUI 框架)
- Plotly (图表可视化)
- `afmlkit.sampling.filters.cusum_filter` (核心算法)

### 4.2 数据流
1. 用户点击"构建 CUSUM K 线"
2. 执行 CUSUM filter 算法
3. 返回事件索引和采样数据
4. 计算统计指标
5. 渲染可视化图表

### 4.3 图表组件
- 价格序列图：`plotly.graph_objects.Candlestick`
- 事件标记：`plotly.graph_objects.Scatter(mode='markers')`
- CUSUM 曲线：`plotly.graph_objects.Scatter`
- 波动率/阈值：`plotly.graph_objects.Scatter`

## 5. 成功标准

- [ ] 点击按钮后显示 4 个可视化图表
- [ ] 采样率指标正确计算并显示
- [ ] 高级模式可展开/收起
- [ ] 参数调整后可重新执行采样

## 6. 备注

- 遵循现有 WebUI 代码风格
- 保持与现有功能的兼容性
- 性能优化：大数据集时考虑降采样显示
