# Dollar Bars with Jarque-Bera Testing Specification

## ADDED Requirements

### Requirement: 测试不同 daily_target 参数 SHALL

通过循环测试不同 daily_target 值，找出最优参数。

#### Scenario: 使用 daily_target=50 生成 Dollar Bars

- **WHEN** 运行 pipeline 并传入 `daily_target=50`
- **THEN** SHALL 生成符合以下条件的 dollar bars:
  - 每日约有 50 根条柱
  - 包含 datetime, open, high, low, close, volume, amount 列
- **AND** 返回生成的条柱数量用于统计

#### Scenario: 循环测试多个参数值

- **WHEN** 运行测试循环: daily_target in [4, 20, 50, 100]
- **THEN** SHALL 对每个参数值:
  - 生成对应的 dollar bars
  - 计算 JB 统计量
  - 收集结果

#### Scenario: Jarque-Bera 正态性测试

- **WHEN** 对 dollar bars 的收盘价计算对数收益率后运行 JB 测试
- **THEN** SHALL 返回包含以下字段的字典:
  - `jb_stat`: JB 统计量
  - `p_value`: p 值
  - `skewness`: 偏度
  - `kurtosis`: 峰度 (Pearson, 正态分布=3)
  - `is_normal`: p_value > 0.05 时为 True

#### Scenario: 参数分析

- **WHEN** 收集了多个 daily_target 的 JB 结果
- **THEN** SHALL 输出:
  - 每个参数的 JB 统计量对比表
  - p_value 最高的参数值
  - 偏度和峰度的趋势分析

### Requirement: JB 统计量可视化 SHALL

生成 JB 测试结果的可视化图表。

#### Scenario: p_value 趋势图

- **WHEN** 完成多参数测试后
- **THEN** SHALL 生成折线图展示 p_value 随 daily_target 变化的趋势
- **AND** 图表保存到 visual_analysis/

#### Scenario: 偏度峰度对比图

- **WHEN** 完成多参数测试后
- **THEN** SHALL 生成柱状图对比 skewness 和 kurtosis

---

## Acceptance Criteria

1. 脚本成功加载 `data/BTCUSDT/parquet_db/` 下的 parquet 文件
2. 使用 `daily_target=50` 成功生成 dollar bars
3. JB 测试输出完整的统计信息
4. 根据 p_value 提供清晰的结论
