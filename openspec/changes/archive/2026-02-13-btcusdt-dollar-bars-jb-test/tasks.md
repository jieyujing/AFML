# Tasks: BTCUSDT Dollar Bars JB Test

## 1. 修改 pipeline 添加 daily_target 参数测试 + 可视化

- [x] 1.1 修改 `run_pipeline()` 函数，添加 `daily_target` 参数支持（已有）
- [x] 1.2 在主函数中添加循环测试不同 daily_target 值 (4, 20, 50, 100)
- [x] 1.3 收集每个参数的 JB 测试结果

## 2. 添加 JB 统计量可视化

- [x] 2.1 添加 p_value vs daily_target 折线图
- [x] 2.2 添加 skewness/kurtosis 对比柱状图
- [x] 2.3 输出结果表格

## 3. 运行测试

- [x] 3.1 使用 `daily_target=50` 运行 pipeline
- [x] 3.2 验证 JB 测试输出和可视化
- [x] 3.3 测试不同参数值

## 4. 分析结果

- [x] 4.1 分析 p_value vs daily_target 关系
- [x] 4.2 找出最优的 daily_target
- [x] 4.3 记录结论

---

每个 checkbox 在 apply 阶段成为一个工作单元。准备好开始实现了么？
