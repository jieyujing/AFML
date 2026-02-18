# Tasks: Add Cumulative Money Flow Feature

## 1. Implement Cumulative Money Flow

- [x] 1.1 在 `load_zip_as_df()` 函数中添加 cum_money_flow 计算
- [x] 1.2 确保数据按 timestamp 排序后再计算 cumsum

## 2. Verify

- [x] 2.1 验证新字段出现在输出的 parquet 文件中
- [x] 2.2 验证 cum_money_flow 计算正确（抽样检查）
