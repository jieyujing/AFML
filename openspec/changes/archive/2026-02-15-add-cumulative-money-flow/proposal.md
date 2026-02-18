# Proposal: Add Cumulative Money Flow Feature

## Why

当前 raw_data_polars.parquet 包含 `is_buyer_maker` 字段（买卖方向）和 `amount` 字段（美元金额）。需要新增一个累积资金流特征，用于：

1. 衡量市场资金流入/流出方向
2. 作为市场微观结构特征的基础
3. 可进一步做 FFD 处理提取更稳健的信号

## What Changes

在 `src/merge_aggtrades.py` 的数据处理流水线中，新增 `cum_money_flow` 列：

- `cum_money_flow`: 累积资金流 = Σ(amount × sign)
  - sign = +1 when is_buyer_maker = False (买方主动，资金流入)
  - sign = -1 when is_buyer_maker = True (卖方主动，资金流出)

## Capabilities

### New Capabilities
- `cum_money_flow`: 累积美元资金流（带方向）

### Modified Capabilities
- `merge_to_monthly_parquet()`: 在加载数据时计算并保留该字段

## Impact

- `src/merge_aggtrades.py`: 修改 `load_zip_as_df()` 函数
- `data/raw_data_polars.parquet`: 重新生成后包含新字段
