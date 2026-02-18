# Design: Add Cumulative Money Flow Feature

## Context

在原始交易数据中，每笔交易有以下关键信息：
- `amount`: 美元金额 (price × quantity)
- `is_buyer_maker`: 买卖方向
  - `False` = 买方主动成交 → 资金流入 (+)
  - `True` = 卖方主动成交 → 资金流出 (-)

## Goals / Non-Goals

**Goals:**
- 新增 `cum_money_flow` 列到 parquet 数据中
- 计算公式: `cum_money_flow = cumsum(amount × sign)`

**Non-Goals:**
- 不修改现有的 dollar bar 生成逻辑
- 不添加 FFD 处理（在特征工程阶段做）

## Decisions

### Decision 1: 计算位置

**选择**: 在 `load_zip_as_df()` 中计算

```python
# signed amount: +amount for buyer, -amount for seller
signed_amount = pl.when(pl.col("is_buyer_maker") == False)
    .then(pl.col("amount"))
    .otherwise(-pl.col("amount"))

# cumulative money flow
cum_money_flow = signed_amount.cum_sum()
```

**理由**: 在数据加载阶段计算可以保留到最终的 parquet 文件中。

### Decision 2: 数据排序

**选择**: 按 timestamp 排序后计算 cumsum

**理由**: 累积计算必须基于时间顺序，否则没有物理意义。
