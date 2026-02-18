# Proposal: Add Dollar Imbalance Feature

## Why

当前 Dollar Bars 只聚合 OHLCV 数据，缺少市场微观结构信息。通过添加每个 bar 内的资金流向信息（买方 vs 卖方成交），可以：

1. 衡量每个 bar 的资金净流入/流出方向
2. 作为市场情绪指标
3. 配合 FFD 可提取更稳健的信号

## What Changes

在 `DollarBarsProcessor` 的输出中添加三个新列：

- `buyer_amount`: 该 bar 内买方主动成交的美元金额 (is_buyer_maker=False)
- `seller_amount`: 该 bar 内卖方主动成交的美元金额 (is_buyer_maker=True)  
- `dollar_imbalance`: 净流入 = buyer_amount - seller_amount

## Capabilities

### New Capabilities
- `dollar_imbalance`: 每个 Dollar Bar 的资金净流入

### Modified Capabilities
- `transform()`, `_transform_fixed()`, `_transform_dynamic()`, `transform_chunked()`: 添加新列

## Impact

- `src/afml/dollar_bars.py`: 修改聚合逻辑
