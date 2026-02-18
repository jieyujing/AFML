# Tasks: Add Dollar Imbalance Feature

## 1. Implement Dollar Imbalance

- [x] 1.1 在 `_transform_fixed()` 中添加 buyer_amount, seller_amount, dollar_imbalance 聚合
- [x] 1.2 在 `_transform_dynamic()` 中添加相同聚合
- [x] 1.3 在 `transform_chunked()` 中添加相同聚合

## 2. Verify

- [x] 2.1 测试数据验证计算正确
- [x] 2.2 验证没有 is_buyer_maker 字段时的向后兼容性
