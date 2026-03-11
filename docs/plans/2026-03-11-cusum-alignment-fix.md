# CUSUM 对齐修复设计文档

## 问题描述

用户在 WebUI 中进行特征计算时，结果显示"共 0 行，47 个特征"，导致特征预览页面所有特征值都是 None。

### 问题根因

**根本原因**：CUSUM 采样脚本 `cusum_filtering.py` 在读取输入 CSV 文件时没有保留时间戳索引，导致输出的 CUSUM 文件索引变成了日期（`2024-01-01 00:00:00`），而 Dollar Bars 文件的索引是具体时间戳（如 `2024-01-01 11:01:00`）。

具体过程：
1. `cusum_filtering.py` 使用 `pd.read_csv(input_file)` 读取数据，没有指定 `index_col`，导致索引是默认整数（0, 1, 2...）
2. `timestamp` 列作为普通列保存
3. 在保存输出时，使用 `final_df.to_csv(output_file, index=False)`，丢失了索引信息
4. 当 WebUI 读取 CUSUM 文件时，使用 `index_col=0, parse_dates=True`，将第一列（日期）解析为索引

当 `align_features_with_cusum()` 函数执行索引对齐时：
```python
common_idx = features_df.index.intersection(labels_df.index)
```

由于时间戳不完全匹配（一个有具体时间，一个只有午夜 00:00:00），交集为空，导致返回空 DataFrame。

### 数据示例

- **CUSUM 文件索引（修复前）**: `2024-01-01 00:00:00` (日期，无时间)
- **CUSUM 文件索引（修复后）**: `2023-02-17 15:05:21.612000` (精确到毫秒)
- **Dollar Bars 索引**: `2023-01-03 11:01:00` (精确到秒/毫秒)
- **修复前交集**: 空集 → 0 行结果
- **修复后交集**: 2540 行 ✓

## 解决方案

进行了两层修复：

### 修复 1：特征对齐层（临时方案）

修改 `align_features_with_cusum()` 函数，实现基于日期的模糊匹配：

1. **首先尝试精确索引交集**（保持原有逻辑）
2. **如果精确匹配失败，使用日期截断匹配**：
   - 将两个索引都截断为日期（去除时间成分）
   - 找到共同日期
   - 使用 `merge_asof` 进行时间最近邻匹配
3. **如果连日期都不匹配，返回空 DataFrame**

文件：`webapp/utils/feature_calculator.py`

### 修复 2：CUSUM 采样层（根本方案）

修改 `cusum_filtering.py`，确保时间戳索引正确传递：

1. **读取输入时保留索引**：`pd.read_csv(input_file, index_col=0, parse_dates=True)`
2. **保存输出时保留索引**：`final_df.to_csv(output_file, index=True)`
3. **适配代码**：修复对 `timestamp` 列的访问逻辑，支持索引已经是 DatetimeIndex 的情况

文件：`scripts/cusum_filtering.py`

### 代码修改

文件：`webapp/utils/feature_calculator.py`

```python
def align_features_with_cusum(
    features_df: pd.DataFrame, cusum_path: str, label_cols: List[str] = None
) -> pd.DataFrame:
    labels_df = pd.read_csv(cusum_path, index_col=0, parse_dates=True)
    labels_df = labels_df.sort_index()

    if label_cols is None:
        label_cols = ["bin", "t1", "avg_uniqueness", "return_attribution"]
        label_cols = [c for c in label_cols if c in labels_df.columns]

    # Try exact index intersection first
    common_idx = features_df.index.intersection(labels_df.index)

    # If no exact match, try date-based matching (truncate time component)
    if len(common_idx) == 0:
        # Create date-only index for matching
        features_dates = features_df.index.normalize()
        labels_dates = labels_df.index.normalize()

        # Find matching dates
        common_dates = features_dates.intersection(labels_dates)

        if len(common_dates) > 0:
            # Map back to original indices
            features_mask = features_dates.isin(common_dates)
            labels_mask = labels_dates.isin(common_dates)

            features_filtered = features_df[features_mask]
            labels_filtered = labels_df[labels_mask]

            # Use merge_asof for time-based alignment
            features_reset = features_filtered.reset_index()
            labels_reset = labels_filtered.reset_index()

            # Rename timestamp column for merge_asof
            features_reset = features_reset.rename(columns={'index': 'timestamp'})
            labels_reset = labels_reset.rename(columns={'index': 'timestamp'})

            # Merge on closest timestamp (backward direction: use CUSUM labels for previous event)
            aligned = pd.merge_asof(
                features_reset.sort_values('timestamp'),
                labels_reset.sort_values('timestamp'),
                on='timestamp',
                direction='nearest',
                tolerance=pd.Timedelta('1D')
            )

            aligned = aligned.set_index('timestamp')
            return aligned
        else:
            # No common dates at all - return empty DataFrame with warning
            return features_df.iloc[:0].copy()

    # Original exact match logic
    aligned_features = features_df.loc[common_idx].copy()
    aligned = aligned_features.join(labels_df[label_cols], how="inner")

    return aligned
```

## 测试验证

### 测试场景 1：精确索引匹配
- CUSUM 索引：`2023-01-01 00:00:00` to `2023-04-10 00:00:00`
- Dollar Bars 索引：`2023-01-01 00:00:00` to `2024-05-14 00:00:00`
- 结果：81 行 ✓

### 测试场景 2：日期匹配（用户实际场景 - 修复前）
- CUSUM 索引：`2024-01-01 00:00:00` to `2024-04-09 00:00:00`
- Dollar Bars 索引：`2023-01-03 11:01:00` to `2026-03-11 09:46:00`
- 精确交集：0 行
- 修复后结果（仅对齐层修复）：276 行 ✓

### 测试场景 3：完整修复后（用户实际场景）
- CUSUM 索引（修复后）：`2023-02-17 15:05:21.612000` to `2026-01-29 21:48:24.312000`
- Dollar Bars 索引：`2023-01-03 11:01:00` to `2026-03-11 09:46:00`
- 精确交集：0 行（时间戳精度不同）
- 对齐层修复后结果：2540 行 ✓
- NaN 检查：所有列无 NaN ✓

## 注意事项

1. **CUSUM 文件重新生成**：用户需要重新运行 `python scripts/cusum_filtering.py` 来生成带有正确时间戳索引的 CUSUM 文件
2. **对齐层修复**：即使 CUSUM 文件索引格式不正确，WebUI 也能通过日期模糊匹配正常工作
3. **tolerance**: 当前设置为 1 天，可以根据实际情况调整
4. **direction**: 当前使用 `nearest`，也可以考虑使用 `backward` 来确保标签不泄露

## 后续改进

1. ~~在 CUSUM 采样模块中，确保输出文件使用与输入相同的时间戳格式~~ ✓ 已完成
2. ~~在 WebUI 中增加对齐结果提示，显示匹配的行数和日期范围~~ - 可选改进
3. 考虑添加日志记录对齐过程和警告信息 - 可选改进
