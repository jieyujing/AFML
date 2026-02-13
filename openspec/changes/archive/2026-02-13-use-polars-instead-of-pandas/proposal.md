# Use Polars Instead of Pandas

## Why

当前项目使用 pandas 作为主要数据处理库，在处理大规模金融时间序列数据时存在严重的内存瓶颈和性能问题。Pandas 的内存占用高、运算速度慢，特别是在处理数十万到数百万级别的价格数据时，经常导致内存溢出（OOM）和长时间的计算等待。Polars 是一个高性能的 DataFrame 库，基于 Rust 实现，具有更低的内存占用和更快的执行速度，能够有效解决这些问题。

## What Changes

- **新增 Polars 依赖**：在 `pyproject.toml` 中添加 `polars>=1.0.0` 依赖
- **重写核心模块**：将所有 19 个使用 pandas 的模块迁移到 Polars
- **新增 OO API 模块**：在 `src/afml/` 目录下创建 Polars 版本的类
- **保持向后兼容**：提供与 sklearn 兼容的接口（`fit()`、`transform()`、`fit_transform()`）
- **优化内存使用**：利用 Polars 的惰性求值和增量处理能力
- **支持 LazyFrame**：处理超大规模数据集时避免一次性加载到内存

## Capabilities

### New Capabilities

- **polars-dollar-bars**: 使用 Polars 生成 Dollar Bars，支持惰性求值和流式处理
- **polars-triple-barrier**: 使用 Polars 实现三重障碍标签生成，性能提升 2-5x
- **polars-feature-engineer**: 使用 Polars 进行特征工程，支持大规模特征计算
- **polars-sample-weights**: 使用 Polars 计算样本权重，内存效率提升 3-10x
- **polars-cv**: 使用 Polars 实现 Purged K-Fold 交叉验证
- **polars-meta-labeling**: 使用 Polars 的元标签策略管线
- **polars-bet-sizer**: 使用 Polars 实现仓位管理

### Modified Capabilities

- 现有的 `dollar-bars-processor`、`triple-barrier-labeler`、`feature-engineer`、`sample-weight-calculator`、`meta-labeling-pipeline`、`bet-sizer` 规格需要更新为 Polars 版本
- 所有现有的 pandas-based specs 需要创建对应的 Polars 兼容版本

## Impact

- **代码文件影响**：19 个 Python 文件需要迁移（src/ 目录 12 个，src/afml/ 目录 7 个）
- **依赖变更**：新增 `polars` 依赖，移除或标记 pandas 为可选
- **API 兼容**：保持与 sklearn Pipeline 的兼容性
- **测试覆盖**：需要为所有迁移后的模块添加单元测试
- **性能提升**：预期内存使用降低 50-80%，运算速度提升 2-5x
