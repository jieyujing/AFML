## Context

当前 AFML 项目使用 pandas 作为主要数据处理库，已实现的功能包括 Dollar Bars 生成、三重障碍标签、特征工程、样本权重计算、交叉验证和元标签策略。Pandas 在处理大规模金融时间序列数据时存在以下问题：

- **内存占用高**：Pandas DataFrame 默认加载全部数据到内存，无法有效处理 GB 级别的时间序列
- **运算速度慢**：GIL 锁限制导致多核 CPU 利用率低
- **惰性求值缺失**：无法优化查询计划和增量处理
- **大数据集处理瓶颈**：超过 100 万行数据时性能急剧下降

Polars 是用 Rust 编写的 DataFrame 库，提供了：
- 零拷贝读取，内存效率提升 3-10x
- 多线程并行处理，速度提升 2-5x
- 惰性求值支持，优化查询计划
- 与 pandas 相似的 API，易于迁移

## Goals / Non-Goals

**Goals:**
- 将所有核心数据处理模块迁移到 Polars
- 保持与 sklearn Pipeline 的兼容性
- 实现惰性求值和流式处理，支持超大规模数据集
- 内存使用降低 50% 以上，运算速度提升 2-5x
- 保持现有功能和测试覆盖

**Non-Goals:**
- 不删除 pandas 依赖（保持与第三方库的兼容性）
- 不重写非数据处理的辅助代码（如可视化、模型训练）
- 不改变核心算法逻辑（仅迁移实现）
- 不提供 pandas 到 Polars 的自动转换工具

## Decisions

### Decision 1: 使用 Polars 作为主要 DataFrame 库

**选择 Polars 而非其他方案：**

- **vs Pandas**：性能优势明显，内存效率高 3-10x
- **vs Modin/Dask**：无需额外集群配置，单机性能更优
- **vs Vaex**：API 更成熟，与 pandas 兼容性好
- **vs PySpark**：更适合单机场景，避免集群开销

**理由**：Polars 在单机性能、内存效率和 API 成熟度之间取得最佳平衡。

### Decision 2: 保持 sklearn 兼容接口

**所有处理器类实现以下方法：**
- `fit(X)` - 拟合并存储内部参数
- `transform(X)` - 转换数据
- `fit_transform(X)` - 拟合并转换

**理由**：确保与 sklearn Pipeline 无缝集成。

### Decision 3: 支持 DataFrame 和 LazyFrame 两种模式

**策略：**
- 默认使用 DataFrame（简单场景）
- 提供 `lazy=True` 参数启用 LazyFrame
- 大数据集自动使用惰性求值

**理由**：平衡易用性和性能。

### Decision 4: API 设计保持一致性

**命名约定：**
- 类名：`PolarsDollarBarsProcessor`、`PolarsTripleBarrierLabeler`
- 方法：`fit()`、`transform()`、`fit_transform()`
- 参数：与现有 pandas 版本保持一致

**理由**：降低迁移成本。

### Decision 5: 渐进式迁移策略

**迁移顺序：**
1. 基础数据处理：`polars-dollar-bars`
2. 标签生成：`polars-triple-barrier`
3. 特征工程：`polars-feature-engineer`
4. 权重计算：`polars-sample-weights`
5. 交叉验证：`polars-cv`
6. 元标签和仓位：`polars-meta-labeling`、`polars-bet-sizer`

**理由**：从简单模块开始，逐步验证。

## Risks / Trade-offs

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| **第三方库兼容性** | 部分依赖 pandas 的库可能不兼容 Polars | 保持 pandas 可选依赖，必要时使用转换函数 |
| **API 差异** | 部分 pandas API 在 Polars 中不存在或不同 | 使用 polars-pandas 兼容层，标注差异 |
| **学习曲线** | 团队需要时间熟悉 Polars | 提供迁移文档，标注 API 对照 |
| **测试覆盖** | 新代码可能引入 bug | 保持 100% 测试覆盖，对比 pandas 结果 |
| **类型提示** | Polars 类型系统与 pandas 不同 | 使用严格的类型检查，逐步添加类型 |

**Trade-offs：**
- 选择性能而非完全的 pandas API 兼容性
- 接受部分场景需要显式转换（如 I/O 操作）
- 优先支持 DataFrame API，LazyFrame 作为高级功能

## Migration Plan

### Phase 1: 准备阶段

1. 添加 Polars 依赖到 `pyproject.toml`
2. 创建 `src/afml/polars/` 目录结构
3. 编写 pandas 到 Polars 的迁移指南

### Phase 2: 逐模块迁移

按决策 5 的顺序，每个模块迁移步骤：
1. 读取现有 pandas 源码
2. 转换为 Polars 实现
3. 保持相同的功能测试
4. 添加性能基准测试
5. 创建使用示例

### Phase 3: 验证和优化

1. 运行完整测试套件
2. 对比性能基准
3. 修复迁移中的问题
4. 优化热点代码

### Rollback Plan

- 使用 Git 标签标记每个阶段的完成状态
- 保留 pandas 版本的旧模块（重命名为 `*_pandas.py`）
- 通过 feature flag 控制使用 Polars 还是 pandas 版本

## Open Questions

1. **I/O 格式支持**：是否需要支持 CSV/Parquet 的 Polars 原生读写？
2. **混合使用策略**：在同一个 Pipeline 中混合使用 pandas 和 Polars 的最佳实践？
3. **性能基准**：是否需要建立完整的性能基准测试套件？
4. **弃用计划**：pandas 版本何时标记为 deprecated？
