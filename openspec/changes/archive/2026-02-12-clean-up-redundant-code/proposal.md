## Why

项目当前存在以下冗余问题需要整理：

1. **Legacy 代码残留**: `src/legacy/` 目录包含旧版实现，这些脚本引用新的 `afml` 包，说明已废弃但未清理
2. **入口文件臃肿**: `src/afml_polars_pipeline.py` (953行) 包含完整 pipeline 逻辑，适合作为示例但不适合作为唯一入口
3. **目录结构不清晰**: 主入口与核心库混在一起，建议分离演示脚本与核心代码

清理冗余代码可以简化项目结构，降低维护成本，提高代码可读性。

## What Changes

- **删除废弃代码**: 移除 `src/legacy/` 目录及其所有文件
- **重构入口文件**: 将 `src/afml_polars_pipeline.py` 移动到 `examples/` 目录
- **更新引用**: 更新 `pyproject.toml` 的脚本入口配置
- **简化项目结构**: 保持 `src/afml/` 为唯一核心代码目录

## Capabilities

### New Capabilities

此变更不引入新功能，仅进行代码整理。

### Modified Capabilities

无 - 仅进行清理和重构，不改变任何现有功能的规格。

## Impact

- **删除**: `src/legacy/` 目录（15个文件）
- **移动**: `src/afml_polars_pipeline.py` → `examples/afml_polars_pipeline.py`
- **更新**: `pyproject.toml` 脚本配置
- **影响**: 用户运行命令从 `uv run python src/afml_polars_pipeline.py` 改为 `uv run python examples/afml_polars_pipeline.py`
