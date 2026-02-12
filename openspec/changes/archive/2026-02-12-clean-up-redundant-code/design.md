## Context

当前项目结构存在以下问题：
- `src/legacy/` 目录包含废弃的旧代码（引用新的 afml 包）
- 主入口文件 `src/afml_polars_pipeline.py` 与核心库混在一起
- 项目根目录有多个脚本入口，结构不清晰

## Goals / Non-Goals

**Goals:**
- 移除所有废弃代码，保持项目整洁
- 分离示例代码与核心库
- 简化项目目录结构

**Non-Goals:**
- 不修改任何核心库代码逻辑
- 不添加新功能
- 不改变 API 接口

## Decisions

1. **删除 legacy 目录**
   - `src/legacy/` 中的脚本已废弃，引用新的 afml 包
   - 删除整个目录及其所有文件

2. **移动入口文件到 examples/**
   - 将 `src/afml_polars_pipeline.py` 移动到 `examples/afml_polars_pipeline.py`
   - 保持代码不变，仅改变位置

3. **更新 pyproject.toml**
   - 更新 scripts 配置指向新位置

## Risks / Trade-offs

- **风险**: 可能有用户依赖 legacy 脚本
  -  mitigation: Legacy 脚本已引用新 afml 包，说明已被新实现取代
  - Legacy 目录中的代码使用 Pandas，新版本使用 Polars，性能提升显著

## Open Questions

无
