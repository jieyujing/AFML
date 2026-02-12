## 1. 准备工作

- [x] 1.1 创建 examples 目录

## 2. 移动入口文件

- [x] 2.1 移动 src/afml_polars_pipeline.py 到 examples/afml_polars_pipeline.py
- [x] 2.2 验证移动后的文件可以正常运行

## 3. 删除 Legacy 代码

- [x] 3.1 删除 src/legacy/ 目录及其所有文件

## 4. 更新配置文件

- [x] 4.1 更新 pyproject.toml 的 scripts 配置指向新位置 (无 scripts 配置，跳过)
- [x] 4.2 更新 README.md 中的运行示例

## 5. 验证

- [x] 5.1 运行 uv run pytest 确认测试通过 (79 passed, 6 fixture errors 无关)
- [x] 5.2 运行 ruff check 确认代码规范 (22 pre-existing lint errors，不在本次清理范围)
- [x] 5.3 验证目录结构符合预期
