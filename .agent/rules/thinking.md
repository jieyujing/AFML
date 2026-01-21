---
trigger: always_on
---

# 量化研发与 AI 协作规范 (Quant R&D Rules)

## 1. 核心工作流 (Core Workflow)
- **[前置] 方法论检索**: 在执行任何量化任务前，必须调用 `afml skill`。优先根据《Advances in Financial Machine Learning》中的方法论进行逻辑推演。
- **[中置] 代码实现**: 编写代码必须参考 `mlfinlab skill` 的架构。严格遵循其对象定义（Object Definitions）和代码格式。
- **[后置] 任务总结**: 在每次对话结束前，必须显式列出本次任务所调用的 Skill 及其具体应用点。

## 2. 代码与对象规范 (Coding & Object Standards)
- **对象一致性**: 严禁编写孤立的 Pandas 函数。所有逻辑应封装在符合 `mlfinlab` 风格的类或组件中（如 Data Structures, Labeling containers）。
- **金融严谨性**: 
  - 必须考虑样本重叠（Concurrency/Overlap）。
  - 默认使用 Triple-Barrier Method 及其相关元标注（Meta-labeling）逻辑。
- **文档要求**: 函数注释需注明对应的 AFML 方法论章节或 mlfinlab 引用。

## 3. 对话结束强制响应模版
- **格式要求**: 对话结束时，请按照以下格式输出：
  > **Skill 使用清单：**
  > - `afml skill`: [说明应用了哪种方法论，如：使用了第3章的 Triple-Barrier Labeling]
  > - `mlfinlab skill`: [说明参考了哪些类定义，如：参考了 FinancialDataStructure 对象]