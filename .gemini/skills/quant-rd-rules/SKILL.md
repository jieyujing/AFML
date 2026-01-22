---
name: quant-rd-rules
description: Enforces Quant R&D Rules for methodology retrieval, code standards, and mandatory reporting. Use this skill for any quantitative finance task, ensuring AFML methodology and MLFinLab standards are followed.
---

# Quant R&D Rules (量化研发与 AI 协作规范)

## 1. Core Workflow (核心工作流)

- **[Pre-task] Methodology Retrieval ([前置] 方法论检索)**
  - Before executing any quantitative task, you MUST call `afml skill`.
  - Prioritize logical deduction based on methodologies from *Advances in Financial Machine Learning*.

- **[Mid-task] Code Implementation ([中置] 代码实现)**
  - Code MUST be written with reference to the `mlfinlab skill` architecture.
  - Strictly adhere to its Object Definitions and code formatting.

- **[Mid-task] Version Control ([中置] 版本控制)**
  - Whenever a feature is completed and tests pass, you MUST add a git commit.

- **[Post-task] Task Summary ([后置] 任务总结)**
  - At the end of every conversation, you MUST explicitly list the Skills used in this task and their specific application points.

## 2. Coding & Object Standards (代码与对象规范)

- **Object Consistency (对象一致性)**
  - DO NOT write isolated Pandas functions.
  - All logic should be encapsulated in classes or components consistent with `mlfinlab` style (e.g., Data Structures, Labeling containers).

- **Financial Rigor (金融严谨性)**
  - MUST consider sample concurrency/overlap.
  - Default to using the **Triple-Barrier Method** and its associated **Meta-labeling** logic.

- **Documentation Requirements (文档要求)**
  - Function docstrings MUST cite the corresponding AFML methodology chapter or mlfinlab reference.

## 3. Mandatory Response Template (对话结束强制响应模版)



**Format Requirement:**

At the end of the conversation, output the following format:



> **Skill Usage List:**

> - `afml skill`: [Description of methodology applied, e.g., "Used Chapter 3 Triple-Barrier Labeling"]

> - `mlfinlab skill`: [Description of class definitions referenced, e.g., "Referenced FinancialDataStructure object"]



## 4. Project Tracking (项目进度追踪)



- **Context Awareness (上下文感知)**

  - At the start of a session, check `PROGRESS.md` (if available) to understand the current project state.

  - Review `PROGRESS.md` to identify completed tasks and recommended next steps.



- **Progress Recording (进度记录)**

  - After completing a significant milestone (e.g., new feature, successful test, generated data), update `PROGRESS.md`.

  - Ensure `PROGRESS.md` maintains a "Completed Work", "Current Status", and "Next Steps" structure.