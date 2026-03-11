---
name: jq-skills
description: 聚宽 (JoinQuant) 量化交易平台 Python SDK 完整专家文档 (V1.4 融合增强版)
---

# 聚宽 (JoinQuant) 量化交易 SDK 专家指南

本 Skill 结合了高频 API 精华总结与 15 篇全量技术文档索引。

## 📚 快速参考与全量索引

### 1. 核心框架 (快速开始)
- [策略程序架构](./references/快速开始/策略程序架构.md) - 初始化、计划任务、生命周期。
- [上下文对象 Context](./references/快速开始/Context对象.md) - 账户资金、持仓、g 对象。
- **🔍 深度参考**：`./references/full_docs/joinquant_api.md`

### 2. 数据获取 (全量数据)
- [行情获取函数](./references/行情数据/获取行情数据.md) - `get_price`, `history`, `get_bars` 等。
- [证券信息与成份股](./references/行情数据/标的查询与分类.md) - `get_all_securities`, `get_index_stocks`。
- [基本面与财务库](./references/财务与基本面/财务指标与市值.md) - `get_fundamentals` 详解。
- **🔍 深度参考**：
    - 股票/指数/基金：`joinquant_stock.MD`, `joinquant_index.md`, `joinquant_fund.md`
    - 财务/JQData：`joinquant_factor_values.md`, `joinquant_JQDatadoc.md`
    - 宏观/板块：`joinquant_macroData.md`, `joinquant_platedata.md`

### 3. 交易执行 (全功能订单)
- [普通订单系统](./references/交易函数/订单处理与撮合.md) - `order` 全系列函数、撤单逻辑。
- [融资融券交易专页](./references/交易函数/融资融券交易.md) - `margincash_open` 等。
- [交易成本与滑点](./references/交易函数/交易成本与滑点.md) - `set_order_cost`, `set_slippage`。
- **🔍 深度参考**：`./references/full_docs/joinquant_api.md`

### 4. 技术分析与因子
- [技术分析指标库](./references/技术指标与因子/技术分析指标.md) - `jqlib`, `TA-Lib` 集成。
- [Alpha 因子与因子库](./references/技术指标与因子/Alpha因子库.md) - Alpha101/191 详解。
- **🔍 深度参考**：`joinquant_Alpha101.md`, `joinquant_Alpha191.md`, `joinquant_technicalanalysis.md`

### 5. 高级模块与特定品种
- [投资组合优化器](./references/高级功能/投资组合优化器.md) - Target 函数与 Constraints 约束。
- [期货与期权专题](./references/高级功能/期货策略专用.md) - 主力合约、保证金。
- **🔍 深度参考**：`joinquant_Future.md`, `joinquant_Option.md`, `joinquant_bond.md`

---

## 🛠️ 调用逻辑要求
1. **常规咨询**：优先查阅上述第一层级（快速参考）的 MD 文件。
2. **精准编程**：涉及具体参数名、默认值、返回结构或复杂因子公式时，**必须**进一步读取 `./references/full_docs/` 对应的源文件。
3. **禁忌检查**：始终核对 `⚠️ 专家级避坑准则`。

## ⚠️ 专家级避坑准则
1. **持久化**: `g` 不存 `query` 和 `open` 对象。
2. **未来函数**: 严禁在 9:00 前获取当日 `get_fundamentals`。
3. **频率陷阱**: `history` 取天数据时不包含当天！

## 📖 交互示例
- "帮我写一个聚宽策略：市值最小 50 支，ROE > 10%。(参考 joinquant_JQDatadoc.md 确认财务字段)"
- "根据 joinquant_Alpha191.md 实现第 50 号因子计算。"
