# Context 与对象模型

## 1. Context 对象
`context` 是策略中最重要的只读对象，记录了当前的资金和时间状态。

### 核心属性
- **subportfolios**: 子账户列表。`context.subportfolios[0]` 为主账户。
- **portfolio**: [Portfolio对象](#2-portfolio-对象)。兼容性属性，指向 `subportfolios[0]`。
- **current_dt**: 当前逻辑时间 (datetime.datetime)。
- **previous_date**: 前一个交易日 (datetime.date)。
- **universe**: [set_universe] 设定的股票池。

---

## 2. Portfolio 对象
通过 `context.portfolio` 访问。

| 属性 | 描述 |
| :--- | :--- |
| cash | 可用现金 |
| available_cash | 等同于 cash |
| total_value | 总资产 (持仓市值 + 现金) |
| positions_value | 总持仓市值 |
| positions | 一个字典，包含所有 [Position对象](#3-position-对象) |
| locked_cash | 冻结资金 (挂单中) |
| returns | 策略建立以来的累计收益率 |

---

## 3. Position 对象
通过 `context.portfolio.positions[code]` 访问。

| 属性 | 描述 |
| :--- | :--- |
| security | 标的代码 |
| price | 当前价格 |
| total_amount | 总仓位 (股) |
| closeable_amount | 可卖仓位 (T+1 限制下的可用量) |
| avg_cost | 开仓均价 (包含佣金) |
| init_time | 最初建仓时间 |

---

## 4. SecurityUnitData 对象
在 `handle_data(context, data)` 中，`data[code]` 返回此对象。

- **open / close / high / low**: 上一个单位时间的价格。
- **volume**: 上一个单位时间的成交量。
- **money**: 上一个单位时间的成交额。
- **paused**: 是否停牌。
- **high_limit / low_limit**: 涨停/跌停价。
