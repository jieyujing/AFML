返回

目录

- [JQData试用及购买](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10868)
- [JQData使用指南](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10031)
- [JQData安装/登录/流量查询/查看账号权限](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10748)
- [JQData常见报错](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10749)
- [JQData数据范围及更新时间](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10261)
- [JQData数据处理规则 ⭐](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10276)
- [全市场通用 ⭐](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=9836)
- [沪深A股](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=9842)
- [股票-单季度/年度财务数据（含新接口）](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=9878)
- [股票-报告期财务数据⭐](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=9892)
- [上市公司相关信息](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10006)
- [期货](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=9903)
- [期权](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=9913)
- [基金](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=9926)
- [指数⭐](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=9927)
- [债券（含可转债⭐）](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=9928)
- [Tick数据](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=9960)
- [资金流因子](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10664)
- [舆情数据](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=9961)
- [风险模型 \- 风格因子（CNE5）⭐](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10446)
- [风险模型-风格因子pro（CNE6）⭐](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10634)
- [聚宽因子⭐](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=9962)
- [alpha101和alpha191](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=9963)
- [技术指标](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=9964)
- [宏观数据](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=9965)

### [JQData使用指南](https://www.joinquant.com/help/api/doc?name=logon&id=9833)

试用账号历史范围：前15个月~前3个月 ; [正式账号历史范围：不限制](https://www.joinquant.com/help/api/doc?name=logon&id=9831)

**学习材料**

- [聚宽-get\_price和get\_bars处理规则](https://docs.qq.com/doc/DRWFudW9HSFJKbkN5)
- [聚宽-复权说明](https://docs.qq.com/doc/DU0hWUnZlSEZiRWZC)

**描述**

- [**JQData的申请试用流程**](https://www.joinquant.com/help/api/doc?name=logon&id=9830)
- [**登录/安装/流量查询/账号权限说明**](https://www.joinquant.com/help/api/doc?name=logon&id=9823)
- [**JQDara查询账号权限（新）**](https://www.joinquant.com/help/api/doc?name=logon&id=10532)
- [**JQData数据范围及接口更新时间**](https://www.joinquant.com/help/api/doc?name=logon&id=9824)

- [**JQData的试用规则**](https://www.joinquant.com/help/api/doc?name=logon&id=9830)
- [**保密协议**](https://www.joinquant.com/help/api/doc?name=logon&id=9825)
- [**链接的定义**](https://www.joinquant.com/help/api/doc?name=logon&id=10263)
- [**使用sdk常见报错**](https://www.joinquant.com/help/api/doc?name=logon&id=10262)

- [**官网VIP和本地数据的区别**](https://www.joinquant.com/help/api/doc?name=logon&id=9828)

**query使用方法**

  - 注意：query函数的更多用法详见： [**Query的简单教程**](https://www.kdocs.cn/l/cgLJ9Kpu2M79)
  - **run\_query** 查询数据库中的数据：为了防止返回数据量过大, 我们每次 **最多返回5000行**，不支持进行连表查询， **即同时查询多张表的数据**；

  - 但我们为了方便用户批量获取，提供了 [**run\_offset\_query**](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10524) 的获取方法， **最多返回20万条数据**；不过随着offset值的增大，查询性能是递减的 ，如查询超过上限，返回数据可能不完整。
  - 查询财务数据时如果提示没有定义，请在最上面添加：from jqdatasdk import \*

#### 新接口

|     |     |     |     |
| --- | --- | --- | --- |
| 模块 | 数据名称 | API接口 | 是否支持试用 |
| 通用接口 | [批量查询股票/期货/基金/期权/债券/宏观数据库](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10524) | run\_offset\_query | ✓ |
| 股票 | [股票1天/分钟行情数据](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=9874) | get\_price(round参数) | ✓ |
| 基金 | [基金1天/分钟行情数据](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=9998) | get\_price(round参数) | ✓ |
| 股票单季度 | [获取多个季度/年度的历史财务数据](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10326) | get\_history\_fundamentals | ✓ |
| [获取多个标的在指定交易日范围内的市值表数据](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10325) | get\_valuation | ✓ |
| 期货 | [获取期货合约的信息](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10324) | get\_futures\_info | ✓ |
| 风险模型 | [获取因子看板列表数据](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10450) | get\_factor\_kanban\_values | ✗ |
| [获取因子看板分位数历史收益率](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10451) | get\_factor\_stats |
| [获取风格因子暴露收益率](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10318) | get\_factor\_style\_returns |
| [获取特异收益率（无法被风格因子解释的收益）](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10319) | get\_factor\_specific\_returns |
| alpha因子 | [批量获取alpha101因子](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10534) | get\_all\_alpha\_101 |
| [批量获取alpha191因子](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10535) | get\_all\_alpha\_191 |
| 债券 | [可转债交易标的列表](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10340) | get\_all\_securities |
| [1天/分钟行情数据](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10335) | get\_price |
| [指定时间周期的分钟/日行情](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10336) | get\_bars |
| [可转债Tick数据](https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=10337) | get\_ticks |

- 在线客服


![](https://xlx03.qiyukf.net/463a7f7a19b9c07a71b61506233233d7.jpg?imageView&thumbnail=300x300)