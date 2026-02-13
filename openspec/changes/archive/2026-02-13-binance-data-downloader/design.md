---

## Context

Binance Vision 提供历史聚合交易数据，URL 格式为：
`https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-YYYY-MM-DD.zip`

用户需要批量下载指定日期范围的数据。

## Goals / Non-Goals

**Goals:**
- 支持日期范围下载
- 支持断点续传（跳过已存在文件）
- 支持增量回补（--backfill 模式）
- 下载失败自动重试
- --end 参数默认为今天

**Non-Goals:**
- 解压 zip 文件（仅下载）
- 支持其他交易对（仅 BTCUSDT）
- 支持其他数据类型（仅 aggTrades）

## Decisions

### Decision 1: 目录结构

- 下载文件保存到 `data/` 目录
- 使用 `argparse` 处理命令行参数

### Decision 2: 重试机制

- 使用指数退避策略（可选）或固定等待时间
- 最大重试次数：3 次
- 每次重试等待 2 秒

### Decision 3: 日期处理

- 使用 `datetime` 模块处理日期
- 日期格式：`YYYY-MM-DD`
- 生成日期列表使用 `datetime.timedelta`

---
