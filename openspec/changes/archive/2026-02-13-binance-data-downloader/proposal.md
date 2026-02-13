## Why

Binance 提供历史合约聚合交易数据（aggTrades），但需要手动下载。用户需要一个工具来批量下载指定日期范围的历史数据，并支持断点续传避免重复下载。还需要增量更新功能，补充缺失日期的数据。

## What Changes

- 新增 `src/binance_downloader.py` 模块
- 支持命令行参数指定开始/结束日期
- 自动检测已下载文件，跳过重复下载
- 下载失败时支持重试
- 支持增量更新：对比本地已有文件与目标日期范围，补充缺失的日期

## Capabilities

### New Capabilities
- `download_agg_trades`: 根据日期范围下载 BTCUSDT 聚合交易数据
- `resume_support`: 检测 data/ 目录下已存在的文件，避免重复下载
- `backfill_missing`: 扫描目标日期范围，识别缺失的日期并补充下载

### Modified Capabilities
- 无

## Impact

- `src/binance_downloader.py`: 新增下载工具（含回补功能）
- `data/`: 保存下载的 zip 文件
