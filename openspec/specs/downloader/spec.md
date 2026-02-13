# downloader Specification

## Purpose
TBD - created by archiving change binance-data-downloader. Update Purpose after archive.
## Requirements
### Requirement: 命令行参数解析

程序 SHALL 支持以下命令行参数：

#### Scenario: 帮助信息

- **WHEN** 用户运行 `python src/binance_downloader.py --help`
- **THEN** 显示帮助信息，包括用法、参数说明和示例
- **AND** 退出码为 0

#### Scenario: 指定日期范围下载

- **WHEN** 用户运行 `python src/binance_downloader.py --start 2026-02-01 --end 2026-02-10`
- **THEN** 下载 2026-02-01 到 2026-02-10 之间的所有数据文件
- **AND** 每个文件下载完成后打印进度

#### Scenario: 不指定结束日期（默认今天）

- **WHEN** 用户只提供 --start 参数，不提供 --end
- **THEN** 结束日期默认为当天日期
- **AND** 下载从 start 到今天的所有数据

#### Scenario: 增量回补模式

- **WHEN** 用户运行 `python src/binance_downloader.py --start 2026-02-01 --end 2026-02-10 --backfill`
- **THEN** 程序扫描 data/ 目录，识别缺失的日期
- **AND** 只下载缺失日期的文件，跳过已存在的文件

---

### Requirement: 断点续传

程序 MUST 能够检测已下载的文件，避免重复下载。

#### Scenario: 检测已存在文件

- **WHEN** 目标日期范围内已有部分文件
- **THEN** 跳过已存在的文件，只下载缺失的
- **AND** 打印 "Skipping existing file: BTCUSDT-aggTrades-2026-02-05.zip"

#### Scenario: 下载完成提示

- **WHEN** 所有文件下载完成
- **THEN** 打印 "Download complete: X files, Y skipped"

---

### Requirement: 下载重试

下载失败时 MUST 自动重试。

#### Scenario: 下载失败重试

- **WHEN** 下载过程中网络错误或其他异常
- **THEN** 自动重试最多 3 次
- **AND** 每次重试前等待 2 秒

#### Scenario: 重试耗尽

- **WHEN** 重试 3 次后仍然失败
- **THEN** 打印错误信息并继续下一个文件
- **AND** 不中断整个下载流程

---

