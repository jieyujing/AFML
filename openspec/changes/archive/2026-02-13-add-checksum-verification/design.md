## Context

现有 `src/binance_downloader.py` 支持下载 BTCUSDT aggTrades 数据，但缺少完整性校验。Binance Vision 提供 .CHECKSUM 文件，格式为 SHA256。

## Goals / Non-Goals

**Goals:**
- 下载 zip 文件后自动下载对应的 .CHECKSUM 文件
- 使用 SHA256 校验文件完整性
- 校验失败时删除损坏文件并提示用户

**Non-Goals:**
- 不支持其他校验算法（仅 SHA256）
- 不解压 zip 文件

## Decisions

### Decision 1: 校验时机

在每个 zip 文件下载完成后立即校验，而不是批量下载后校验。

**理由**：早期失败可以避免浪费带宽下载后续文件。

### Decision 2: 校验失败处理

校验失败时删除损坏的 zip 文件，标记下载失败，用户可使用 --retry 重新下载。

**理由**：保留损坏文件会干扰后续断点续传逻辑。

### Decision 3: CHECKSUM URL 格式

CHECKSUM 文件 URL = 数据文件 URL + ".CHECKSUM"

**示例**：
- 数据：`BTCUSDT-aggTrades-2026-02-11.zip`
- 校验：`BTCUSDT-aggTrades-2026-02-11.zip.CHECKSUM`

## Risks / Trade-offs

- **风险**：CHECKSUM 文件下载失败导致无法校验
  - **缓解**：CHECKSUM 文件很小，重试机制应能处理
- **权衡**：额外的网络请求增加下载时间，但确保数据完整性
