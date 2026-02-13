## Why

下载的 zip 文件可能因网络问题损坏，需要添加完整性校验功能确保数据正确。

## What Changes

- 下载时自动下载对应的 .CHECKSUM 文件
- 下载完成后校验文件完整性
- 校验失败时自动删除损坏文件并提示重新下载

## Capabilities

### New Capabilities
- `checksum-verification`: 下载并校验文件完整性

### Modified Capabilities
- `binance-downloader`: 现有下载功能扩展

## Impact

- `src/binance_downloader.py`: 新增校验逻辑
